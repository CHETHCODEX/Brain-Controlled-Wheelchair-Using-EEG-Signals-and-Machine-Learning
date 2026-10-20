"""Bounded native Webots scene validation; --launch runs an isolated temporary world."""

import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def validate():
    import zmq
    from src.ipc.command_schema import make_command
    from src.ipc.zmq_publisher import intent_probabilities
    from src.webots_controller.safety_override import SafetyController
    from src.webots_controller.wheelchair_driver import MOTORS, SURROUND_SENSORS, drive_step

    robot = importlib.import_module("controller").Supervisor()
    timestep = int(robot.getBasicTimeStep())
    try:
        sensors = [robot.getDevice(name) for name in SURROUND_SENSORS]
        if any(sensor is None for sensor in sensors):
            raise ValueError("Missing surround sensor")
        for name in MOTORS:
            motor = robot.getDevice(name)
            motor.setPosition(float("inf"))
            motor.setVelocity(0)
        for sensor in sensors:
            sensor.enable(timestep)
        for step in range(90):
            if robot.step(timestep) == -1:
                return
        readings = {name: sensor.getValue() for name, sensor in zip(SURROUND_SENSORS, sensors)}
        if not all(math.isfinite(value) and 0 < value <= 2.01 for value in readings.values()):
            raise ValueError("Invalid stationary sensor readings: " + str(readings))
        position = robot.getSelf().getPosition()
        if abs(position[0] + 7) > 0.1 or abs(position[1]) > 0.1 or not 0.1 < position[2] < 0.2:
            raise ValueError("Wheelchair starting pose is unstable: " + str(position))
        output = ROOT / "artifacts" / "hospital-webots.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        robot.exportImage(str(output), 100)
        class Inbox:
            pending = None

            def recv(self, flags=0):
                if self.pending is None:
                    raise zmq.Again()
                message, self.pending = self.pending, None
                return message

        inbox, safety = Inbox(), SafetyController()
        motors = [robot.getDevice(name) for name in MOTORS]
        sequence = 0

        def phase(intent, steps):
            nonlocal sequence
            for step in range(steps):
                if step % 16 == 0:
                    inbox.pending = json.dumps(make_command(intent_probabilities(intent), sequence)).encode()
                    sequence += 1
                drive_step(motors, sensors, inbox, safety, dt=timestep / 1000)
                if robot.step(timestep) == -1:
                    raise RuntimeError("Simulator terminated during motion validation")
            matrix = robot.getSelf().getOrientation()
            return robot.getSelf().getPosition(), math.atan2(matrix[3], matrix[0])

        forward_position, initial_yaw = phase("FORWARD", 130)
        phase("REST", 80)
        _, left_yaw = phase("TURN_LEFT", 100)
        phase("REST", 80)
        _, right_yaw = phase("TURN_RIGHT", 100)
        stopped_position, _ = phase("REST", 80)
        reverse_position, _ = phase("BACKWARD", 130)
        phase("REST", 80)
        before_emergency = robot.getSelf().getPosition()
        safety.emergency_stop()
        after_emergency, _ = phase("FORWARD", 50)
        motion = {"forward_dx": forward_position[0] - position[0],
                  "left_yaw_delta": left_yaw - initial_yaw,
                  "right_yaw_delta": right_yaw - left_yaw,
                  "reverse_dx": reverse_position[0] - stopped_position[0],
                  "emergency_displacement": math.dist(before_emergency, after_emergency)}
        if not (motion["forward_dx"] > 0.15 and motion["left_yaw_delta"] > 0.3
                and motion["right_yaw_delta"] < -0.3 and motion["reverse_dx"] < -0.1
                and motion["emergency_displacement"] < 0.05):
            raise ValueError("Native motion validation failed: " + json.dumps(motion))
        report = {"passed": True, "sensors": readings, "position": position,
                  "motion": motion, "screenshot": str(output)}
        (ROOT / "artifacts/hospital-validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("SCENE_VALIDATION_PASS " + json.dumps(report), flush=True)
        robot.simulationQuit(0)
    except Exception as error:
        print("SCENE_VALIDATION_FAIL " + str(error), flush=True)
        robot.simulationQuit(1)


def launch():
    executable = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Webots/msys64/mingw64/bin/webots.exe"
    if not executable.exists():
        raise FileNotFoundError("Webots not found; run this validation on a machine with Webots installed")
    source = (ROOT / "simulation/worlds/hospital_rehab.wbt").read_text(encoding="utf-8")
    original = 'Wheelchair { translation -7 0 0.15 }'
    if source.count(original) != 1:
        raise ValueError("Hospital world must contain exactly one expected starting wheelchair")
    temporary = None
    report_path = ROOT / "artifacts/hospital-validation.json"
    report_path.unlink(missing_ok=True)
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".wbt", prefix="validation_",
                                         dir=ROOT / "simulation/worlds", delete=False, encoding="utf-8") as world:
            world.write(source.replace(original, 'Wheelchair { translation -7 0 0.15 controller "scene_validator" supervisor TRUE }'))
            temporary = Path(world.name)
        result = subprocess.run([str(executable), "--batch", "--mode=fast", "--minimize",
                                 "--port=1239", "--stdout", "--stderr", str(temporary)],
                                capture_output=True, text=True, timeout=75)
        print(result.stdout)
        print(result.stderr)
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        if result.returncode or not report.get("passed"):
            raise RuntimeError("Native Webots scene validation failed")
        print("SCENE_VALIDATION_PASS " + json.dumps(report))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
            temporary.with_name("." + temporary.stem + ".wbproj").unlink(missing_ok=True)


if __name__ == "__main__":
    launch() if "--launch" in sys.argv else validate()