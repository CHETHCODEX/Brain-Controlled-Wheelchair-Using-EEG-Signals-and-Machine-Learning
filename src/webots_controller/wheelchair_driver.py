"""Nonblocking ZeroMQ adapter for the Webots simulation."""

import json
import math
import time

import zmq

from src.bci.runtime import EMERGENCY, TELEMETRY, read_state, write_state
from src.ipc.zmq_publisher import ENDPOINT
from src.webots_controller.safety_override import SafetyController

MOTORS = ("left wheel motor", "right wheel motor")
SENSORS = ("front_left", "front_center", "front_right")
SURROUND_SENSORS = SENSORS + ("rear_left", "rear_center", "rear_right", "side_left", "side_right")


def drive_step(motors, sensors, subscriber, safety, dt=None):
    for attempt in range(32):
        try:
            message = subscriber.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
        try:
            if len(message) > 4096:
                raise ValueError("Command exceeds message size limit")
            packet = json.loads(message)
        except (ValueError, UnicodeError, RecursionError):
            safety.stop("invalid_json")
            break
        if not safety.accept(packet):
            break
    readings = [sensor.getValue() for sensor in sensors]
    distances = dict(zip(SURROUND_SENSORS, readings)) if len(readings) == 8 else readings
    velocities = safety.wheel_velocities(distances, dt=dt)
    for motor, velocity in zip(motors, velocities):
        motor.setVelocity(velocity)
    return velocities


def run(robot):
    timestep = int(robot.getBasicTimeStep())
    motors = [robot.getDevice(name) for name in MOTORS]
    sensors = [robot.getDevice(name) for name in SURROUND_SENSORS]
    if any(device is None for device in motors + sensors):
        raise RuntimeError("Wheelchair motor or sensor names do not match the controller")
    for motor in motors:
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)
    for sensor in sensors:
        sensor.enable(timestep)
    gps = robot.getDevice("position")
    imu = robot.getDevice("orientation")
    if gps is not None:
        gps.enable(timestep)
    if imu is not None:
        imu.enable(timestep)
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    subscriber.setsockopt(zmq.CONFLATE, 1)
    subscriber.setsockopt(zmq.MAXMSGSIZE, 4096)
    subscriber.setsockopt(zmq.LINGER, 0)
    subscriber.connect(ENDPOINT)
    safety = SafetyController()
    previous_status = None
    previous_emergency = None
    last_telemetry = 0.0
    previous_position = None
    previous_simulation_time = None
    print("Simulation controller ready on " + ENDPOINT, flush=True)
    try:
        while robot.step(timestep) != -1:
            try:
                active = read_state(EMERGENCY)["active"] if EMERGENCY.exists() else False
                if type(active) is not bool:
                    raise ValueError("Invalid emergency state")
                if active != previous_emergency:
                    safety.emergency_stop(active)
                    previous_emergency = active
            except (OSError, ValueError, KeyError):
                safety.emergency_stop()
                previous_emergency = True
            velocities = drive_step(motors, sensors, subscriber, safety, timestep / 1000)
            if time.monotonic() - last_telemetry >= 0.1:
                position = gps.getValues() if gps is not None else None
                simulation_time = robot.getTime()
                measured_speed = None
                if position is not None and previous_position is not None and simulation_time > previous_simulation_time:
                    measured_speed = math.hypot(position[0] - previous_position[0],
                                                position[1] - previous_position[1]) / (simulation_time - previous_simulation_time)
                previous_position, previous_simulation_time = position, simulation_time
                payload = {
                    "timestamp": time.time(), "engine": "Webots", "intent": safety.intent,
                    "detected_intent": safety.detected_intent, "confidence": safety.confidence,
                    "signal_quality": safety.signal_quality, "source": safety.source,
                    "linear_velocity": safety.linear_velocity, "angular_velocity": safety.angular_velocity,
                    "measured_speed": measured_speed,
                    "wheel_velocities": list(velocities), "reason": safety.reason,
                    "emergency_stop": safety.emergency_latched,
                    "distances": {name: sensor.getValue() for name, sensor in zip(SURROUND_SENSORS, sensors)},
                    "position": position,
                    "yaw": imu.getRollPitchYaw()[2] if imu is not None else None,
                }
                try:
                    write_state(TELEMETRY, payload)
                except (OSError, ValueError):
                    safety.stop("telemetry_fault")
                    for motor in motors:
                        motor.setVelocity(0.0)
                last_telemetry = time.monotonic()
            status = (safety.intent, safety.reason)
            if status != previous_status:
                print(f"Wheelchair: {status[0]} ({status[1]})", flush=True)
                previous_status = status
    finally:
        for motor in motors:
            motor.setVelocity(0.0)
        subscriber.close()
        context.term()