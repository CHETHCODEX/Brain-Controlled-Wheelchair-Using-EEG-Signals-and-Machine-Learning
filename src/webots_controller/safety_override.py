"""Pure-Python command debounce, watchdog, and differential-drive safety."""

import math
import time

from src.ipc.command_schema import validate_command


class SafetyController:
    def __init__(self):
        self.intent = "REST"
        self.candidate = None
        self.consecutive = 0
        self.last_sequence = -1
        self.last_received = None
        self.reason = "waiting_for_command"
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.emergency_latched = False
        self.session = None
        self.retired_sessions = set()
        self.confidence = 0.0
        self.signal_quality = 0.0
        self.detected_intent = "REST"
        self.source = "none"
        self.obstacle_locks = set()

    def stop(self, reason, immediate=True):
        self.intent = "REST"
        self.candidate = None
        self.consecutive = 0
        self.reason = reason
        if immediate:
            self.linear_velocity = 0.0
            self.angular_velocity = 0.0

    def emergency_stop(self, active=True):
        self.emergency_latched = active
        self.stop("emergency_stop" if active else "caregiver_reset")

    def accept(self, packet, now=None, wall_time=None):
        now = time.monotonic() if now is None else now
        try:
            validate_command(packet, wall_time)
            session = packet.get("session_id", "legacy")
            if session in self.retired_sessions:
                raise ValueError("Retired publisher session")
            if self.session is not None and session != self.session:
                if len(self.retired_sessions) >= 64:
                    raise ValueError("Too many publisher sessions; restart controller")
                self.retired_sessions.add(self.session)
                self.last_sequence = -1
                self.stop("publisher_changed")
            self.session = session
            if packet["sequence_id"] <= self.last_sequence:
                raise ValueError("Duplicate or out-of-order command")
        except (ValueError, TypeError, OverflowError):
            self.stop("invalid_command")
            return False
        self.last_sequence = packet["sequence_id"]
        self.last_received = now
        self.detected_intent = packet["intent"]
        self.confidence = packet["confidence"]
        self.signal_quality = packet.get("signal_quality", 1.0)
        self.source = packet.get("source", "legacy")
        if packet["safety_flag"]:
            self.emergency_stop()
        if self.emergency_latched:
            self.stop("emergency_stop")
            return True
        if not packet.get("headset_connected", True) or self.signal_quality < 0.7:
            self.stop("signal_lost_or_poor")
            return True
        ordered = sorted(packet["probabilities"].values(), reverse=True)
        if packet["confidence"] < 0.65 or ordered[0] - ordered[1] < 0.15:
            self.stop("rest_or_low_confidence")
            return True
        if packet["intent"] == "REST":
            self.stop("neutral", immediate=False)
            return True
        intent = packet["intent"]
        self.consecutive = self.consecutive + 1 if intent == self.candidate else 1
        self.candidate = intent
        if self.consecutive >= 2:
            self.intent = intent
            self.reason = "command"
        return True

    def wheel_velocities(self, distances, now=None, dt=None):
        now = time.monotonic() if now is None else now
        if self.emergency_latched:
            self.stop("emergency_stop")
        elif self.last_received is None or now - self.last_received >= 1.0:
            self.stop("heartbeat_timeout")
        else:
            if isinstance(distances, dict):
                expected = {"front_left", "front_center", "front_right", "rear_left",
                            "rear_center", "rear_right", "side_left", "side_right"}
                valid = set(distances) == expected and all(
                    type(value) in (int, float) and math.isfinite(value) and value >= 0
                    for value in distances.values())
                front = [distances.get(name, 0) for name in ("front_left", "front_center", "front_right")]
                rear = [distances.get(name, 0) for name in ("rear_left", "rear_center", "rear_right")]
                turning = list(distances.values())
            else:
                valid = len(distances) == 3 and all(math.isfinite(value) and value >= 0 for value in distances)
                front, rear, turning = distances, [], distances
            stopping_margin = 0.35 + self.linear_velocity ** 2 / (2 * 0.65)
            if not valid:
                self.stop("sensor_fault")
            else:
                if min(front) > 0.6:
                    self.obstacle_locks.discard("FORWARD")
                if rear and min(rear) > 0.6:
                    self.obstacle_locks.discard("BACKWARD")
            if not valid:
                pass
            elif self.intent in self.obstacle_locks:
                self.stop("rear_obstacle" if self.intent == "BACKWARD" else "obstacle")
            elif self.intent == "BACKWARD" and not rear:
                self.stop("rear_sensor_missing")
            elif ((self.intent == "BACKWARD" or self.linear_velocity < -0.001)
                  and rear and min(rear) <= stopping_margin):
                self.obstacle_locks.add("BACKWARD")
                self.stop("rear_obstacle")
            elif ((self.intent == "FORWARD" or self.linear_velocity > 0.001)
                  and min(front) <= stopping_margin):
                self.obstacle_locks.add("FORWARD")
                self.stop("obstacle")
            elif ((self.intent in ("TURN_LEFT", "TURN_RIGHT") or abs(self.angular_velocity) > 0.001)
                  and min(turning) <= 0.35):
                self.stop("turn_clearance")
            elif min(front) <= 0.35 and not rear:
                self.stop("obstacle")
        linear, angular = {
            "REST": (0.0, 0.0), "FORWARD": (0.4, 0.0),
            "TURN_LEFT": (0.0, 0.8), "TURN_RIGHT": (0.0, -0.8),
            "BACKWARD": (-0.2, 0.0),
        }[self.intent]
        if dt is None:
            self.linear_velocity, self.angular_velocity = linear, angular
        else:
            if not math.isfinite(dt) or not 0 < dt <= 0.1:
                self.stop("invalid_control_timestep")
            else:
                if self.linear_velocity * linear < 0:
                    linear = 0.0
                acceleration = 0.35 if abs(linear) > abs(self.linear_velocity) else 0.65
                change = acceleration * dt
                self.linear_velocity += max(-change, min(change, linear - self.linear_velocity))
                turn_change = 1.2 * dt
                self.angular_velocity += max(-turn_change, min(turn_change, angular - self.angular_velocity))
        return ((self.linear_velocity - self.angular_velocity * 0.55 / 2) / 0.15,
                (self.linear_velocity + self.angular_velocity * 0.55 / 2) / 0.15)