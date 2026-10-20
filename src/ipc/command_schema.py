"""Validate untrusted JSON command fields before simulator actuation."""

import math
import time

INTENTS = ("REST", "TURN_LEFT", "TURN_RIGHT", "FORWARD")
NAVIGATION_INTENTS = INTENTS + ("BACKWARD",)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_command(packet, wall_time=None):
    now = time.time() if wall_time is None else wall_time
    if not isinstance(packet, dict):
        raise ValueError("Command must be a JSON object")
    sequence = packet.get("sequence_id")
    if type(sequence) is not int or sequence < 0:
        raise ValueError("sequence_id must be a non-negative integer")
    timestamp = packet.get("timestamp")
    if not finite_number(timestamp) or not -0.25 <= now - timestamp <= 1.0:
        raise ValueError("Command timestamp is stale or in the future")
    intent = packet.get("intent")
    if not isinstance(intent, str) or intent not in NAVIGATION_INTENTS:
        raise ValueError("Unknown navigation intent")
    if type(packet.get("safety_flag")) is not bool:
        raise ValueError("safety_flag must be boolean")
    probabilities = packet.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) not in (set(INTENTS), set(NAVIGATION_INTENTS)):
        raise ValueError("Four legacy or five navigation probabilities are required")
    if intent not in probabilities:
        raise ValueError("Selected intent must be present in the probability vector")
    if not all(finite_number(value) and 0 <= value <= 1 for value in probabilities.values()):
        raise ValueError("Probabilities must be finite and between zero and one")
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6):
        raise ValueError("Probabilities must sum to one")
    confidence = packet.get("confidence")
    if not finite_number(confidence) or not math.isclose(confidence, probabilities[intent], abs_tol=1e-6):
        raise ValueError("Confidence must equal the selected intent probability")
    if confidence < max(probabilities.values()) - 1e-6:
        raise ValueError("Intent must match the largest probability")
    quality = packet.get("signal_quality", 1.0)
    if not finite_number(quality) or not 0 <= quality <= 1:
        raise ValueError("Signal quality must be between zero and one")
    if type(packet.get("headset_connected", True)) is not bool:
        raise ValueError("Headset connection status must be boolean")
    session = packet.get("session_id")
    if session is not None and (not isinstance(session, str) or not 1 <= len(session) <= 64):
        raise ValueError("Session ID must be a nonempty string of at most 64 characters")
    if type(packet.get("caregiver_reset", False)) is not bool:
        raise ValueError("Caregiver reset must be boolean")
    return packet


def make_command(probabilities, sequence, wall_time=None, *, session_id=None,
                 source="legacy", signal_quality=1.0, headset_connected=True,
                 safety_flag=False, caregiver_reset=False):
    intent = max(probabilities, key=probabilities.get)
    packet = {
        "timestamp": time.time() if wall_time is None else wall_time,
        "sequence_id": sequence, "intent": intent,
        "confidence": probabilities[intent], "probabilities": dict(probabilities),
        "safety_flag": safety_flag, "source": source, "signal_quality": signal_quality,
        "headset_connected": headset_connected, "caregiver_reset": caregiver_reset,
    }
    if session_id is not None:
        packet["session_id"] = session_id
    return validate_command(packet, wall_time)