"""The released version-2 protocol fingerprint (``reports/v2/protocol_v2.json``).

Kept apart from ``v2_release``: the hash of that module is part of the fingerprint, so the fingerprint
cannot live inside it. ``None`` only before the first release build.
"""

FROZEN_FINGERPRINT: str | None = "f13c2eedaba79948278a1004d1a1c71014d43c17e84e739e4c1cb0b4c8ab33e0"
