"""
damage_analytics.py
===================
Generates structured damage report + health score.

Attribution strategy (per damage type):
  - tire flat   → only tyre parts
  - lamp broken → only headlight / taillight parts
  - glass shatter → only glass parts
  - dent        → body parts only (no glass, no lamps, no tyre)
  - scratch     → body parts only (no glass)
  - crack       → any named part

For every damage detection that passes validation, we compute the overlap
fraction with each named car part.  Any part that meets the minimum
MIN_ATTRIBUTION_OVERLAP threshold AND belongs to the allowed set for that
damage type is credited in the report.  This means:

  • A scratch genuinely spanning fender AND door → two report rows (correct).
  • A tire-flat mask touching front-bumper     → bumper is excluded by type
    rule, only tyre is reported (correct).
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))\

from config import SEVERITY_RULES, SEVERITY_PENALTY, MIN_EXTERNAL_PIXELS

# ── Part sets (lower-cased) ───────────────────────────────────────────────────
_GLASS_PARTS = {
    'diggi_back_door_glass', 'front_windshield_glass',
    'left_front_door_glass', 'left_rear_door_glass',
    'right_front_door_glass', 'right_rear_door_glass',
}
_LAMP_PARTS = {'left_headlight', 'left_taillight', 'right_headlight', 'right_taillight'}
_TYRE_PARTS = {'tyre'}

# Allowed attribution parts per damage class index
# None means "all named parts are allowed"
_ATTRIBUTION_ALLOWED = {
    0: None,                                # crack  → any part
    1: None,                                # dent   → any part, but exclude invalid sets below
    2: _GLASS_PARTS,                        # glass shatter → glass only
    3: _LAMP_PARTS,                         # lamp broken   → lamp only
    4: None,                                # scratch → any part, exclude glass below
    5: _TYRE_PARTS,                         # tire flat → tyre only
}

# Parts that must be EXCLUDED from attribution for a damage type
# (applied after the "allowed" filter)
_ATTRIBUTION_EXCLUDED = {
    0: set(),                                             # crack
    1: _GLASS_PARTS | _LAMP_PARTS | _TYRE_PARTS,         # dent
    2: set(),                                             # glass shatter
    3: set(),                                             # lamp broken
    4: _GLASS_PARTS,                                      # scratch — no glass
    5: set(),                                             # tire flat
}

# Minimum fraction of the damage mask that must overlap a part for it
# to be credited in the report.  Lower than the validation threshold so
# that genuine multi-part damage (e.g. 25 % on each of two door panels)
# is still captured.
MIN_ATTRIBUTION_OVERLAP = 0.15


def mask_area(mask):
    return float(mask.sum())


def classify_severity(damage_name, percent):
    rules = SEVERITY_RULES.get(damage_name.lower(), [(999, "Minor")])
    for limit, label in rules:
        if percent <= limit:
            return label
    return "Minor"


def _allowed_for_damage(part_name_lower, damage_cid):
    """Return True if part_name is a valid attribution target for this damage class."""
    allowed   = _ATTRIBUTION_ALLOWED.get(damage_cid)
    excluded  = _ATTRIBUTION_EXCLUDED.get(damage_cid, set())

    if part_name_lower in excluded:
        return False
    if allowed is not None and part_name_lower not in allowed:
        return False
    return True


def generate_damage_report(damage_detections, parts_detections,
                           damage_class_names, parts_class_names):
    if damage_detections is None or len(damage_detections) == 0:
        return {
            "overall_health_score": 100,
            "total_damages": 0,
            "parts": [],
            "summary": "No visible damage detected",
        }

    # grouped[(part_name, damage_type)] = {
    #   "part_name", "damage_type", "confidence",
    #   "intersection_mask",   <- union of intersections with this part
    #   "part_mask"            <- the part mask (for area reference)
    # }
    grouped = {}

    for dmask, dcid, conf in zip(
        damage_detections.mask, damage_detections.class_id, damage_detections.confidence
    ):
        damage_cid    = int(dcid)
        damage_name   = damage_class_names[damage_cid]
        damage_pixels = mask_area(dmask)
        damage_area   = dmask.sum()
        matched_any   = False

        if parts_detections is not None and parts_detections.mask is not None:
            for pmask, pcid in zip(parts_detections.mask, parts_detections.class_id):
                intersection = np.logical_and(dmask, pmask)
                inter_pixels = intersection.sum()
                if inter_pixels == 0:
                    continue

                # Absolute overlap fraction gate
                frac = float(inter_pixels) / damage_area if damage_area > 0 else 0.0
                if frac < MIN_ATTRIBUTION_OVERLAP:
                    continue

                # Type-specific attribution filter
                part_name_lower = parts_class_names[int(pcid)].lower()
                if not _allowed_for_damage(part_name_lower, damage_cid):
                    continue

                # Credit this part
                matched_any = True
                part_name   = parts_class_names[int(pcid)]
                key         = (part_name, damage_name)

                if key not in grouped:
                    grouped[key] = {
                        "part_name":         part_name,
                        "damage_type":       damage_name,
                        "confidence":        float(conf),
                        "intersection_mask": intersection.copy(),
                        "part_mask":         pmask,
                    }
                else:
                    grouped[key]["intersection_mask"] = np.logical_or(
                        grouped[key]["intersection_mask"], intersection
                    )
                    grouped[key]["confidence"] = max(grouped[key]["confidence"], float(conf))

        # If no valid part matched, treat as external damage
        if not matched_any:
            if damage_pixels < MIN_EXTERNAL_PIXELS:
                continue
            key = ("Unknown_External", damage_name)
            if key not in grouped:
                grouped[key] = {
                    "part_name":         "Unknown_External",
                    "damage_type":       damage_name,
                    "confidence":        float(conf),
                    "intersection_mask": dmask.copy(),
                    "part_mask":         None,
                }
            else:
                grouped[key]["intersection_mask"] = np.logical_or(
                    grouped[key]["intersection_mask"], dmask
                )
                grouped[key]["confidence"] = max(grouped[key]["confidence"], float(conf))

    # Build final report
    reports       = []
    total_penalty = 0

    for _, item in grouped.items():
        dp = mask_area(item["intersection_mask"])

        if item["part_mask"] is not None:
            pp  = mask_area(item["part_mask"])
            pct = round(min(100.0, (dp / pp) * 100), 2) if pp > 0 else 0.0
            severity = classify_severity(item["damage_type"], pct)
        else:
            # No reference part area — cannot compute meaningful %, skip severity
            pct      = None
            severity = None

        total_penalty += SEVERITY_PENALTY.get(severity, 0)

        reports.append({
            "part_name":       item["part_name"],
            "damage_type":     item["damage_type"],
            "confidence":      round(item["confidence"], 3),
            "damage_percent":  pct,
            "severity":        severity,
            "affected_pixels": int(dp),
        })

    health = max(0, 100 - total_penalty)
    return {
        "overall_health_score": health,
        "total_damages":        len(reports),
        "parts":                reports,
        "summary":              f"{len(reports)} grouped damages found",
    }
