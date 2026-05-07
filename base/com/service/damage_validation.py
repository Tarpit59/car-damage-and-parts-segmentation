"""
damage_validation.py
====================
Filters false damage predictions based on spatial overlap with car parts.

Validation rules (updated):
  - dent        : invalid on headlights, taillights, glass parts, tyre, or unknown regions.
                  Also invalid if the bulk of the mask sits on unknown area and only a
                  small portion touches a named car part.
  - tire flat   : must overlap a tyre part. Invalid if only on unknown region.
  - lamp broken : must overlap a headlight or taillight only. Invalid on any other
                  named part or unknown region.
  - scratch     : invalid on any glass part (windshield, door glass, diggi glass)
                  and invalid if on unknown region.
  - any damage  : if the majority of the damage mask is on an unknown (un-segmented)
                  region and only a small fraction touches any named car part, the
                  detection is discarded.
"""

import numpy as np

DAMAGE_CLASS_NAMES = [
    "crack", "dent", "glass shatter", "lamp broken", "scratch", "tire flat",
]

PARTS_CLASS_NAMES = [
    'Diggi_Back_Door', 'Diggi_Back_Door_Glass', 'Front_Bumper',
    'Front_Windshield_Glass', 'Grill', 'Hood_Bonnet',
    'Left_Fender', 'Left_Front_Door', 'Left_Front_Door_Glass',
    'Left_Headlight', 'Left_Quarter_Panel', 'Left_Rear_Door',
    'Left_Rear_Door_Glass', 'Left_Running_Board', 'Left_Side_Mirror',
    'Left_Taillight', 'Rear_Bumper',
    'Right_Fender', 'Right_Front_Door', 'Right_Front_Door_Glass',
    'Right_Headlight', 'Right_Quarter_Panel', 'Right_Rear_Door',
    'Right_Rear_Door_Glass', 'Right_Running_Board',
    'Right_Side_Mirror', 'Right_Taillight',
    'Roof', 'tyre',
]

# ── Part sets (lower-cased for easy comparison) ──────────────────────────────
_GLASS_PARTS = {
    'diggi_back_door_glass', 'front_windshield_glass',
    'left_front_door_glass', 'left_rear_door_glass',
    'right_front_door_glass', 'right_rear_door_glass',
}
_LAMP_PARTS  = {'left_headlight', 'left_taillight', 'right_headlight', 'right_taillight'}
_TYRE_PARTS  = {'tyre'}

# All known named car parts (everything the segmentation model can label)
_ALL_KNOWN_PARTS = {p.lower() for p in PARTS_CLASS_NAMES}

# ── Thresholds ────────────────────────────────────────────────────────────────
# Minimum fraction of the damage mask that must overlap a part for it
# to be counted as "overlapping" during validation.
OVERLAP_THRESHOLD = 0.20

# If > this fraction of the damage mask is on unknown area → reject entirely.
UNKNOWN_DOMINANCE_THRESHOLD = 0.60


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _overlap_fraction(damage_mask, part_mask):
    """Fraction of damage_mask pixels that also belong to part_mask."""
    damage_area = damage_mask.sum()
    if damage_area == 0:
        return 0.0
    return float(np.logical_and(damage_mask, part_mask).sum()) / damage_area


def _total_known_overlap_fraction(damage_mask, parts_detections):
    """
    Fraction of the damage mask that overlaps *any* named car part.
    Used to detect 'mostly unknown region' cases.
    """
    if parts_detections is None or parts_detections.mask is None:
        return 0.0
    # Union all part masks, then measure overlap
    union = np.zeros(damage_mask.shape, dtype=bool)
    for part_mask in parts_detections.mask:
        union |= part_mask.astype(bool)
    damage_area = damage_mask.sum()
    if damage_area == 0:
        return 0.0
    return float(np.logical_and(damage_mask, union).sum()) / damage_area


def _which_parts_overlap(damage_mask, parts_detections):
    """
    Returns the set of named part labels whose mask overlaps the damage mask
    by at least OVERLAP_THRESHOLD (absolute fraction of the damage mask area).

    Type-specific filtering (e.g. tire-flat only on tyre) is handled separately
    inside each validator and in damage_analytics.py — NOT here.  This function
    only answers: "which named parts does this damage mask substantially touch?"
    """
    if parts_detections is None or parts_detections.mask is None:
        return set()

    overlapping = set()
    for part_mask, part_cid in zip(parts_detections.mask, parts_detections.class_id):
        if _overlap_fraction(damage_mask, part_mask) >= OVERLAP_THRESHOLD:
            overlapping.add(PARTS_CLASS_NAMES[int(part_cid)].lower())
    return overlapping


def _is_mostly_unknown(damage_mask, parts_detections):
    """
    Returns True when the damage mask sits predominantly on an un-segmented
    (unknown) region — i.e. more than UNKNOWN_DOMINANCE_THRESHOLD of the mask
    pixels do NOT belong to any named car part.
    """
    known_fraction = _total_known_overlap_fraction(damage_mask, parts_detections)
    return known_fraction < (1.0 - UNKNOWN_DOMINANCE_THRESHOLD)


# ── Per-damage-type validators ────────────────────────────────────────────────

def _is_valid_glass_shatter(damage_mask, op, parts_detections):
    """Glass shatter is valid only on glass parts."""
    if _is_mostly_unknown(damage_mask, parts_detections):
        return False
    return bool(op & _GLASS_PARTS)


def _is_valid_dent(damage_mask, op, parts_detections):
    """
    Dent is invalid when:
      - mostly on unknown region
      - overlaps headlights / taillights
      - overlaps glass parts
      - overlaps tyre
    """
    if _is_mostly_unknown(damage_mask, parts_detections):
        return False
    invalid_for_dent = _GLASS_PARTS | _TYRE_PARTS | _LAMP_PARTS
    # Must overlap at least one named part that is NOT in the invalid set
    valid_parts = op - invalid_for_dent
    return bool(valid_parts)


def _is_valid_lamp_broken(damage_mask, op, parts_detections):
    """
    Lamp broken is valid ONLY when overlapping a headlight or taillight.
    Any other named part or unknown region → invalid.
    """
    if _is_mostly_unknown(damage_mask, parts_detections):
        return False
    # Must overlap at least one lamp part and nothing else significant
    return bool(op & _LAMP_PARTS) and not bool(op - _LAMP_PARTS)


def _is_valid_scratch(damage_mask, op, parts_detections):
    """
    Scratch is invalid on:
      - any glass part (windshield, door glass, diggi glass)
      - unknown region
    """
    if _is_mostly_unknown(damage_mask, parts_detections):
        return False
    if op & _GLASS_PARTS:
        return False
    # Must touch at least one valid named part
    return bool(op - _GLASS_PARTS)


def _is_valid_tire_flat(damage_mask, op, parts_detections):
    """
    Tire flat must be on a tyre. Invalid if on unknown region or if no tyre overlap.
    """
    if _is_mostly_unknown(damage_mask, parts_detections):
        return False
    return bool(op & _TYRE_PARTS)


# ── Validator dispatch table ──────────────────────────────────────────────────
# Signature for all validators: (damage_mask, overlapping_parts_set, parts_detections) → bool

_VALIDATORS = {
    0: None,                  # crack – no special rule (kept as-is)
    1: _is_valid_dent,
    2: _is_valid_glass_shatter,
    3: _is_valid_lamp_broken,
    4: _is_valid_scratch,
    5: _is_valid_tire_flat,
}


# ── Public API ────────────────────────────────────────────────────────────────

def filter_false_damage_predictions(damage_detections, damage_labels, parts_detections):
    if damage_detections is None or len(damage_detections) == 0:
        return damage_detections, damage_labels

    keep = []
    for i, (mask, cid, conf) in enumerate(zip(
        damage_detections.mask, damage_detections.class_id, damage_detections.confidence
    )):
        validator = _VALIDATORS.get(int(cid))
        if validator is None:
            # No special rule → apply only the generic unknown-region check
            if _is_mostly_unknown(mask, parts_detections):
                print(f"[damage_validation] REMOVED '{DAMAGE_CLASS_NAMES[int(cid)]}' "
                      f"conf={conf:.2f} — mostly unknown region")
            else:
                keep.append(i)
            continue

        overlapping_parts = _which_parts_overlap(mask, parts_detections)
        if validator(mask, overlapping_parts, parts_detections):
            keep.append(i)
        else:
            print(f"[damage_validation] REMOVED '{DAMAGE_CLASS_NAMES[int(cid)]}' "
                  f"conf={conf:.2f} — overlapping={overlapping_parts}")

    if not keep:
        return _subset_detections(damage_detections, []), []
    return _subset_detections(damage_detections, keep), [damage_labels[i] for i in keep]


# ── Internal subset helper ────────────────────────────────────────────────────

def _subset_detections(detections, indices):
    import supervision as sv
    if not indices:
        return sv.Detections(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            confidence=np.zeros((0,), dtype=np.float32),
            class_id=np.zeros((0,), dtype=np.int32),
            mask=np.zeros((0, *detections.mask.shape[1:]), dtype=bool)
                 if detections.mask is not None else None,
        )
    idx = np.array(indices, dtype=int)
    return sv.Detections(
        xyxy=detections.xyxy[idx],
        confidence=detections.confidence[idx],
        class_id=detections.class_id[idx],
        mask=detections.mask[idx] if detections.mask is not None else None,
    )
