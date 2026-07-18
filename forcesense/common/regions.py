"""Body-name → contact-region mapping (shared across forcesense).

5 regions: {left,right}_arm, {left,right}_leg, trunk. Used by the trainer,
finetune, eval, and the live viewers to group the ~29 links into regions.
"""


def region_of(name):
    side = "left" if name.startswith("left") else ("right" if name.startswith("right") else "")
    if any(k in name for k in ("shoulder", "elbow", "wrist", "hand")):
        return f"{side}_arm"
    if any(k in name for k in ("hip", "knee", "ankle")):
        return f"{side}_leg"
    return "trunk"
