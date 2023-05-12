from modsim2.utils.config import create_transforms


def opts2dmpairArgs(opt: dict, seed: int, val_split: float) -> dict:
    return {
        "drop_percent_A": opt["A"]["drop"],
        "drop_percent_B": opt["B"]["drop"],
        "transforms_A": create_transforms(opt["A"]["transforms"]),
        "transforms_B": create_transforms(opt["B"]["transforms"]),
        "seed": seed,
        "val_split": val_split,
    }
