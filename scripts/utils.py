from modsim2.utils.config import compose_transform


def opts2dmpairArgs(opt: dict, seed: int) -> dict:
    return {
        "drop_percent_A": opt["A"]["drop"],
        "drop_percent_B": opt["B"]["drop"],
        "transforms_A": compose_transform(opt["A"]["transforms"]),
        "transforms_B": compose_transform(opt["B"]["transforms"]),
        "seed": seed,
    }
