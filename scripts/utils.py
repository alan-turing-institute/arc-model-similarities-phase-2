def opts2dmpairArgs(opt: dict, seed: int) -> dict:
    return {
        "drop_percent_A": opt["A"]["drop"],
        "drop_percent_B": opt["B"]["drop"],
        "transforms_A": opt["A"]["transforms"],
        "transforms_B": opt["B"]["transforms"],
        "seed": seed,
    }
