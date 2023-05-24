import os

import constants
from jinja2 import Environment, FileSystemLoader

from modsim2.utils.config import create_transforms


def write_slurm_script(
    template_name,
    called_script_name,
    combinations,
    account_name,
    experiment_names,
    conda_env_path,
    generated_script_names,
):
    # Jinja env
    template_path = os.path.join(
        constants.PROJECT_ROOT,
        "scripts",
        "templates",
    )
    environment = Environment(loader=FileSystemLoader(template_path))
    template = environment.get_template(template_name)

    # Loop over, writing scripts
    for index, combo in enumerate(combinations):
        python_call = f"python scripts/{called_script_name}.py {combo}"
        script_content = template.render(
            account_name=account_name,
            experiment_name=experiment_names[index],
            conda_env_path=conda_env_path,
            python_call=python_call,
        )
        with open(generated_script_names[index], "w") as f:
            f.write(script_content)


def write_bash_script(
    called_script_name,
    combinations,
    generated_script_names,
):
    for index, combo in enumerate(combinations):
        with open(generated_script_names[index], "w") as f:
            f.write(f"python scripts/{called_script_name}.py {combo}")


def opts2dmpairArgs(opt: dict, seed: int, val_split: float) -> dict:
    return {
        "drop_percent_A": opt["A"]["drop"],
        "drop_percent_B": opt["B"]["drop"],
        "transforms_A": create_transforms(opt["A"]["transforms"]),
        "transforms_B": create_transforms(opt["B"]["transforms"]),
        "seed": seed,
        "val_split": val_split,
    }
