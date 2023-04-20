import os

import wandb


def _get_runs(
    run_name: str,
    entity: str,
    project_name: str,
) -> wandb.Api.runs:
    """
    A private function to extract all wandb.API.runs for a given run_name

    Args:
        run_name: The run name to get runs for
        entity: The wandb entity name
        project_name: The wandb project name

    Returns: The wandb.Api.runs with all runs with the run name in question
    """
    api = wandb.Api()
    path = os.path.join(entity, project_name)
    runs = api.runs(path=path, filters={"display_name": run_name})
    return runs


def run_exists(
    run_name: str,
    entity: str,
    project_name: str,
) -> bool:
    """
    A convinience function for checking if a given wandb run name already exists
    on wandb. Returns True or False

    Args:
        run_name: The name of the run to check for existence
        entity: The wandb entity name
        project_name: The wandb project name

    Returns: True of False denoting the existence of the run
    """
    runs = _get_runs(run_name=run_name, entity=entity, project_name=project_name)
    return len(runs) > 0


def _get_run_info_from_name(
    run_name: str,
    entity: str,
    project_name: str,
) -> wandb.Api.run:
    """
    A convinience function for returning a wandb.API.run based on its name.
    Exceptions in the function mean that the run name must be unique. Returns
    the corresponding wandb.API.run if its name is unique.

    Args:
        run_name: The name of the run to return
        entity: The wandb entity name
        project_name: The wandb project name

    Returns: the corresponding wandb.API.run
    """
    runs = _get_runs(run_name=run_name, entity=entity, project_name=project_name)

    # Check the run is unique
    if len(runs) == 0:
        raise Exception("The run does not exist")
    if len(runs) > 1:
        raise Exception("Too many runs: run_name is not unique on wandb")

    # Return if so
    return runs[0]


def get_run_from_name(
    model_suffix: str, experiment_pair_name: str, entity: str, project_name: str
) -> wandb.run:
    """
    A convinience function for returning a wandb.run based on its name. The run
    name must be unique. The run name is described by a combination of the
    experiment_pair_name and the model_suffix. Returns the corresponding wandb.run
    if its name is unique.

    Args:
        model_suffix: Suffix for the specific run to be downloaded (A or B)
        experiment_pair_name: Name of the experiment (e.g. drop-only_1_0)
        entity: The wandb entity name
        project_name: The wandb project name

    Returns: the corresponding wandb.run
    """
    if model_suffix not in ["A", "B"]:
        raise Exception("only A and B model suffixes are supported")

    run_name = experiment_pair_name + "_" + model_suffix
    run_info = _get_run_info_from_name(
        run_name=run_name, entity=entity, project_name=project_name
    )
    run = wandb.init(
        project=project_name,
        entity=entity,
        name=run_name,
        id=run_info.id,
        resume=True,
    )
    return run
