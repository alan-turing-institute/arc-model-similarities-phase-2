import os

import wandb


def _get_runs(
    run_name: str,
    entity: str,
    project_name: str,
) -> wandb.Api.runs:
    """
    A private function to extract all wandb runs for a given run_name

    Args:
        run_name: The run name to get runs for
        entity: The wandb entity name
        project_name: The wandb project name

    Returns:
        wandb.Api.runs: _description_
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
    """
    runs = _get_runs(run_name=run_name, entity=entity, project_name=project_name)
    return len(runs) > 0


def get_run_from_name(
    run_name: str,
    entity: str,
    project_name: str,
) -> wandb.Api.run:
    """
    A convinience function for return a wandb run based on its name. Exceptions in
    the function mean that the run name must be unique. Returns the corresponding
    wandb run if its name is unique.

    Args:
        run_name: The name of the run to return
        entity: The wandb entity name
        project_name: The wandb project name
    """
    runs = _get_runs(run_name=run_name, entity=entity, project_name=project_name)

    # Check the run is unique
    if len(runs) == 0:
        raise Exception("The run does not exist")
    if len(runs) > 1:
        raise Exception("Too many runs: run_name is not unique on wandb")

    # Return if so
    return runs[0]
