import argparse
import os

import torch
import yaml
from utils import opts2dmpairArgs

from modsim2.attack import compute_transfer_attack, generate_over_combinations
from modsim2.data.loader import DMPair
from modsim2.model.load_model import download_AB_models
from modsim2.model.utils import get_wandb_run


def main(
    experiment_group_config: dict,
    experiment_group: str,
    dmpair_config: str,
    trainer_config: dict,
    attack_config: dict,
    dataset_index: int,
    seed_index: int,
):
    # Get attack vars
    attack_names = attack_config["attack_names"]
    fga_epsilons = attack_config["fga_epsilons"]
    ba_epsilons = attack_config["ba_epsilons"]
    num_attack_images = attack_config["num_attack_images"]

    # Get trainer vars
    devices = trainer_config["trainer_kwargs"]["devices"]
    accelerator = trainer_config["trainer_kwargs"]["accelerator"]

    # Generate experiment pair name string
    experiment_pair_name = f"{experiment_group}_{dataset_index}_{seed_index}"

    # Get model vars
    entity = trainer_config["wandb"]["entity"]
    project_name = trainer_config["wandb"]["project"]
    version = attack_config["model_version"]

    # Download and prepare models - only one wandb process allowed at a time
    model_A, model_B = download_AB_models(
        experiment_pair_name=experiment_pair_name,
        entity=entity,
        project_name=project_name,
        version=version,
    )
    model_A.eval()
    model_B.eval()

    # Prepare dmpair
    dmpair_kwargs = opts2dmpairArgs(
        opt=experiment_group_config["dmpairs"][dataset_index],
        seed=dmpair_config["seeds"][seed_index],
        val_split=dmpair_config["val_split"],
    )
    dmpair = DMPair(**dmpair_kwargs)

    # Get base images from test datasets
    images_A, labels_A, images_B, labels_B = dmpair.sample_from_test_pairs(
        num_attack_images
    )

    # Generate adversarial images
    # 2 models * 2 distributions * 2 attacks = 8
    fga_images = generate_over_combinations(
        model_A=model_A,
        model_B=model_B,
        images_A=images_A,
        labels_A=labels_A,
        images_B=images_B,
        labels_B=labels_B,
        attack_fn_name=attack_names[0],
        epsilons=fga_epsilons,
        device=accelerator,
    )
    boundary_images = generate_over_combinations(
        model_A=model_A,
        model_B=model_B,
        images_A=images_A,
        labels_A=labels_A,
        images_B=images_B,
        labels_B=labels_B,
        attack_fn_name=attack_names[1],
        epsilons=ba_epsilons,
        device=accelerator,
        steps=500,
    )

    # Model Vulnerability Metrics
    vulnerability_AA_fga = (
        torch.sum(fga_images["model_A_dist_A"][1]).item() / num_attack_images
    )
    vulnerability_AB_fga = (
        torch.sum(fga_images["model_A_dist_B"][1]).item() / num_attack_images
    )
    vulnerability_BA_fga = (
        torch.sum(fga_images["model_B_dist_A"][1]).item() / num_attack_images
    )
    vulnerability_BB_fga = (
        torch.sum(fga_images["model_B_dist_B"][1]).item() / num_attack_images
    )
    vulnerability_AA_boundary = (
        torch.sum(boundary_images["model_A_dist_A"][1]).item() / num_attack_images
    )
    vulnerability_AB_boundary = (
        torch.sum(boundary_images["model_A_dist_B"][1]).item() / num_attack_images
    )
    vulnerability_BA_boundary = (
        torch.sum(boundary_images["model_B_dist_A"][1]).item() / num_attack_images
    )
    vulnerability_BB_boundary = (
        torch.sum(boundary_images["model_B_dist_B"][1]).item() / num_attack_images
    )

    # Transfer attack over model*dist combinations, comptue succes
    # Names: AB_to_B imples attack trained on model A with images from distribution B,
    # transferred to model B
    transfer_metrics_AA_to_B = compute_transfer_attack(
        model=model_B,
        images=images_A,
        labels=labels_A,
        advs_images=[
            fga_images["model_A_dist_A"][0],
            boundary_images["model_A_dist_A"][0],
        ],
        attack_names=attack_names,
        batch_size=attack_config["batch_size"],
        devices=devices,
        accelerator=accelerator,
    )
    transfer_metrics_AB_to_B = compute_transfer_attack(
        model=model_B,
        images=images_B,
        labels=labels_B,
        advs_images=[
            fga_images["model_A_dist_B"][0],
            boundary_images["model_A_dist_B"][0],
        ],
        attack_names=attack_names,
        batch_size=attack_config["batch_size"],
        devices=devices,
        accelerator=accelerator,
    )
    transfer_metrics_BA_to_A = compute_transfer_attack(
        model=model_A,
        images=images_A,
        labels=labels_A,
        advs_images=[
            fga_images["model_B_dist_A"][0],
            boundary_images["model_B_dist_A"][0],
        ],
        attack_names=attack_names,
        batch_size=attack_config["batch_size"],
        devices=devices,
        accelerator=accelerator,
    )
    transfer_metrics_BB_to_A = compute_transfer_attack(
        model=model_A,
        images=images_B,
        labels=labels_B,
        advs_images=[
            fga_images["model_B_dist_B"][0],
            boundary_images["model_B_dist_B"][0],
        ],
        attack_names=attack_names,
        batch_size=attack_config["batch_size"],
        devices=devices,
        accelerator=accelerator,
    )

    # Output
    A_to_B_metrics = {
        "dist_A": transfer_metrics_AA_to_B,
        "dist_B": transfer_metrics_AB_to_B,
    }
    B_to_A_metrics = {
        "dist_A": transfer_metrics_BA_to_A,
        "dist_B": transfer_metrics_BB_to_A,
    }

    # Close once finished
    run_A = get_wandb_run(
        model_suffix="A",
        experiment_pair_name=experiment_pair_name,
        entity=entity,
        project_name=project_name,
    )
    run_A.log({"A_to_B_metrics": A_to_B_metrics}, commit=True)
    run_A.log(vulnerability_AA_fga, commit=True)
    run_A.log(vulnerability_BA_fga, commit=True)
    run_A.log(vulnerability_AA_boundary, commit=True)
    run_A.log(vulnerability_BA_boundary, commit=True)
    run_A.finish()

    run_B = get_wandb_run(
        model_suffix="B",
        experiment_pair_name=experiment_pair_name,
        entity=entity,
        project_name=project_name,
    )
    run_B.log({"B_to_A_metrics": B_to_A_metrics}, commit=True)
    run_B.log(vulnerability_AB_fga, commit=True)
    run_B.log(vulnerability_BB_fga, commit=True)
    run_B.log(vulnerability_AB_boundary, commit=True)
    run_B.log(vulnerability_BB_boundary, commit=True)
    run_B.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        This script takes some paths to dataset, trainer, and attack configs,
        along with arguments to the dataset config as inputs. The arguments to
        the dataset config specify which dataset pair to perform the experiment on.

        It then performs the transfer attacks specified in the attack config, and
        logs the results to wandb. See README for more information.
        """
    )
    parser.add_argument(
        "--experiment_groups_path",
        type=str,
        help="path to experiment groups config folder",
        required=True,
    )
    parser.add_argument(
        "--experiment_group",
        type=str,
        help="experiment group to use",
        required=True,
    )
    parser.add_argument(
        "--dmpair_config_path",
        type=str,
        help="path to dmpair config file",
        required=True,
    )
    parser.add_argument(
        "--trainer_config_path",
        type=str,
        help="path to trainer config file",
        required=True,
    )
    parser.add_argument(
        "--attack_config_path",
        type=str,
        help="path to attack config file",
        required=True,
    )
    parser.add_argument(
        "--dataset_index",
        type=int,
        help="index of dataset options within group",
        required=True,
    )
    parser.add_argument(
        "--seed_index", type=int, help="index of seed within seeds", required=True
    )

    args = parser.parse_args()

    with open(
        os.path.join(args.experiment_groups_path, args.experiment_group + ".yaml"), "r"
    ) as stream:
        experiment_group_config = yaml.safe_load(stream)

    with open(args.dmpair_config_path, "r") as stream:
        dmpair_config = yaml.safe_load(stream)

    with open(args.trainer_config_path, "r") as stream:
        trainer_config = yaml.safe_load(stream)

    with open(args.attack_config_path, "r") as stream:
        attack_config = yaml.safe_load(stream)

    main(
        experiment_group_config=experiment_group_config,
        experiment_group=args.experiment_group,
        dmpair_config=dmpair_config,
        trainer_config=trainer_config,
        attack_config=attack_config,
        dataset_index=args.dataset_index,
        seed_index=args.seed_index,
    )
