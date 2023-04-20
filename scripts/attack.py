import argparse

import yaml
from utils import opts2dmpairArgs

from modsim2.attack import compute_transfer_attack, generate_over_combinations
from modsim2.data.loader import DMPair
from modsim2.model.load_model import download_model


def main(
    dataset_config: dict,
    trainer_config: dict,
    attack_config: dict,
    experiment_group: str,
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

    # Convinience functions to get runs + models for A and B respectively
    def download_model_A():
        model_A, run_A = download_model(
            experiment_name=experiment_pair_name + "_A",
            entity=trainer_config["wandb"]["entity"],
            project_name=trainer_config["wandb"]["project"],
            version=attack_config["model_version"],
        )
        return model_A, run_A

    def download_model_B():
        model_B, run_B = download_model(
            experiment_name=experiment_pair_name + "_B",
            entity=trainer_config["wandb"]["entity"],
            project_name=trainer_config["wandb"]["project"],
            version=attack_config["model_version"],
        )
        return model_B, run_B

    # Download and prepare models - only one wandb process allowed at a time
    model_A, run_A = download_model_A()
    run_A.finish()
    model_B, run_B = download_model_B()
    run_B.finish()
    model_A.eval()
    model_B.eval()

    # Prepare dmpair
    dmpair_kwargs = opts2dmpairArgs(
        opt=dataset_config["experiment_groups"][experiment_group][dataset_index],
        seed=dataset_config["seeds"][seed_index],
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

    # Transfer attack over model*dist combinations, comptue succes
    # Names: AB_to_B imples attack trained on model A with images from distribution B,
    # transferred to model B
    transfer_metrics_AA_to_B = compute_transfer_attack(
        model=model_B,
        images=images_A,
        labels=labels_A,
        advs_images=[fga_images["model_A_dist_A"], boundary_images["model_A_dist_A"]],
        attack_names=attack_names,
        batch_size=attack_config["batch_size"],
        devices=devices,
        accelerator=accelerator,
    )
    transfer_metrics_AB_to_B = compute_transfer_attack(
        model=model_B,
        images=images_B,
        labels=labels_B,
        advs_images=[fga_images["model_A_dist_B"], boundary_images["model_A_dist_B"]],
        attack_names=attack_names,
        batch_size=attack_config["batch_size"],
        devices=devices,
        accelerator=accelerator,
    )
    transfer_metrics_BA_to_A = compute_transfer_attack(
        model=model_A,
        images=images_A,
        labels=labels_A,
        advs_images=[fga_images["model_B_dist_A"], boundary_images["model_B_dist_A"]],
        attack_names=attack_names,
        batch_size=attack_config["batch_size"],
        devices=devices,
        accelerator=accelerator,
    )
    transfer_metrics_BB_to_A = compute_transfer_attack(
        model=model_A,
        images=images_B,
        labels=labels_B,
        advs_images=[fga_images["model_B_dist_B"], boundary_images["model_B_dist_B"]],
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
    _, run_A = download_model_A()
    run_A.log({"A_to_B_metrics": A_to_B_metrics}, commit=True)
    run_A.finish()

    _, run_B = download_model_B()
    run_B.log({"B_to_A_metrics": B_to_A_metrics}, commit=True)
    run_B.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument(
        "--dataset_config", type=str, help="path to datasets config file", required=True
    )
    parser.add_argument(
        "--trainer_config", type=str, help="path to trainer config file", required=True
    )
    parser.add_argument(
        "--attack_config", type=str, help="path to attack config file", required=True
    )
    parser.add_argument(
        "--experiment_group",
        type=str,
        help="which experiment group to run",
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

    with open(args.dataset_config, "r") as stream:
        dataset_config = yaml.safe_load(stream)

    with open(args.trainer_config, "r") as stream:
        trainer_config = yaml.safe_load(stream)

    with open(args.attack_config, "r") as stream:
        attack_config = yaml.safe_load(stream)

    main(
        dataset_config=dataset_config,
        trainer_config=trainer_config,
        attack_config=attack_config,
        experiment_group=args.experiment_group,
        dataset_index=args.dataset_index,
        seed_index=args.seed_index,
    )
