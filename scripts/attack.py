import yaml
from utils import opts2dmpairArgs

from modsim2.attack import compute_transfer_attack, generate_adversial_images
from modsim2.data.loader import DMPair
from modsim2.model.load_model import download_model

# TODO: delete this, add argparse to this script, make generation script for calls
project_path = "/Users/pswatton/Documents/Code/arc-model-similarities-phase-2/"
dataset_config_path = project_path + "configs/datasets.yaml"
experiment_group = "drop-only"
seed_index = 0
dataset_index = 1
num_attack_images = 16


def main(
    dataset_config: str,
    experiment_group: str,
    dataset_index: int,
    seed_index: int,
    num_attack_images: int,
):

    # Generate strings
    experiment_pair_name = f"{experiment_group}_{dataset_index}_{seed_index}"
    model_name_A = experiment_pair_name + "_A_model"
    model_name_B = experiment_pair_name + "_B_model"

    # Download and prepare models
    model_A = download_model(model_name_A)
    model_B = download_model(model_name_B)
    model_A.eval()
    model_B.eval()

    # Prepare dmpair
    dmpair_kwargs = opts2dmpairArgs(
        opt=dataset_config["experiment_groups"][experiment_group][dataset_index],
        seed=dataset_config["seeds"][seed_index],
    )
    dmpair = DMPair(**dmpair_kwargs)

    # Get transfer images
    images_A, labels_A, _, _ = dmpair.sample_from_test_pairs(num_attack_images)
    # images_B, labels_B = get_transfer_images(test_dl_B, num_attack_images)

    # Generate attack images
    # TODO: make this a loop over combos of A and B models/images
    epsilons = [
        0.0,
        0.1,
        0.3,
        0.5,
        1.0,
    ]
    model_A_dl_A_attacks = generate_adversial_images(
        model=model_A,
        images=images_A,
        labels=labels_A,
        num_attack_images=num_attack_images,
        attack_fn_name="BoundaryAttack",
        epsilons=epsilons,
        steps=500,
    )

    # Transfer the attack, compute metrics based on results
    transfer_metrics_AA_to_B = compute_transfer_attack(
        model=model_B,
        images=images_A,
        labels=labels_A,
        advs_images=model_A_dl_A_attacks,
    )

    # Output
    print(transfer_metrics_AA_to_B)


if __name__ == "__main__":
    # TODO: set this up via argparse

    # with open(args.dataset_config, "r") as stream:
    with open(dataset_config_path, "r") as stream:
        dataset_config = yaml.safe_load(stream)

    main(
        dataset_config=dataset_config,
        experiment_group=experiment_group,
        dataset_index=dataset_index,
        seed_index=seed_index,
        num_attack_images=num_attack_images,
    )
