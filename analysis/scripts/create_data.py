import os

import numpy as np

from modsim2.data.loader import DMPair
from modsim2.similarity.embeddings import EMBEDDING_FN_DICT
from modsim2.utils.config import create_transforms

metrics_config = {"metric": {"arguments": {"embedding_name": "matrix"}}}

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

device = "auto"  # where to run the inception embeddings

# Save copy of CIFAR-10 as numpy array - used to view some images
dmp = DMPair(
    drop_percent_A=0,
    drop_percent_B=0,
    transforms_A=None,
    transforms_B=None,
    metrics_config=metrics_config,
)
dmp.A.setup()
dmp.B.setup()
train_data_A, _ = dmp.get_A_data()
data_A = train_data_A[:20, :, :, :]
cifar_10_file_path = os.path.join(project_root, "data/", "cifar_10_sample.npz")
np.savez_compressed(cifar_10_file_path, cifar_10=data_A)


# Save data as numpy arrays representing the CIFAR-10 dataset with varying
# amounts of data dropped - no embeddings applied (only matrix embedding)
# These data are used to view histograms of individual pixels
di_no_embedding = {}
for drop_percent in [0, 25, 50, 75]:
    dmp = DMPair(
        drop_percent_A=0,
        drop_percent_B=drop_percent / 100,
        transforms_A=None,
        transforms_B=None,
        metrics_config=metrics_config,
    )
    dmp.A.setup()
    dmp.B.setup()
    train_data_A, val_data_A = dmp.get_A_data()
    data_A = np.concatenate((train_data_A, val_data_A), axis=0)

    # Extract embedding callable
    embedding_fn = EMBEDDING_FN_DICT["matrix"]

    # Apply embeddigns
    embed_A = embedding_fn(data_A)

    di_no_embedding["no_embedding_no_transform_drop_" + str(drop_percent)] = embed_A
    print("no_embedding_no_transform_drop_" + str(drop_percent))

no_embedding_file_path = os.path.join(project_root, "data/", "no_embedding.npz")
np.savez_compressed(no_embedding_file_path, **di_no_embedding)


# Save data as numpy arrays representing the CIFAR-10 dataset with varying
# amounts of data dropped - umap embedding applied
# These data are used to view data in 2D
di_umap = {}
for drop_percent in [0, 25, 50, 75]:
    dmp = DMPair(
        drop_percent_A=0,
        drop_percent_B=drop_percent / 100,
        transforms_A=None,
        transforms_B=None,
        metrics_config=metrics_config,
    )
    dmp.A.setup()
    dmp.B.setup()
    train_data_A, val_data_A = dmp.get_A_data()
    data_A = np.concatenate((train_data_A, val_data_A), axis=0)

    # Extract embedding callable
    embedding_fn = EMBEDDING_FN_DICT["inception_umap"]

    # Apply embeddigns
    embed_A = embedding_fn(
        data_A, batch_size=16, device=device, n_components=2, random_seed=42
    )

    di_umap["umap_no_transform_drop_" + str(drop_percent)] = embed_A
    print("umap_no_transform_drop_" + str(drop_percent))

no_embedding_file_path = os.path.join(project_root, "data/", "umap_no_transform.npz")
np.savez_compressed(no_embedding_file_path, **di_umap)
print("finished umap no transform")

# Save data as numpy arrays representing the CIFAR-10 dataset with no data dropped
# and the various transforma applied - umap embedding applied
# These data are used to view data in 2D
di_umap_transforms = {}
li_transforms = [
    [
        {"name": "Grayscale", "kwargs": {"num_output_channels": 3}},
        {
            "name": "ToTensor",
        },
    ],
    [
        {"name": "GaussianBlur", "kwargs": {"kernel_size": 3, "sigma": 1}},
        {
            "name": "ToTensor",
        },
    ],
    [
        {"name": "GaussianBlur", "kwargs": {"kernel_size": 3, "sigma": 3}},
        {
            "name": "ToTensor",
        },
    ],
    [
        {"name": "RandomVerticalFlip", "kwargs": {"p": 1}},
        {
            "name": "ToTensor",
        },
    ],
]
li_transform_names = ["grayscale", "little_blur", "big_blur", "rotate_180"]
i = 0
for transform in li_transforms:
    transform_A = create_transforms(transform)
    dmp = DMPair(
        drop_percent_A=0,
        drop_percent_B=0,
        transforms_A=transform_A,
        transforms_B=None,
        metrics_config=metrics_config,
    )
    dmp.A.setup()
    dmp.B.setup()
    train_data_A, val_data_A = dmp.get_A_data()
    data_A = np.concatenate((train_data_A, val_data_A), axis=0)

    # Extract embedding callable
    embedding_fn = EMBEDDING_FN_DICT["inception_umap"]

    # Apply embeddigns
    embed_A = embedding_fn(
        data_A, batch_size=16, device=device, n_components=2, random_seed=42
    )

    di_umap_transforms["umap_" + li_transform_names[i] + "_drop_0"] = embed_A
    print("umap_" + li_transform_names[i] + "_drop_0")
    i += 1

umap_transforms_file_path = os.path.join(project_root, "data/", "umap_transforms.npz")
np.savez_compressed(umap_transforms_file_path, **di_umap_transforms)
print("finished")

# Save data as numpy arrays representing the CIFAR-10 dataset with no data dropped
# and the various transforma applied - umap embedding applied
# These data are used to view data in 2D
di_transforms = {}
i = 0
for transform in li_transforms:
    transform_A = create_transforms(transform)
    dmp = DMPair(
        drop_percent_A=0,
        drop_percent_B=0,
        transforms_A=transform_A,
        transforms_B=None,
        metrics_config=metrics_config,
    )
    dmp.A.setup()
    dmp.B.setup()
    train_data_A, val_data_A = dmp.get_A_data()
    data_A = np.concatenate((train_data_A, val_data_A), axis=0)

    # Extract embedding callable
    embedding_fn = EMBEDDING_FN_DICT["matrix"]

    # Apply embeddigns
    embed_A = embedding_fn(data_A)

    di_transforms["no_embedding_" + li_transform_names[i] + "_drop_0"] = embed_A
    print("no_embedding_" + li_transform_names[i] + "_drop_0")
    i += 1

no_embedding_transforms_file_path = os.path.join(
    project_root, "data/", "transforms.npz"
)
np.savez_compressed(no_embedding_transforms_file_path, **di_transforms)
print("finished")
