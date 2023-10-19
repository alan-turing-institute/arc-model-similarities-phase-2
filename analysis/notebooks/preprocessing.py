import os

import constants
import pandas as pd

# wandb filename containing the data to be analysed
# the data are exported from wandb using a csv export
filename = "wandb_export_2023-09-11T17 18 50.517+01 00.csv"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

wandb_file_path = os.path.join(project_root, "data/", filename)

# Read CSV file and load into dataframe
DF_DATA = pd.read_csv(wandb_file_path)

# Tidy up - remove columns with all blank values
DF_DATA.dropna(axis=1, how="all", inplace=True)

# Drop experiment groups that don't feature in final report
DF_DATA.drop(columns=["pad_linear_inception", "pad_rbf_inception"], inplace=True)

# Add name for transform group - used in plots
DF_DATA["Transform Group"] = DF_DATA["Name"].str[:4]
di_transform_map = {
    "gray": "Grayscale",
    "rota": "Rotate 180",
    "big-": "Big blur",
    "litt": "Little blur",
    "drop": "No transform",
}
DF_DATA["Transform Group"] = DF_DATA["Transform Group"].map(di_transform_map)

# Create dissimilarity metric using target model classifications
DF_DATA["A_to_B_dissimilarity"] = 1 - (
    DF_DATA["A_to_B_metrics.dist_A.base_success_rate"]
    / DF_DATA["A_to_B_metrics.dist_B.base_success_rate"]
)
DF_DATA["B_to_A_dissimilarity"] = 1 - (
    DF_DATA["B_to_A_metrics.dist_B.base_success_rate"]
    / DF_DATA["B_to_A_metrics.dist_A.base_success_rate"]
)
DF_DATA["model_dissimilarity"] = DF_DATA["A_to_B_dissimilarity"].fillna(0) + DF_DATA[
    "B_to_A_dissimilarity"
].fillna(0)

# Move metadata for each experiment group into its own column
DF_DATA["A or B"] = DF_DATA["Name"].map(lambda x: str(x)[-1])
DF_DATA["Transform"] = DF_DATA["Name"].map(lambda x: str(x)[:-2])
DF_DATA["Random Seed Group"] = DF_DATA["Transform"].map(lambda x: str(x)[-1])
DF_DATA["Drop Group"] = DF_DATA["Transform"].map(lambda x: str(x)[-3:-2])


# For 'drop only' experiment groups, the drop group labels need to be amended
# The experiment group without any records dropped is not run for the 'drop
# only' experiments
# The labelling of drop groups is not ideal. It relies on them being in the
# same order in the config files for the various transform groups
def amend_drop_group(row):
    if row["Transform Group"] == "Drop only":
        if row["Drop Group"] == "0":
            row["Drop Group"] = "1"
            row["Name"] = row["Name"][:-5] + "1" + row["Name"][-4:]
        elif row["Drop Group"] == "1":
            row["Drop Group"] = "3"
            row["Name"] = row["Name"][:-5] + "3" + row["Name"][-4:]
        elif row["Drop Group"] == "2":
            row["Drop Group"] = "5"
            row["Name"] = row["Name"][:-5] + "5" + row["Name"][-4:]
        elif row["Drop Group"] == "3":
            row["Drop Group"] = "2"
            row["Name"] = row["Name"][:-5] + "2" + row["Name"][-4:]
        else:
            row["Drop Group"] = "4"
            row["Name"] = row["Name"][:-5] + "4" + row["Name"][-4:]
    return row


DF_DATA = DF_DATA.apply(lambda row: amend_drop_group(row), axis=1)

DF_DATA["Drop Group Name"] = DF_DATA["Drop Group"].map(constants.DI_DROP_GROUP_NAMES)
