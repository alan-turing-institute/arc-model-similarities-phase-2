import os

import constants
from scipy.stats import pearsonr

import wandb

# Constants
ENTITY = "turing-arc"
PROJ = "ms2"

# Wandb
api = wandb.Api()
path = os.path.join(ENTITY, PROJ)
runs = api.runs(path=path)

# Metrics Keys
DIRECTIONS = ["A_to_B_metrics", "B_to_A_metrics"]

# Similarity Keys
SIM_METRIC_NAMES = [
    "mmd_rbf_raw",
    "mmd_rbf_umap",
    "mmd_rbf_pca",
    "otdd_exact_raw",
    "otdd_exact_umap",
    "otdd_exact_pca",
]

# Attack keys + labels
ATTACKS = ["L2FastGradientAttack", "BoundaryAttack"]
ATK_LABELS = ["Fast Gradient Attack", "Boundary Attack"]

# Distribution keys
DISTS = ["dist_A", "dist_B"]

# Attack metrics keys + labels
ATK_METRIC_NAMES = ["success_rate", "mean_loss_increase"]
ATK_METRIC_LABELS = ["Success Rate", "Mean Loss Increase"]

# Path


# Goal: tables of correlations
# 2 directions of attack
# 2 distributions attacks are based on
# 2 kinds of attack
# 2 transfer success metrics
# = 16 correlation stats per similarity metric


# Function for getting correlations in a similarity_metric x attack_metric grid
# 4 tables: split apart by attack and attack metrics
# 1 table has: 2 directions x 2 distributions
def make_corr_table(
    wandb_runs: wandb.Api.runs,
    sim_metric_names: list[str],
    directions: list[str],
    distributions: list[str],
    atk_name: str,
    atk_metric_name: str,
) -> list[list[pearsonr]]:

    # Output Array: similarity metrics on rows, attack metrics on columns
    out = []

    # Counter
    i = 0

    # Loop over directions (A2B, B2A)
    for direction in directions:
        # Filter down runs
        runs = [run for run in wandb_runs if direction in run.summary.keys()]

        # Loop over distributions (A, B)
        for distribution in distributions:

            # New column in output
            out.append([])

            # Get attack metric for ddirection + dist
            atk_metric = []
            for run in runs:
                atk_metric.append(
                    run.summary[direction][distribution][atk_name][atk_metric_name]
                )

            # Compute correlations
            for sim_metric_name in sim_metric_names:
                sim_metric = [run.summary[sim_metric_name] for run in runs]
                out[i].append(pearsonr(sim_metric, atk_metric))

            # Iterate
            i += 1

    # Output
    return out


# Function that writes bold around a number if at a certain signifiance level
def write_cor_num(
    cor,
    threshold,
):
    num = round(cor[0], 2)
    p = cor[1]

    if p < threshold:
        return f"\\textbf{{{num:.2f}}}"
    else:
        return f"{num}"


# Function that takes a correlation table, converts it to a latex table
# Need to:
# - Add upper and lower headers Y - opinionated
# - Add row names Y
# - Round numbers to 2 decimal places - Y
# - Bold significant numbers - Y
# - Vary label + title according to attack and metric
def cor_to_tex(
    cor: list[list[pearsonr]],
    atk_name: str,
    atk_metric_name: str,
    atk_label: str,
    atk_metric_label: str,
    sim_metric_names: list[str],
    confidence_level: float = 0.95,
) -> None:
    # Get output dimensions
    ncol = len(cor)
    nrow = len(cor[0])

    # Threshold for significance
    threshold = 1 - confidence_level

    # Table name for latex label + file name
    tname = f"{atk_name}_{atk_metric_name}"

    # Write into latex table components
    t_head = (
        "\\begin{table}[h!]\n"
        "\\centering\n"
        f"\\caption{{{atk_label} - {atk_metric_label}}}\n"
        f"\\label{{tab:{tname}}}\n"
        "\\begin{tabular}{c|c|c|c|c}\n"
        "Similarity Metric & \\multicolumn{2}{c|}{A to B} & "
        "\\multicolumn{2}{c}{B to A}\\\\\n\\hline\n"
        "& A Distribution & B Distribution & A Distribution & "
        "B Distribution \\\\\n\\hline\n"
    )
    t_content = ""
    for i in range(nrow):
        t_content = t_content + sim_metric_names[i].replace("_", "\\_") + " & "
        for j in range(ncol):
            t_content = t_content + write_cor_num(cor[j][i], threshold)
            if j != (ncol - 1):
                t_content = t_content + " & "
            else:
                t_content = t_content + " \\\\\n"
    t_tail = (
        f"\\end{{tabular}}\n"
        "\\caption*{{All values rounded to 2 decimal places. "
        f"Values significant at the {confidence_level*100:.0f}"
        "\\% confidence level highlighted in bold.}}"
        "\n\\end{{table}}"
    )

    # Full table
    tex_table = t_head + t_content + t_tail

    # Folder for results - if it does not exist, create it
    results_folder = os.path.join(constants.PROJECT_ROOT, "results")
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    # Save
    with open(
        os.path.join(results_folder, "sim_correlations_" + tname + ".tex"), "w"
    ) as f:
        f.write(tex_table)

    # No Return Value
    return None


# Main Function
def main():
    # Loop over attacks and attack metrics, make latex tables
    for i in range(len(ATTACKS)):
        for j in range(len(ATK_METRIC_NAMES)):
            cor = make_corr_table(
                wandb_runs=runs,
                sim_metric_names=SIM_METRIC_NAMES,
                directions=DIRECTIONS,
                distributions=DISTS,
                atk_name=ATTACKS[i],
                atk_metric_name=ATK_METRIC_NAMES[j],
            )
            cor_to_tex(
                cor=cor,
                atk_name=ATTACKS[i],
                atk_label=ATK_LABELS[i],
                atk_metric_name=ATK_METRIC_NAMES[j],
                atk_metric_label=ATK_METRIC_LABELS[j],
                sim_metric_names=SIM_METRIC_NAMES,
                confidence_level=0.95,
            )


if __name__ == "__main__":
    main()
