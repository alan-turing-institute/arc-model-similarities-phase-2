import os

import constants
from scipy.stats import pearsonr

import wandb

# Constants
ENTITY = "turing-arc"
PROJ = "ms2"

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

# Vulnerability keys
A_VULN = "vulnerability_A"
B_VULN = "vulnerability_B"


# Goal: tables of correlations

# H1: Target vulnerability and transfer success
# H2: Surrogate vulnerability and transfer success
# Note: transfer metrics are always associated with the surrogate run
# 2 direcitons of attack
# 2 distributions attacks are based on
# 2 kinds of attack
# 2 transfer sucess metrics
# 1 vulnerability metric
# = 12 correlation stats total per hypothesis

# H3: Similarity metrics and transfer success
# 2 directions of attack
# 2 distributions attacks are based on
# 2 kinds of attack
# 2 transfer success metrics
# = 16 correlation stats per similarity metric

# H4: Dataset size (or ratios) and transfer success
# 2 directions of attack
# 2 distributions attacks are based on
# 2 kinds of attack
# 2 transfer success metrics
# 4+ versions (target size, surrogate size, t/s ratio, s/t ratio)
# = lots of possible ways of doing this

# === Reusable functions === #

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


# === H1 and H2 Functions === #

# H1: Target vulnerability and transfer success
# H2: Surrogate vulnerability and transfer success
# Note: transfer metrics are always associated with the surrogate run
# 2 direcitons of attack
# 2 distributions attacks are based on
# 2 kinds of attack
# 2 transfer sucess metrics
# 1 vulnerability metric
# = 16 correlation stats total per hypothesis

# H1 and H2 function to make list of strings
def make_vuln_keys(model: str) -> list[str]:
    return [
        f"vulnerability_{model}A_fga",
        f"vulnerability_{model}B_fga",
        f"vulnerability_{model}A_boundary",
        f"vulnerability_{model}B_boundary",
    ]


# H1 Function for getting vuln T x success corr
# other vuln, own transfer success

# H2 Function for getting vuln S x success corr
# own vuln, own transfer success
def make_vuln_corr_table(
    wandb_runs: wandb.Api.runs,
    directions: list[str],
    distributions: list[str],
    atk_names: list[str],
    atk_metric_name: str,
):
    # Filter runs
    runs = [
        run
        for run in wandb_runs
        if A_VULN in run.summary.keys() or B_VULN in run.summary.keys()
    ]

    # Model vulnerability keys
    A_keys = make_vuln_keys("A")
    B_keys = make_vuln_keys("B")

    # Lists to store results in
    A_vuln = [[]] * len(A_keys)
    A_success = [[]] * len(distributions) * len(atk_names)
    B_vuln = [[]] * len(B_keys)
    B_success = [[]] * len(distributions) * len(atk_names)

    # Extract model vulnerability stats and transfer success
    it = 0
    for run in runs:
        if run.name[-1] == "A":
            direction = DIRECTIONS[0]
            for i, key in enumerate(A_keys):
                it += 1
                A_vuln[i].append(run.summary["vulnerability_A"][key])
            count = 0
            for distribution in distributions:
                for atk_name in atk_names:
                    A_success[count].append(
                        run.summary[direction][distribution][atk_name][atk_metric_name]
                    )
                    count += 1
        if run.name[-1] == "B":
            direction = DIRECTIONS[1]
            for i, key in enumerate(B_keys):
                B_vuln[i].append(run.summary["vulnerability_B"][key])
            count = 0
            for distribution in distributions:
                for atk_name in atk_names:
                    B_success[count].append(
                        run.summary[direction][distribution][atk_name][atk_metric_name]
                    )
                    count += 1
    print(it)

    # Make 2 x 4 grid of correlations - attacks on rows, same A2B,
    # B2A > A B A B format as H3
    print("A vulnerability:")
    print(len(A_vuln))
    print(len(A_vuln[0]))
    print(len(A_vuln[1]))
    print(len(A_vuln[2]))
    print(len(A_vuln[3]))
    print("A Success:")
    print(len(A_success))
    print(len(A_success[0]))
    print(len(A_success[1]))
    print(len(A_success[2]))
    print(len(A_success[3]))
    print("B vulnerability:")
    print(len(B_vuln))
    print(len(B_vuln[0]))
    print(len(B_vuln[1]))
    print(len(B_vuln[2]))
    print(len(B_vuln[3]))
    print("B Success:")
    print(len(B_success))
    print(len(B_success[0]))
    print(len(B_success[1]))
    print(len(B_success[2]))
    print(len(B_success[3]))


# === H3 Functions === #

# H3: Similarity metrics and transfer success
# 2 directions of attack
# 2 distributions attacks are based on
# 2 kinds of attack
# 2 transfer success metrics
# = 16 correlation stats per similarity metric

# H3 Function for getting correlations in a similarity_metric x attack_metric grid
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


# H3 Function that takes a correlation table, converts it to a latex table
# Need to:
# - Add upper and lower headers - opinionated
# - Add row names
# - Round numbers to 2 decimal places
# - Bold significant numbers
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
        "\\begin{table}[H]\n"
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
        "\\end{tabular}\n"
        "\\caption*{All values rounded to 2 decimal places. "
        f"Values significant at the {confidence_level*100:.0f}"
        "\\% confidence level highlighted in bold.}"
        "\n\\end{table}"
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
    # Wandb
    api = wandb.Api()
    path = os.path.join(ENTITY, PROJ)
    runs = api.runs(path=path)

    # H1 & H2
    make_vuln_corr_table(
        wandb_runs=runs,
        directions=DIRECTIONS,
        distributions=DISTS,
        atk_names=ATTACKS,
        atk_metric_name=ATK_METRIC_NAMES[0],
    )

    # H3: Loop over attacks and attack metrics, make latex tables
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

    # H4


if __name__ == "__main__":
    main()
