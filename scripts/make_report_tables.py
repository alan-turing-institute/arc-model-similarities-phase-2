import os

import constants
import yaml
from scipy.stats import pearsonr

import wandb

# Constants
ENTITY = "turing-arc"
PROJ = "ms2"
CIFAR_SIZE = 50000

# Metrics Keys
DIRECTIONS = ["A_to_B_metrics", "B_to_A_metrics"]

# Similarity Keys
SIM_METRIC_NAMES = [
    "mmd_rbf_raw",
    "mmd_rbf_inception",
    "mmd_rbf_umap",
    "mmd_rbf_pca",
    "otdd_exact_raw",
    "otdd_exact_inception",
    "otdd_exact_umap",
    "otdd_exact_pca",
    "kde_umap_kl_approx",
    "kde_pca_kl_approx",
    "kde_gaussian_umap_l2",
    "kde_gaussian_umap_tv",
    "pad_linear_umap",
    "pad_rbf_umap",
    "pad_linear_pca",
    "pad_rbf_pca",
]
SIM_METRIC_LABELS = [
    "MMD (None)",
    "MMD (Inception)",
    "MMD (UMAP)",
    "MMD (PCA)",
    "hline",
    "OTDD (None)",
    "OTDD (Inception)",
    "OTDD (UMAP)",
    "OTDD (PCA)",
    "hline",
    "KL Approx (UMAP)",
    "KL Approx (PCA)",
    "hline",
    "KDE - L2 (UMAP)",
    "KDE - TV (UMAP)",
    "hline",
    "PAD - Linear (UMAP)",
    "PAD - RBF (UMAP)",
    "PAD - Linear (PCA)",
    "PAD - RBF (PCA)",
]

# Attack keys + labels
ATTACKS = ["L2FastGradientAttack", "BoundaryAttack"]
ATK_LABELS = ["Fast Gradient Attack", "Boundary Attack"]

# Distribution keys
DISTS = ["dist_A", "dist_B"]

# Attack metrics keys + labels
ATK_METRIC_NAMES = ["success_rate", "mean_loss_increase"]
ATK_METRIC_LABELS = ["Success Rate", "Mean Loss Increase"]
ATK_METRIC_PAIRS = [
    [ATK_METRIC_NAMES[i], ATK_METRIC_LABELS[i]] for i in range(len(ATK_METRIC_NAMES))
]

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


# Correlation tables
# first heading: direction
# second heading: image (test) distribution
# rows: attack
# one table per metric

# H1 Function for getting target vuln x success corr
# other vuln, own transfer success (A has A -> B success, B has B vuln)
def make_h1_corr_table(
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
    A_vuln = [[] for _ in range(len(A_keys))]
    A_success = [[] for _ in range(len(distributions) * len(atk_names))]
    B_vuln = [[] for _ in range(len(B_keys))]
    B_success = [[] for _ in range(len(distributions) * len(atk_names))]

    # Extract model vulnerability stats and transfer success
    for run in runs:
        # If run is A
        if run.name[-1] == "A":
            # A -> B direction
            direction = directions[0]
            count = 0
            for atk_name in atk_names:
                for distribution in distributions:
                    # Get A -> B transfer
                    A_success[count].append(
                        run.summary[direction][distribution][atk_name][atk_metric_name]
                    )
                    count += 1
            # Get opposite model B vulnerability
            new_run_name = run.name[0:-1] + "B"
            new_run = [run for run in runs if run.name == new_run_name][0]
            for i, key in enumerate(B_keys):
                B_vuln[i].append(
                    new_run.summary["vulnerability_B"][key][atk_metric_name]
                )
        # If run is B
        if run.name[-1] == "B":
            # B -> A direction
            direction = directions[1]
            count = 0
            for atk_name in atk_names:
                for distribution in distributions:
                    # Get B -> A transfer
                    B_success[count].append(
                        run.summary[direction][distribution][atk_name][atk_metric_name]
                    )
                    count += 1
            # Get opposite model A vulnerability
            new_run_name = run.name[0:-1] + "A"
            new_run = [run for run in runs if run.name == new_run_name][0]
            for i, key in enumerate(A_keys):
                A_vuln[i].append(
                    new_run.summary["vulnerability_A"][key][atk_metric_name]
                )

    # Make correlation table
    # Key order is mA_fga, m_B_fga, m_A_ba, m_B_ba where m is the model,
    # second letter is the distribution, last bit is fga or boundary attack

    # so for target vuln * A->B transfer success for fga on dist A, we want
    # vuln_B and the 1st A success (AA_fga)

    # we want FGA on 1st row, BA on 2nd
    # A->B dist A, A->B dist B, B->A dist A, B->A dist B for columns
    # Note: for A->B always vuln B, for B->A always vuln B
    out = [
        [
            pearsonr(B_vuln[0], A_success[0]),
            pearsonr(B_vuln[1], A_success[1]),
            pearsonr(A_vuln[0], B_success[0]),
            pearsonr(A_vuln[1], B_success[1]),
        ],
        [
            pearsonr(B_vuln[2], A_success[2]),
            pearsonr(B_vuln[3], A_success[3]),
            pearsonr(A_vuln[2], B_success[2]),
            pearsonr(A_vuln[3], B_success[3]),
        ],
    ]

    return out


# H2 Function for getting vuln S x success corr
# own vuln, own transfer success
def make_h2_corr_table(
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
    A_vuln = [[] for _ in range(len(A_keys))]
    A_success = [[] for _ in range(len(distributions) * len(atk_names))]
    B_vuln = [[] for _ in range(len(B_keys))]
    B_success = [[] for _ in range(len(distributions) * len(atk_names))]

    # Extract model vulnerability stats and transfer success
    for run in runs:
        # If run is A
        if run.name[-1] == "A":
            # A -> B direction
            direction = directions[0]
            count = 0
            for atk_name in atk_names:
                for distribution in distributions:
                    # Get A -> B transfer
                    A_success[count].append(
                        run.summary[direction][distribution][atk_name][atk_metric_name]
                    )
                    count += 1
            # Get same model A vulnerability
            for i, key in enumerate(A_keys):
                A_vuln[i].append(run.summary["vulnerability_A"][key][atk_metric_name])
        # If run is B
        if run.name[-1] == "B":
            # B -> A direction
            direction = directions[1]
            count = 0
            for atk_name in atk_names:
                for distribution in distributions:
                    # Get B -> A transfer
                    B_success[count].append(
                        run.summary[direction][distribution][atk_name][atk_metric_name]
                    )
                    count += 1
            # Get same model B vulnerability
            for i, key in enumerate(B_keys):
                B_vuln[i].append(run.summary["vulnerability_B"][key][atk_metric_name])

    # Make correlation table
    out = [
        [
            pearsonr(A_vuln[0], A_success[0]),
            pearsonr(A_vuln[1], A_success[1]),
            pearsonr(B_vuln[0], B_success[0]),
            pearsonr(B_vuln[1], B_success[1]),
        ],
        [
            pearsonr(A_vuln[2], A_success[2]),
            pearsonr(A_vuln[3], A_success[3]),
            pearsonr(B_vuln[2], B_success[2]),
            pearsonr(B_vuln[3], B_success[3]),
        ],
    ]
    return out


def h1h2cor_to_text(
    cor,
    confidence_level,
    atk_labels,
    atk_metric_name,
    atk_metric_label,
    hypothesis,
):
    # Output dimensions
    ncol = 4
    nrow = 2

    # Threshold for significance
    threshold = 1 - confidence_level

    # Table name for latex label + file name
    tname = f"{hypothesis}_{atk_metric_name}"

    # For the table caption
    if hypothesis == "H1":
        hyp_text = "Target Vulnerability"
    if hypothesis == "H2":
        hyp_text = "Surrogate Vulnerability"

    # Write into latex table components
    t_head = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        f"\\caption{{{hyp_text} - {atk_metric_label}}}\n"
        f"\\label{{tab:{tname}}}\n"
        "\\begin{tabular}{c|c|c|c|c}\n"
        "Attack & \\multicolumn{2}{c|}{Surrogate A, Target B} & "
        "\\multicolumn{2}{c}{Surrogate B, Target A}\\\\\n\\hline\n"
        "& A Distribution & B Distribution & A Distribution & "
        "B Distribution \\\\\n\\hline\n"
    )
    t_content = ""
    for i in range(nrow):
        t_content = t_content + atk_labels[i] + " & "
        for j in range(ncol):
            t_content = t_content + write_cor_num(cor[i][j], threshold)
            if j != (ncol - 1):
                t_content = t_content + " & "
            else:
                t_content = t_content + " \\\\\n"
    t_tail = (
        "\\end{tabular}\n"
        "\\caption*{\\textit{All values rounded to 2 decimal places. "
        f"Values significant at the {confidence_level*100:.0f}"
        "\\% confidence level highlighted in bold.}}"
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
        os.path.join(results_folder, "atk_correlations_" + tname + ".tex"), "w"
    ) as f:
        f.write(tex_table)

    # No Return Value
    return None


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
def make_h3_corr_table(
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
    sim_metric_labels: list[str],
    confidence_level: float = 0.95,
) -> None:
    # Get output dimensions
    ncol = len(cor)
    nrow = len(sim_metric_labels)

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
        "Similarity Metric & \\multicolumn{2}{c|}{Surrogate A, Target B} & "
        "\\multicolumn{2}{c}{Surrogate B, Target A}\\\\\n\\hline\n"
        "& A Distribution & B Distribution & A Distribution & "
        "B Distribution \\\\\n\\hline\n"
    )
    t_content = ""
    extra = 0
    for i in range(nrow):
        if sim_metric_labels[i] != "hline":
            t_content = t_content + sim_metric_labels[i].replace("_", "\\_") + " & "
            for j in range(ncol):
                t_content = t_content + write_cor_num(cor[j][i - extra], threshold)
                if j != (ncol - 1):
                    t_content = t_content + " & "
                else:
                    t_content = t_content + " \\\\\n"
        if sim_metric_labels[i] == "hline":
            t_content = t_content + " \\hline\n "
            extra += 1
    t_tail = (
        "\\end{tabular}\n"
        "\\caption*{\\textit{All values rounded to 2 decimal places. "
        f"Values significant at the {confidence_level*100:.0f}"
        "\\% confidence level highlighted in bold.}}"
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


# === H4 Functions === #

# H4: Dataset size (or ratios) and transfer success
# 2 directions of attack
# 2 distributions attacks are based on
# 2 kinds of attack
# 2 transfer success metrics
# 4+ versions (target size, surrogate size, t/s ratio, s/t ratio)
# = lots of possible ways of doing this

TYPES = ["ratio", "diff", "abs-diff"]
TYPE_LABELS = ["Ratio", "Difference", "Absolute Difference"]


def make_h4_cor_table(
    wandb_runs: wandb.Api.runs,
    directions: list[str],
    distributions: list[str],
    atk_name: str,
    atk_metric_names: str,
    type: str,
) -> list[list[pearsonr]]:
    # Output Array: similarity metrics on rows, attack metrics on columns
    out = []

    # Counter
    i = 0

    # Read in all dataset dicts to extract sizes
    experiment_groups_file_names = os.listdir(constants.EXPERIMENT_GROUPS_PATH)
    experiment_groups = {}
    for eg_file_name in experiment_groups_file_names:
        with open(
            os.path.join(
                constants.EXPERIMENT_GROUPS_PATH, experiment_groups_file_names[0]
            )
        ) as file:
            experiment_groups[eg_file_name[:-5]] = yaml.safe_load(file)

    # To access dmpair 0 of little-blur experiment group:
    # experiment_groups["little-blur"]["dmpairs"][0]

    # Loop over directions (A2B, B2A)
    for direction in directions:

        # Filter down runs
        runs = [run for run in wandb_runs if direction in run.summary.keys()]

        # Loop over distributions (A, B)
        for distribution in distributions:

            # New column in output
            out.append([])

            # Get dataset drop ratios
            drop_scores = []
            for run in runs:
                # Get experiment group name, dmpair num, and surrogate letter
                run_eg = run.name[:-6]
                dmpair_num = int(run.name[-5])
                surrogate = run.name[-1]

                # Get corresponding dataset sizes
                drops = experiment_groups[run_eg]["dmpairs"][dmpair_num]
                A_size = CIFAR_SIZE * (1 - drops["A"]["drop"])
                B_size = CIFAR_SIZE * (1 - drops["B"]["drop"])

                # Get surrogate and target drop score depending on type
                if type == "ratio":
                    # surrogate/target drop ratio
                    if surrogate == "A":
                        drop_scores.append(B_size / A_size)
                    if surrogate == "B":
                        drop_scores.append(A_size / B_size)
                if type == "diff":
                    # surrogat - target difference
                    if surrogate == "A":
                        drop_scores.append(A_size - B_size)
                    if surrogate == "B":
                        drop_scores.append(B_size - A_size)
                if type == "abs-diff":
                    # surrogate - target absolute difference
                    drop_scores.append(abs(A_size - B_size))

            # Compute Correlations
            for atk_metric_name in atk_metric_names:

                # Extract the metric
                atk_metric = []
                for run in runs:
                    atk_metric.append(
                        run.summary[direction][distribution][atk_name][atk_metric_name]
                    )

                # Append the correlation
                out[i].append(pearsonr(drop_scores, atk_metric))

            # Iterate
            i += 1

    # Output
    return out


def h4_cor_to_tex(
    cor: list[list[pearsonr]],
    type: str,
    type_label: str,
    atk_name: str,
    atk_label: str,
    atk_metric_labels: str,
    confidence_level: float = 0.95,
) -> None:

    # Get output dimensions
    ncol = len(cor)
    nrow = len(atk_metric_labels)

    # Threshold for significance
    threshold = 1 - confidence_level

    # Table name for latex label + file name
    tname = f"{atk_name}_{type}"

    # Write into latex table components
    t_head = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        f"\\caption{{Dataset Size, {type_label} - {atk_label}}}\n"
        f"\\label{{tab:{tname}}}\n"
        "\\begin{tabular}{c|c|c|c|c}\n"
        "Attack Success Metric & \\multicolumn{2}{c|}{Surrogate A, Target B} & "
        "\\multicolumn{2}{c}{Surrogate B, Target A}\\\\\n\\hline\n"
        "& A Distribution & B Distribution & A Distribution & "
        "B Distribution \\\\\n\\hline\n"
    )
    t_content = ""
    for i in range(nrow):
        t_content = t_content + atk_metric_labels[i].replace("_", "\\_") + " & "
        for j in range(ncol):
            t_content = t_content + write_cor_num(cor[j][i], threshold)
            if j != (ncol - 1):
                t_content = t_content + " & "
            else:
                t_content = t_content + " \\\\\n"
    t_tail = (
        "\\end{tabular}\n"
        "\\caption*{\\textit{All values rounded to 2 decimal places. "
        f"Values significant at the {confidence_level*100:.0f}"
        "\\% confidence level highlighted in bold.}}"
        "\n\\end{table}"
    )

    # Full table
    tex_table = t_head + t_content + t_tail

    # Folder for results - if it does not exist, create it
    results_folder = os.path.join(constants.PROJECT_ROOT, "results")
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    # Save
    with open(os.path.join(results_folder, "dataset_size_" + tname + ".tex"), "w") as f:
        f.write(tex_table)

    # No Return Value
    return None


# === Main Function === #


def main() -> None:
    # Get wandb runs
    api = wandb.Api()
    path = os.path.join(ENTITY, PROJ)
    runs = api.runs(path=path)

    # H1 & H2
    for atk_name, atk_label in ATK_METRIC_PAIRS:
        tab_h1 = make_h1_corr_table(
            wandb_runs=runs,
            directions=DIRECTIONS,
            distributions=DISTS,
            atk_names=ATTACKS,
            atk_metric_name=atk_name,
        )
        tab_h2 = make_h2_corr_table(
            wandb_runs=runs,
            directions=DIRECTIONS,
            distributions=DISTS,
            atk_names=ATTACKS,
            atk_metric_name=atk_name,
        )
        h1h2cor_to_text(
            cor=tab_h1,
            confidence_level=0.95,
            atk_labels=ATK_LABELS,
            atk_metric_label=atk_label,
            atk_metric_name=atk_name,
            hypothesis="H1",
        )
        h1h2cor_to_text(
            cor=tab_h2,
            confidence_level=0.95,
            atk_labels=ATK_LABELS,
            atk_metric_label=atk_label,
            atk_metric_name=atk_name,
            hypothesis="H2",
        )

    # H3: Loop over attacks and attack metrics, make latex tables
    for i in range(len(ATTACKS)):
        for j in range(len(ATK_METRIC_NAMES)):
            cor = make_h3_corr_table(
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
                sim_metric_labels=SIM_METRIC_LABELS,
                confidence_level=0.95,
            )

    # H4 - loop over attacks and correlations
    for i in range(len(ATTACKS)):
        for j in range(len(TYPES)):
            cor = make_h4_cor_table(
                wandb_runs=runs,
                directions=DIRECTIONS,
                distributions=DISTS,
                atk_name=ATTACKS[i],
                atk_metric_names=ATK_METRIC_NAMES,
                type=TYPES[j],
            )
            h4_cor_to_tex(
                cor=cor,
                type=TYPES[j],
                type_label=TYPE_LABELS[j],
                atk_name=ATTACKS[i],
                atk_label=ATK_LABELS[i],
                atk_metric_labels=ATK_METRIC_LABELS,
                confidence_level=0.95,
            )


if __name__ == "__main__":
    main()
