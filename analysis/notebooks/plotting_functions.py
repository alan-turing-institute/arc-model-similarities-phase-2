import seaborn as sns


def plot_white_boxplot(
    data,
    success_metric,
    title,
    ax,
    y=None,
    order=None,
    xlabel="",
    ylabel="",
    xmin=0,
    xmax=1,
    xticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
):
    props = {
        "boxprops": {"facecolor": "none", "edgecolor": "gray"},
        "medianprops": {"color": "gray"},
        "whiskerprops": {"color": "gray"},
        "capprops": {"color": "gray"},
    }
    sns.boxplot(data=data, x=success_metric, y=y, ax=ax, order=order, **props)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(xticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_scatter(
    data,
    x,
    y,
    ax,
    title="",
    xlabel="",
    ylabel="",
    xmin=0,
    xmax=1,
    ymin=None,
    ymax=None,
    xticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
    yticks=None,
    hue=None,
    hue_order=None,
    style=None,
    style_order=None,
    marker_size=None,
    colors=None,
):
    sns.scatterplot(
        data=data,
        x=x,
        y=y,
        ax=ax,
        hue=hue,
        hue_order=hue_order,
        style=style,
        style_order=style_order,
        s=marker_size,
        palette=colors,
    )
    ax.get_legend().remove()
    ax.set_xlim(xmin, xmax)
    if ymin is not None:
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_blank_scatter(ax):
    ax.spines.set_visible(False)
