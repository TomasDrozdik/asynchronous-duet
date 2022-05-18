import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


def compute_overlaps(df_iterations: pd.DataFrame):
    df_overlap = df_iterations[df_iterations.pair == "A"].merge(
        df_iterations[df_iterations.pair == "B"],
        on=["benchmark", "runid"],
        suffixes=["_A", "_B"],
        how="inner",
    )
    df_overlap["overlap_start_ns"] = df_overlap[
        ["iteration_start_ns_A", "iteration_start_ns_B"]
    ].max(axis=1)
    df_overlap["overlap_end_ns"] = df_overlap[
        ["iteration_end_ns_A", "iteration_end_ns_B"]
    ].min(axis=1)

    # Filter out positive overlap size
    df_overlap["overlap_size"] = (
        df_overlap["overlap_end_ns"] - df_overlap["overlap_start_ns"]
    )
    df_overlap = df_overlap[df_overlap["overlap_size"] > 0]

    df_overlap.drop(["pair_A", "pair_B"], axis=1, inplace=True)
    df_overlap.reset_index(drop=True, inplace=True)
    return df_overlap


def overlapplot(xstart, xend, ycategory, data=None, hue=None, palette=None, **kwargs):
    # Load data
    start = "start"
    end = "end"
    cat = "cat"
    label_color = "label_color"

    df_intervals = pd.DataFrame()
    df_intervals[start] = xstart if data is None else data[xstart]
    df_intervals[end] = xend if data is None else data[xend]
    df_intervals[cat] = ycategory if data is None else data[ycategory]

    categories = data[ycategory].unique()

    # Figure out coloring and labeling
    if hue:
        hue = hue if data is None else data[hue]
        unique_hue = list(hue.unique())

        palette = palette if palette else sns.color_palette(n_colors=len(unique_hue))

        df_intervals[label_color] = hue.apply(
            lambda x: (x, palette[unique_hue.index(x)])
        )
    else:
        df_intervals = df_intervals.melt(
            id_vars=["category"],
            var_name="type",
            value_vars=[start, end],
            value_name="time",
        )
        df_intervals["value"] = np.select(
            [df_intervals["type"] == start, df_intervals["type"] == end], [0, -1]
        )
        df_intervals["overlaps"] = (
            df_intervals.sort_values(by=["category", "time"])
            .groupby("category")["value"]
            .cumsum()
        )
        df_intervals[start] = df_intervals["time"]
        df_intervals[end] = (
            df_intervals.sort_values(by=["category", "time"])
            .groupby("category")["time"]
            .shift(-2, fill_value=None)
        )
        df_intervals.dropna(inplace=True)
        df_intervals.drop(
            df_intervals[df_intervals["overlaps"] == -1].index, inplace=True
        )

        palette = palette if palette else sns.color_palette("rocket", as_cmap=True)
        df_intervals[label_color] = df_intervals["overlaps"].apply(
            lambda x: (x, palette[x])
        )

    # Plot boxes (horizontal bars)
    df_intervals["xrange"] = df_intervals.apply(
        lambda row: (row[start], row[end] - row[start]), axis=1
    )

    yheight = 1
    yheight_shrink = yheight * 0.2
    fig, ax = plt.subplots(**kwargs)
    for label, color in df_intervals[label_color].unique():
        for category_idx, category in enumerate(categories):
            xranges = df_intervals[
                (df_intervals[label_color] == (label, color))
                & (df_intervals[cat] == category)
            ]["xrange"]

            ax.broken_barh(
                xranges,
                yrange=(
                    category_idx * yheight + yheight_shrink,
                    yheight - yheight_shrink,
                ),
                color=color,
            )

    # Create legend
    legend_elements = [
        matplotlib.patches.Patch(facecolor=x[1], label=x[0])
        for x in df_intervals[label_color].unique()
    ]
    ax.legend(handles=legend_elements)

    ax.set_yticks(
        [
            (category_idx * yheight) + yheight / 2
            for category_idx in range(len(categories))
        ],
        labels=categories,
    )
    return ax
