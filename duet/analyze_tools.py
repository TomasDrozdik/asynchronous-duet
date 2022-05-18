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


def overlapplot(xstart, xend, ycategory, data=None, hue=None, palette=None):
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
        df_overlaps = df_intervals.melt(
            id_vars=[cat],
            var_name="type",
            value_vars=[start, end],
            value_name="time",
        )
        df_overlaps["value"] = np.select(
            [df_overlaps["type"] == start, df_overlaps["type"] == end], [1, -1]
        )
        df_overlaps["overlaps"] = (
            df_overlaps.sort_values(by=[cat, "time"]).groupby(cat)["value"].cumsum()
        )
        df_overlaps[start] = df_overlaps["time"]
        df_overlaps[end] = (
            df_overlaps.sort_values(by=[cat, "time"])
            .groupby(cat)["time"]
            .shift(-1, fill_value=None)
        )

        df_overlaps.dropna(inplace=True)
        df_overlaps.drop(df_overlaps[df_overlaps["overlaps"] == 0].index, inplace=True)

        unique_hue = list(df_overlaps["overlaps"].unique())
        palette = palette if palette else sns.color_palette("rocket", len(unique_hue))
        df_overlaps[label_color] = df_overlaps["overlaps"].apply(
            lambda x: (x, palette[unique_hue.index(x)])
        )

        # Pass back to intervals since the format is kept
        df_intervals = df_overlaps

    # Filter nonzero ranges
    df_intervals = df_intervals[df_intervals[start] < df_intervals[end]]

    # Horizontal bars require xrange tuple in format (x_start, x_length)
    df_intervals["xrange"] = df_intervals.apply(
        lambda row: (row[start], row[end] - row[start]), axis=1
    )

    yheight = 1
    yheight_shrink = yheight * 0.2
    ax = plt.gca()
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
    if len(unique_hue) > 10 and hue is None:
        pass
    else:
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
