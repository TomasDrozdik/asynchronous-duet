import plotly.graph_objects as go
import pandas as pd
from duet.process import convert_ns, determine_environment
from duet.constants import RF

# Scipy data types
from collections import namedtuple
from dataclasses import make_dataclass

fields = ["confidence_interval", "bootstrap_distribution", "standard_error"]
BootstrapResult = make_dataclass("BootstrapResult", fields)
ConfidenceInterval = namedtuple("ConfidenceInterval", ["low", "high"])


class StopExecution(Exception):
    def _render_traceback_(self):
        pass


def load_raw():
    df = pd.concat(
        [
            pd.read_csv("../results/results.bare-metal.csv"),
            pd.read_csv("../results/amazon-mid-july.csv"),
            pd.read_csv("../results/results.teaching.csv"),
            pd.read_csv("../results/cirrus-december-2022-second-try.csv"),
        ]
    )
    df = convert_ns(df)
    df = determine_environment(df)
    df = df[~df[RF.benchmark].isin(["jython", "dummy"])]
    return df


def save_fig_facet_col_env(
    fig: go.Figure, xaxis_title, yaxis_title, legend_title, filename, **kwargs
):
    # Update facet axis description
    fig.for_each_xaxis(lambda a: a.title.update(text=""))
    fig.for_each_yaxis(lambda a: a.title.update(text=""))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    xaxis_annotation = go.layout.Annotation(
        x=0.5,
        y=-0.15,
        # font=dict(size=14),
        showarrow=False,
        text=xaxis_title,
        textangle=0,
        xref="paper",
        yref="paper",
    )
    yaxis_annotation = go.layout.Annotation(
        x=-0.1,
        y=0.5,
        # font=dict(size=14),
        showarrow=False,
        text=yaxis_title,
        textangle=-90,
        xref="paper",
        yref="paper",
    )
    fig.update_layout(
        legend_title=legend_title,
        legend=dict(orientation="h", yanchor="top", xanchor="center", y=1.2, x=0.5),
        annotations=list(fig.layout.annotations) + [xaxis_annotation, yaxis_annotation],
        **kwargs
    )
    fig.write_image(filename)
    return fig
