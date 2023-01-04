import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from duet.process import convert_ns, determine_environment
from duet.constants import RF, DF

# Scipy data types
from collections import namedtuple
from dataclasses import make_dataclass

fields = ["confidence_interval", "bootstrap_distribution", "standard_error"]
BootstrapResult = make_dataclass("BootstrapResult", fields)
ConfidenceInterval = namedtuple("ConfidenceInterval", ["low", "high"])


class StopExecution(Exception):
    def _render_traceback_(self):
        pass


def translate(df):
    if RF.type in df:
        type_translate = {
            "seqn": "Sequential",
            "duet": "Asynchronous duet",
            "syncduet": "Synchronous duet",
        }
        df[RF.type] = df[RF.type].astype("category")
        df[RF.type] = df[RF.type].cat.rename_categories(type_translate)

    if RF.suite in df:
        suite_translate = {
            "renaissance": "Renaissance",
            "dacapo": "DaCapo",
            "scalabench": "Scalabench",
            "speccpu": "SPEC CPU",
        }
        df[RF.suite] = df[RF.suite].astype("category")
        df[RF.suite] = df[RF.suite].cat.rename_categories(suite_translate)

    if RF.suite in df:
        suite_translate = {
            "shared": "Renaissance",
            "dacapo": "DaCapo",
            "scalabench": "Scalabench",
            "speccpu": "SPEC CPU",
        }
        df[RF.suite] = df[RF.suite].astype("category")
        df[RF.suite] = df[RF.suite].cat.rename_categories(suite_translate)
    return df


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


order_env = {
    DF.env: ["bare-metal", "AWS t3.medium", "shared-vm"],
}

order_suite = {
    RF.suite: ["Renaissance", "DaCapo", "Scalabench", "SPEC CPU"],
}

order_type = {
    RF.type: ["Sequential", "Asynchronous duet", "Synchronous duet"],
}

orders = {**order_env, **order_suite, **order_type}

colormap = {
    "Sequential": px.colors.qualitative.Plotly[0],
    "Asynchronous duet": px.colors.qualitative.Plotly[1],
    "Synchronous duet": px.colors.qualitative.Plotly[2],
    "Renaissance": px.colors.qualitative.Plotly[3],
    "DaCapo": px.colors.qualitative.Plotly[4],
    "ScalaBench": px.colors.qualitative.Plotly[5],
    "SPEC CPU": px.colors.qualitative.Plotly[6],
}


def save_fig_facet_col_env(
    fig: go.Figure, xaxis_title, yaxis_title, legend_title, filename, **kwargs
):
    # Update facet axis description
    fig.for_each_xaxis(lambda a: a.title.update(text=""))
    fig.for_each_yaxis(lambda a: a.title.update(text=""))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    xaxis_annotation = go.layout.Annotation(
        x=0.5,
        y=-0.17,
        font=dict(size=14),
        showarrow=False,
        text=xaxis_title,
        textangle=0,
        xref="paper",
        yref="paper",
    )
    yaxis_annotation = go.layout.Annotation(
        x=-0.1,
        y=0.5,
        font=dict(size=14),
        showarrow=False,
        text=yaxis_title,
        textangle=-90,
        xref="paper",
        yref="paper",
    )
    fig.update_layout(
        legend_title=legend_title,
        legend=dict(orientation="h", yanchor="top", xanchor="center", y=1.27, x=0.5),
        annotations=list(fig.layout.annotations) + [xaxis_annotation, yaxis_annotation],
        height=380,
        width=720,
        template="plotly_white",
        **kwargs
    )
    fig.write_image(filename)
    return fig
