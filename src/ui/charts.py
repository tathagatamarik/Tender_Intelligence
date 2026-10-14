"""Shared Plotly styling: restrained palette, thin marks, recessive axes, one y-axis per chart."""
from __future__ import annotations

import plotly.graph_objects as go

# Validated categorical order (assign in this order, never cycled; fold extras into "Other").
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PRIMARY = SERIES[0]
SECONDARY = SERIES[1]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0.0, "#d03b3b"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#ffffff"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Fixed colour per entity so filters never repaint survivors
CHANNEL_COLORS = {"Trade": SERIES[0], "Non-Trade": SERIES[1]}
SCENARIO_COLORS = {"Baseline (current practice)": "#898781", "Optimised": SERIES[0]}


def style(fig: go.Figure, title: str | None = None, x_title: str | None = None, y_title: str | None = None,
          height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title or "", font=dict(size=14, color=INK), x=0, xanchor="left"),
        font=dict(family=FONT, size=12, color=INK2),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
        margin=dict(l=10, r=10, t=44 if title else 16, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title_text="",
                    font=dict(size=11)) if legend else None,
        showlegend=legend, hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)),
        barcornerradius=4, bargap=0.25, colorway=SERIES,
    )
    fig.update_xaxes(title_text=x_title, showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED), zeroline=False)
    fig.update_yaxes(title_text=y_title, gridcolor=GRID, gridwidth=1, linecolor=AXIS, tickfont=dict(color=MUTED),
                     zeroline=False)
    fig.update_traces(selector=dict(type="scatter"), line=dict(width=2))
    fig.update_traces(selector=dict(type="bar"), marker_line_color=SURFACE, marker_line_width=1)
    return fig


def config() -> dict:
    """Plotly config for st.plotly_chart (no logo, offline)."""
    # The icon toolbar is hidden to keep the UI free of decorative icons; hover tooltips remain.
    return {"displaylogo": False, "displayModeBar": False}
