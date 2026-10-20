"""
Plotly visualization functions for TrustRAG research dashboard.
Generates metric comparison charts, Reliability Diagrams (ECE), and Risk-Coverage curves.
"""

from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px


def create_metric_comparison_chart(
    baseline_metrics: Dict[str, Any], trustrag_metrics: Dict[str, Any]
) -> go.Figure:
    """Create grouped bar chart comparing key metrics between Baseline RAG and TrustRAG."""
    categories = [
        "Answer Accuracy",
        "Answerable Coverage",
        "Abstention Rate",
        "Unsupported Rate (Attempted)",
        "Correct Abstention Rate",
    ]

    base_vals = [
        baseline_metrics.get("accuracy", 0.0) * 100,
        baseline_metrics.get("answerable_coverage", 0.0) * 100,
        baseline_metrics.get("abstention_rate", 0.0) * 100,
        baseline_metrics.get("unsupported_rate_attempted", 0.0) * 100,
        baseline_metrics.get("correct_abstention_rate", 0.0) * 100,
    ]

    trust_vals = [
        trustrag_metrics.get("accuracy", 0.0) * 100,
        trustrag_metrics.get("answerable_coverage", 0.0) * 100,
        trustrag_metrics.get("abstention_rate", 0.0) * 100,
        trustrag_metrics.get("unsupported_rate_attempted", 0.0) * 100,
        trustrag_metrics.get("correct_abstention_rate", 0.0) * 100,
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                name="Baseline RAG",
                x=categories,
                y=base_vals,
                marker_color="#94a3b8",
                text=[f"{v:.1f}%" for v in base_vals],
                textposition="auto",
            ),
            go.Bar(
                name="TrustRAG (Ours)",
                x=categories,
                y=trust_vals,
                marker_color="#3b82f6",
                text=[f"{v:.1f}%" for v in trust_vals],
                textposition="auto",
            ),
        ]
    )

    fig.update_layout(
        title="Baseline RAG vs. TrustRAG Performance Comparison",
        yaxis_title="Percentage (%)",
        yaxis=dict(range=[0, 105]),
        barmode="group",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=420,
    )
    return fig


def create_reliability_diagram(
    bin_stats: List[Dict[str, Any]], ece_value: float
) -> go.Figure:
    """
    Generate Reliability Diagram showing Observed Accuracy vs Mean Confidence.
    Includes diagonal reference line (Perfect Calibration).
    """
    fig = go.Figure()

    # Perfect calibration reference diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="#94a3b8", width=2),
            name="Perfect Calibration",
        )
    )

    if bin_stats:
        confs = [b["confidence"] for b in bin_stats]
        accs = [b["accuracy"] for b in bin_stats]
        counts = [b["count"] for b in bin_stats]
        text_labels = [
            f"Bin: {b['bin_range']}<br>Count: {b['count']}<br>Acc: {b['accuracy']:.2f}<br>Conf: {b['confidence']:.2f}"
            for b in bin_stats
        ]

        # Observed accuracy bars
        fig.add_trace(
            go.Bar(
                x=confs,
                y=accs,
                width=0.15,
                marker_color="#2563eb",
                opacity=0.7,
                name="Observed Accuracy",
                hovertext=text_labels,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=f"Reliability Diagram (ECE = {ece_value:.3f})",
        xaxis_title="Confidence (Mean per Bin)",
        yaxis_title="Observed Accuracy",
        xaxis=dict(range=[0, 1.05]),
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_risk_coverage_chart(
    trust_points: List[Dict[str, Any]], baseline_points: List[Dict[str, Any]] = None
) -> go.Figure:
    """Plot Risk vs. Coverage trade-off curve across confidence threshold sweeps."""
    fig = go.Figure()

    if trust_points:
        covs = [p["coverage"] * 100 for p in trust_points]
        risks = [p["risk"] * 100 for p in trust_points]
        hover = [
            f"Threshold: {p['threshold']}<br>Coverage: {covs[i]:.1f}%<br>Risk: {risks[i]:.1f}%"
            for i, p in enumerate(trust_points)
        ]

        fig.add_trace(
            go.Scatter(
                x=covs,
                y=risks,
                mode="lines+markers",
                marker=dict(size=8, color="#2563eb"),
                line=dict(color="#2563eb", width=3),
                name="TrustRAG Frontier",
                hovertext=hover,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title="Risk–Coverage Analysis (Error vs Answer Rate)",
        xaxis_title="Answer Coverage (%)",
        yaxis_title="Empirical Risk / Error Rate (%)",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(range=[0, 105]),
        yaxis=dict(range=[0, 105]),
    )
    return fig
