
from itertools import combinations, product
import math
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# Shock Sim AI — Commercial MVP v2.3
# Single-file Streamlit application
# =========================================================

st.set_page_config(
    page_title="Shock Sim AI — Commercial MVP v2.3",
    page_icon="🧠",
    layout="wide",
)

# -----------------------------
# Constants
# -----------------------------
DEFAULT_STEPS = 4
INTERVENTION_LEVELS = [25, 50, 75, 100]
REVERSE_TARGETS = [25, 40, 50, 60, 70, 75, 80, 90]


# =========================================================
# 1. NETWORK DATA
# =========================================================

def build_default_network():
    edges = [
        ("Energy", "Suppliers", 0.85, 35),
        ("Raw_Materials", "Suppliers", 0.80, 30),
        ("Suppliers", "Manufacturing", 0.90, 50),
        ("Suppliers", "Logistics", 0.65, 25),
        ("Manufacturing", "Distribution", 0.75, 40),
        ("Logistics", "Distribution", 0.70, 30),
        ("Distribution", "Market", 0.85, 20),
    ]

    graph = nx.DiGraph()
    for source, target, weight, cost in edges:
        graph.add_edge(
            source,
            target,
            weight=float(weight),
            cost=float(cost),
        )
    return graph


def load_network_from_csv(uploaded_file):
    required = {"source", "target", "weight", "cost"}

    try:
        df = pd.read_csv(uploaded_file)

        missing = required - set(df.columns)
        if missing:
            st.error(
                "CSV missing required columns: "
                + ", ".join(sorted(missing))
            )
            return build_default_network(), False

        graph = nx.DiGraph()

        for _, row in df.iterrows():
            source = str(row["source"]).strip()
            target = str(row["target"]).strip()
            weight = float(row["weight"])
            cost = float(row["cost"])

            if not source or not target:
                continue

            if not math.isfinite(weight) or not math.isfinite(cost):
                continue

            # Keep weights in a usable propagation range.
            weight = float(np.clip(weight, 0.0, 1.0))
            cost = max(0.0, cost)

            graph.add_edge(
                source,
                target,
                weight=weight,
                cost=cost,
            )

        if graph.number_of_edges() == 0:
            st.warning("CSV produced an empty network. Using demo network.")
            return build_default_network(), False

        return graph, True

    except Exception as exc:
        st.error(f"CSV read error: {exc}")
        return build_default_network(), False


# =========================================================
# 2. PROPAGATION ENGINE
# =========================================================

def combine_impacts(existing, contribution):
    """
    Nonlinear capped aggregation:
    1 - (1-existing)*(1-contribution)

    This prevents multiple pathways from exceeding 100%.
    """
    existing = float(np.clip(existing, 0, 100))
    contribution = float(np.clip(contribution, 0, 100))

    result = 100.0 * (
        1.0
        - (1.0 - existing / 100.0)
        * (1.0 - contribution / 100.0)
    )
    return float(np.clip(result, 0, 100))


def get_reduction_map(actions):
    reductions = {}
    for action in actions or []:
        key = (action["source"], action["target"])
        reductions[key] = max(
            reductions.get(key, 0.0),
            float(action["reduction"]),
        )
    return reductions


def get_timeline_impacts(
    graph,
    shock_sources_dict,
    actions=None,
    steps=DEFAULT_STEPS,
):
    """
    Persistent temporal propagation.

    Important:
    - T0 contains the shock.
    - At every next step, each currently exposed source transmits
      exposure through outgoing dependencies.
    - Existing exposure persists into the next period.
    - This produces a real T0 -> T1 -> ... propagation replay.
    """
    reductions = get_reduction_map(actions)

    current = {node: 0.0 for node in graph.nodes}

    # Apply all initial shocks.
    for source, magnitude in shock_sources_dict.items():
        if source in current:
            current[source] = max(
                current[source],
                float(np.clip(magnitude, 0, 100)),
            )

    timeline = {"T0": current.copy()}

    for step in range(1, steps + 1):
        # Exposure persists. This is intentional.
        next_impacts = current.copy()

        for source, target, data in graph.edges(data=True):
            source_impact = current.get(source, 0.0)

            reduction = reductions.get((source, target), 0.0)
            effective_weight = float(data["weight"]) * (
                1.0 - reduction / 100.0
            )

            contribution = source_impact * effective_weight

            if contribution > 0:
                next_impacts[target] = combine_impacts(
                    next_impacts.get(target, 0.0),
                    contribution,
                )

        current = {
            node: float(np.clip(value, 0, 100))
            for node, value in next_impacts.items()
        }

        timeline[f"T{step}"] = current.copy()

    return timeline


def timeline_to_dataframe(timeline, graph):
    nodes = list(graph.nodes)

    rows = []
    for time_label, values in timeline.items():
        row = {"Time": time_label}
        for node in nodes:
            row[node] = round(float(values.get(node, 0.0)), 2)
        rows.append(row)

    return pd.DataFrame(rows)


def final_impacts(timeline):
    return timeline[list(timeline.keys())[-1]]


def calculate_total_damage(impacts):
    return float(sum(max(0.0, value) for value in impacts.values()))


# =========================================================
# 3. INTERVENTION / OPTIMIZATION ENGINE
# =========================================================

def intervention_cost(base_cost, reduction):
    return float(base_cost) * (float(reduction) / 100.0)


def estimate_portfolio_count(graph, intervention_levels, max_interventions):
    edge_count = graph.number_of_edges()
    max_interventions = min(max_interventions, edge_count)

    total = 0
    for count in range(1, max_interventions + 1):
        total += math.comb(edge_count, count) * (
            len(intervention_levels) ** count
        )
    return total


def generate_portfolios(
    graph,
    intervention_levels,
    max_interventions,
):
    edges = list(graph.edges(data=True))

    if not edges:
        return []

    max_interventions = min(max_interventions, len(edges))
    portfolios = []

    for count in range(1, max_interventions + 1):
        for selected_edges in combinations(edges, count):
            for reductions in product(
                intervention_levels,
                repeat=count,
            ):
                actions = []
                total_cost = 0.0

                for edge, reduction in zip(
                    selected_edges,
                    reductions,
                ):
                    source, target, data = edge

                    cost = intervention_cost(
                        data["cost"],
                        reduction,
                    )

                    total_cost += cost

                    actions.append(
                        {
                            "source": source,
                            "target": target,
                            "reduction": float(reduction),
                            "cost": float(cost),
                        }
                    )

                portfolios.append(
                    {
                        "actions": actions,
                        "cost": float(total_cost),
                    }
                )

    return portfolios


def calculate_containment(
    baseline_damage,
    final_damage,
):
    if baseline_damage <= 0:
        return 0.0

    containment = (
        (baseline_damage - final_damage)
        / baseline_damage
        * 100.0
    )

    return float(np.clip(containment, 0, 100))


def calculate_efficiency(containment, cost):
    if cost <= 0:
        return 0.0
    return float(containment / cost)


def evaluate_portfolio(
    graph,
    shock_sources_dict,
    portfolio,
    baseline_damage,
):
    timeline = get_timeline_impacts(
        graph=graph,
        shock_sources_dict=shock_sources_dict,
        actions=portfolio["actions"],
        steps=DEFAULT_STEPS,
    )

    impacts = final_impacts(timeline)
    damage = calculate_total_damage(impacts)

    containment = calculate_containment(
        baseline_damage,
        damage,
    )

    damage_avoided = max(
        0.0,
        baseline_damage - damage,
    )

    efficiency = calculate_efficiency(
        containment,
        portfolio["cost"],
    )

    return {
        "actions": portfolio["actions"],
        "cost": float(portfolio["cost"]),
        "timeline": timeline,
        "final_impacts": impacts,
        "final_damage": damage,
        "damage_avoided": damage_avoided,
        "containment": containment,
        "efficiency": efficiency,
    }


def pareto_frontier(results):
    """
    A portfolio is Pareto-efficient if no other portfolio has:
    - lower/equal cost AND
    - higher/equal containment,
    with at least one strict improvement.
    """
    frontier = []

    for candidate in results:
        dominated = False

        for other in results:
            if other is candidate:
                continue

            better_or_equal = (
                other["cost"] <= candidate["cost"]
                and other["containment"] >= candidate["containment"]
            )

            strictly_better = (
                other["cost"] < candidate["cost"]
                or other["containment"] > candidate["containment"]
            )

            if better_or_equal and strictly_better:
                dominated = True
                break

        if not dominated:
            frontier.append(candidate)

    return sorted(
        frontier,
        key=lambda result: (
            result["cost"],
            -result["containment"],
        ),
    )


def select_budget_strategy(results, budget):
    feasible = [
        result
        for result in results
        if result["cost"] <= budget + 1e-9
    ]

    if not feasible:
        return None, []

    selected = max(
        feasible,
        key=lambda result: (
            result["containment"],
            -result["cost"],
        ),
    )

    return selected, feasible


def reverse_resilience(results, target_containment):
    feasible = [
        result
        for result in results
        if result["containment"] + 1e-9 >= target_containment
    ]

    if not feasible:
        return None

    return min(
        feasible,
        key=lambda result: (
            result["cost"],
            -result["containment"],
        ),
    )


def build_reverse_resilience_table(
    results,
    targets=REVERSE_TARGETS,
):
    rows = []

    for target in targets:
        solution = reverse_resilience(
            results,
            target,
        )

        rows.append(
            {
                "Target Containment (%)": target,
                "Minimum Investment ($k)": (
                    round(solution["cost"], 2)
                    if solution
                    else None
                ),
                "Achieved Containment (%)": (
                    round(solution["containment"], 2)
                    if solution
                    else None
                ),
                "Avoided Damage Units": (
                    round(solution["damage_avoided"], 2)
                    if solution
                    else None
                ),
            }
        )

    return pd.DataFrame(rows)


def build_best_by_budget_curve(results):
    """
    Clean investment curve:
    each distinct investment is represented by the maximum
    containment available at or below that investment.
    """
    if not results:
        return pd.DataFrame()

    rows = []

    costs = sorted(
        set(round(result["cost"], 8) for result in results)
    )

    running_best = None

    for cost in costs:
        candidates = [
            result
            for result in results
            if result["cost"] <= cost + 1e-8
        ]

        best = max(
            candidates,
            key=lambda result: (
                result["containment"],
                -result["cost"],
            ),
        )

        if (
            running_best is None
            or best["containment"] > running_best["containment"] + 1e-9
        ):
            running_best = best
            rows.append(
                {
                    "Investment ($k)": round(cost, 2),
                    "Containment (%)": round(
                        best["containment"],
                        2,
                    ),
                    "Avoided Damage Units": round(
                        best["damage_avoided"],
                        2,
                    ),
                }
            )

    return pd.DataFrame(rows)


def find_most_affected_node(
    impacts,
    shock_sources_dict,
):
    downstream = {
        node: impact
        for node, impact in impacts.items()
        if node not in shock_sources_dict
    }

    if downstream:
        return max(
            downstream,
            key=downstream.get,
        )

    return max(
        impacts,
        key=impacts.get,
    ) if impacts else "N/A"


def find_leverage_explanation(
    graph,
    baseline_impacts,
    selected,
):
    """
    Explains why the selected portfolio is important.

    It compares the source's baseline exposure and edge weight,
    while also reporting the modeled outcome of the intervention.
    """
    explanations = []

    for action in selected["actions"]:
        source = action["source"]
        target = action["target"]

        edge_data = graph.get_edge_data(
            source,
            target,
            default={},
        )

        weight = float(edge_data.get("weight", 0.0))
        source_impact = float(
            baseline_impacts.get(source, 0.0)
        )

        leverage_score = source_impact * weight

        explanations.append(
            {
                "source": source,
                "target": target,
                "source_impact": source_impact,
                "dependency_weight": weight,
                "leverage_score": leverage_score,
                "reduction": action["reduction"],
                "investment": action["cost"],
            }
        )

    return sorted(
        explanations,
        key=lambda item: item["leverage_score"],
        reverse=True,
    )


# =========================================================
# 4. FINANCIAL MODEL
# =========================================================

def calculate_avoided_damage_value(
    baseline_damage,
    final_damage,
    financial_value_per_damage_unit,
):
    damage_avoided = max(
        0.0,
        baseline_damage - final_damage,
    )

    return (
        damage_avoided
        * financial_value_per_damage_unit
    )


def calculate_avoided_damage_ratio(
    avoided_value,
    investment,
):
    if investment <= 0:
        return 0.0

    return avoided_value / investment


def build_financial_sensitivity(
    results,
    baseline_damage,
    conversion_factor,
):
    """
    Uses only Pareto-efficient results to keep the economics table
    readable and decision-oriented.
    """
    frontier = pareto_frontier(results)

    rows = []

    for result in frontier:
        avoided_value = calculate_avoided_damage_value(
            baseline_damage=baseline_damage,
            final_damage=result["final_damage"],
            financial_value_per_damage_unit=conversion_factor,
        )

        ratio = calculate_avoided_damage_ratio(
            avoided_value=avoided_value,
            investment=result["cost"],
        )

        rows.append(
            {
                "Investment ($k)": round(
                    result["cost"],
                    2,
                ),
                "Containment (%)": round(
                    result["containment"],
                    2,
                ),
                "Avoided Loss ($k)": round(
                    avoided_value,
                    2,
                ),
                "Avoided Loss / Investment": round(
                    ratio,
                    2,
                ),
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# 5. VISUALIZATION
# =========================================================

def network_impact_figure(
    graph,
    impacts,
    title,
    highlight_edges=None,
):
    positions = nx.spring_layout(
        graph,
        seed=42,
    )

    edge_x = []
    edge_y = []

    highlight_edges = set(
        highlight_edges or []
    )

    normal_edge_x = []
    normal_edge_y = []
    selected_edge_x = []
    selected_edge_y = []

    for source, target in graph.edges():
        x0, y0 = positions[source]
        x1, y1 = positions[target]

        if (source, target) in highlight_edges:
            selected_edge_x += [
                x0, x1, None
            ]
            selected_edge_y += [
                y0, y1, None
            ]
        else:
            normal_edge_x += [
                x0, x1, None
            ]
            normal_edge_y += [
                y0, y1, None
            ]

    normal_edges = go.Scatter(
        x=normal_edge_x,
        y=normal_edge_y,
        mode="lines",
        line=dict(width=1),
        hoverinfo="none",
        name="Dependencies",
    )

    traces = [normal_edges]

    if selected_edge_x:
        selected_edges = go.Scatter(
            x=selected_edge_x,
            y=selected_edge_y,
            mode="lines",
            line=dict(width=4),
            hoverinfo="none",
            name="Intervened Dependencies",
        )
        traces.append(selected_edges)

    node_x = []
    node_y = []
    labels = []
    values = []

    for node in graph.nodes():
        x, y = positions[node]

        node_x.append(x)
        node_y.append(y)
        labels.append(node)
        values.append(
            round(
                impacts.get(node, 0.0),
                2,
            )
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=labels,
        textposition="bottom center",
        marker=dict(
            size=34,
            color=values,
            colorscale="Reds",
            cmin=0,
            cmax=100,
            showscale=True,
            colorbar=dict(title="Impact"),
            line=dict(width=1),
        ),
        hovertemplate=(
            "<b>%{text}</b>"
            "<br>Impact: %{marker.color:.2f}"
            "<extra></extra>"
        ),
        name="Nodes",
    )

    traces.append(node_trace)

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(
            l=10,
            r=10,
            t=50,
            b=10,
        ),
        height=520,
    )

    return fig


def pareto_figure(frontier):
    fig = go.Figure()

    if frontier:
        fig.add_trace(
            go.Scatter(
                x=[
                    result["cost"]
                    for result in frontier
                ],
                y=[
                    result["containment"]
                    for result in frontier
                ],
                mode="lines+markers",
                text=[
                    f"{result['containment']:.1f}%"
                    for result in frontier
                ],
                hovertemplate=(
                    "Investment: %{x:.1f} $k"
                    "<br>Containment: %{y:.1f}%"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Pareto Frontier — Investment vs Containment",
        xaxis_title="Investment ($k)",
        yaxis_title="Containment (%)",
        yaxis=dict(range=[0, 100]),
        height=450,
    )

    return fig


def clean_investment_figure(
    curve,
    conversion_factor,
):
    fig = go.Figure()

    if curve.empty:
        return fig

    curve = curve.copy()

    curve["Avoided Loss ($k)"] = (
        curve["Avoided Damage Units"]
        * conversion_factor
    )

    fig.add_trace(
        go.Scatter(
            x=curve["Investment ($k)"],
            y=curve["Avoided Loss ($k)"],
            mode="lines+markers",
            name="Modeled Avoided Loss ($k)",
            hovertemplate=(
                "Investment: %{x:.1f} $k"
                "<br>Avoided Loss: %{y:.1f} $k"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=curve["Investment ($k)"],
            y=curve["Containment (%)"],
            mode="lines+markers",
            name="Containment (%)",
            yaxis="y2",
            hovertemplate=(
                "Investment: %{x:.1f} $k"
                "<br>Containment: %{y:.1f}%"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Clean Investment Curve",
        xaxis_title="Investment ($k)",
        yaxis=dict(
            title="Modeled Avoided Loss ($k)"
        ),
        yaxis2=dict(
            title="Containment (%)",
            overlaying="y",
            side="right",
            range=[0, 100],
        ),
        legend=dict(
            orientation="h"
        ),
        height=450,
    )

    return fig


# =========================================================
# 6. EXECUTIVE REPORT
# =========================================================

def generate_executive_report(
    company_name,
    industry,
    revenue,
    shock_desc,
    baseline_damage,
    critical_node,
    selected,
    avoided_loss_ratio,
    avoided_loss_value,
    optimization_mode,
    target_or_budget,
    leverage_items,
):
    lines = []

    lines.append("=" * 62)
    lines.append(
        "SHOCK SIM AI — EXECUTIVE COMMAND REPORT"
    )
    lines.append("Commercial MVP v2.3")
    lines.append("=" * 62)

    lines.append("")
    lines.append("COMPANY PROFILE")
    lines.append("-" * 62)
    lines.append(
        f"Company Name: {company_name}"
    )
    lines.append(
        f"Industry: {industry}"
    )
    lines.append(
        f"Annual Revenue: ${revenue:,.2f}M"
    )

    lines.append("")
    lines.append("SCENARIO")
    lines.append("-" * 62)
    lines.append(
        f"Scenario: {shock_desc}"
    )
    lines.append(
        f"Optimization Mode: {optimization_mode}"
    )

    if optimization_mode == "Budget Constraint":
        lines.append(
            f"Available Budget: ${target_or_budget:,.2f}k"
        )
    else:
        lines.append(
            f"Target Containment: {target_or_budget:.1f}%"
        )

    lines.append("")
    lines.append("BASELINE SYSTEMIC EXPOSURE")
    lines.append("-" * 62)
    lines.append(
        f"Baseline Damage Units: {baseline_damage:.2f}"
    )
    lines.append(
        f"Most Affected Downstream Node: {critical_node}"
    )

    lines.append("")
    lines.append("RECOMMENDED COUNTERFACTUAL PLAN")
    lines.append("-" * 62)
    lines.append(
        f"Required Investment: ${selected['cost']:.2f}k"
    )
    lines.append(
        f"Achieved Containment: {selected['containment']:.2f}%"
    )
    lines.append(
        f"Damage Avoided: {selected['damage_avoided']:.2f} units"
    )
    lines.append(
        f"Modeled Avoided Loss: ${avoided_loss_value:.2f}k"
    )
    lines.append(
        f"Avoided Loss / Investment: "
        f"{avoided_loss_ratio:.2f}x"
    )

    lines.append("")
    lines.append("WHY THESE INTERVENTIONS?")
    lines.append("-" * 62)

    for item in leverage_items:
        lines.append(
            f"- {item['source']} -> {item['target']}: "
            f"{item['reduction']:.0f}% reduction | "
            f"Investment ${item['investment']:.2f}k | "
            f"Baseline source impact "
            f"{item['source_impact']:.2f} | "
            f"Dependency weight "
            f"{item['dependency_weight']:.2f}"
        )

    lines.append("")
    lines.append("MODEL GOVERNANCE")
    lines.append("-" * 62)
    lines.append(
        "This is a decision-support simulation. "
        "Damage units, dependency weights, intervention costs, "
        "and financial conversion assumptions must be calibrated "
        "against company-specific data before being used as audited "
        "financial forecasts."
    )

    lines.append("")
    lines.append("=" * 62)

    return "\n".join(lines)


# =========================================================
# 7. PAGE HEADER
# =========================================================

st.title(
    "🧠 Shock Sim AI — Commercial MVP v2.3"
)

st.subheader(
    "Systemic Shock Containment & Investment Decision Engine"
)

st.caption(
    "Core IP: determine the minimum modeled intervention investment "
    "required to achieve a target resilience / containment level."
)


# =========================================================
# 8. SIDEBAR
# =========================================================

st.sidebar.header("🏢 Company Profile")

company_name = st.sidebar.text_input(
    "Company Name",
    "Global Manufacturing Co.",
)

industry = st.sidebar.selectbox(
    "Industry",
    [
        "Manufacturing & Logistics",
        "Energy & Utilities",
        "Supply Chain & Retail",
        "Financial Services",
        "Other",
    ],
)

annual_revenue = st.sidebar.number_input(
    "Annual Revenue ($M)",
    min_value=0.0,
    value=250.0,
    step=10.0,
)


st.sidebar.subheader("💰 Financial Model")

financial_conversion = st.sidebar.number_input(
    "Damage Unit Value ($k)",
    min_value=0.1,
    value=1.0,
    step=0.1,
    help=(
        "How many thousand dollars one modeled damage unit represents. "
        "Calibrate this value using company-specific historical data."
    ),
)

st.sidebar.divider()
st.sidebar.header("📁 Network Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload Custom Network CSV",
    type=["csv"],
    help=(
        "Required columns: source, target, weight, cost. "
        "Weight should normally be between 0 and 1."
    ),
)

if uploaded_file is not None:
    graph, custom_network = load_network_from_csv(
        uploaded_file
    )
    if custom_network:
        st.sidebar.success(
            "Custom company network loaded."
        )
    else:
        st.sidebar.info(
            "Demo network is being used."
        )
else:
    graph = build_default_network()
    st.sidebar.info(
        "Using standard demo network."
    )

st.sidebar.caption(
    f"Nodes: {graph.number_of_nodes()} | "
    f"Edges: {graph.number_of_edges()}"
)


# =========================================================
# 9. SCENARIO LIBRARY
# =========================================================

st.sidebar.divider()
st.sidebar.header("📚 Scenario Library")

scenario_option = st.sidebar.selectbox(
    "Select Scenario",
    [
        "⚡ Energy Shock (Energy +80%)",
        "🚢 Logistics Disruption (Logistics +75%)",
        "🏭 Factory Shutdown (Manufacturing +90%)",
        "🌍 Multi-Shock: Energy + Supplier Failure",
        "⚙️ Custom Scenario Setup",
    ],
)

node_list = list(graph.nodes)

shock_sources_dict = {}

def choose_existing_node(preferred, nodes):
    if preferred in nodes:
        return preferred
    return nodes[0] if nodes else preferred


if "Energy Shock" in scenario_option:
    source = choose_existing_node(
        "Energy",
        node_list,
    )
    shock_sources_dict = {
        source: 80
    }
    shock_desc = (
        f"Energy Sector Severe Spike "
        f"({source} +80%)"
    )

elif "Logistics Disruption" in scenario_option:
    source = choose_existing_node(
        "Logistics",
        node_list,
    )
    shock_sources_dict = {
        source: 75
    }
    shock_desc = (
        f"Logistics Channel Disruption "
        f"({source} +75%)"
    )

elif "Factory Shutdown" in scenario_option:
    source = choose_existing_node(
        "Manufacturing",
        node_list,
    )
    shock_sources_dict = {
        source: 90
    }
    shock_desc = (
        f"Manufacturing Plant Shutdown "
        f"({source} +90%)"
    )

elif "Multi-Shock" in scenario_option:
    energy = choose_existing_node(
        "Energy",
        node_list,
    )
    suppliers = choose_existing_node(
        "Suppliers",
        node_list,
    )

    shock_sources_dict = {
        energy: 70,
        suppliers: 85,
    }

    shock_desc = (
        "Compound Systemic Shock "
        "(Energy + Supplier Failure)"
    )

else:
    st.sidebar.subheader(
        "Custom Shock Parameters"
    )

    custom_source = st.sidebar.selectbox(
        "Primary Shock Source",
        node_list if node_list else ["Energy"],
    )

    custom_magnitude = st.sidebar.slider(
        "Shock Magnitude (%)",
        min_value=30,
        max_value=100,
        value=75,
    )

    shock_sources_dict = {
        custom_source: custom_magnitude
    }

    shock_desc = (
        f"Custom Shock on "
        f"{custom_source} "
        f"({custom_magnitude}%)"
    )


# =========================================================
# 10. OPTIMIZATION SETTINGS
# =========================================================

st.sidebar.divider()
st.sidebar.header("⚙️ Optimization Objective")

opt_mode = st.sidebar.radio(
    "Mode",
    [
        "Budget Constraint",
        "Reverse Resilience (Target Containment)",
    ],
)

if opt_mode == "Budget Constraint":
    budget_limit = st.sidebar.slider(
        "Available Budget ($k)",
        min_value=5,
        max_value=250,
        value=100,
    )
    target_containment_val = None
    target_or_budget = float(
        budget_limit
    )
else:
    target_containment_val = st.sidebar.slider(
        "Target Containment (%)",
        min_value=20,
        max_value=95,
        value=80,
    )
    budget_limit = None
    target_or_budget = float(
        target_containment_val
    )

max_interventions = st.sidebar.slider(
    "Maximum Interventions in Portfolio",
    min_value=1,
    max_value=3,
    value=3,
)

estimated_count = estimate_portfolio_count(
    graph,
    INTERVENTION_LEVELS,
    max_interventions,
)

if estimated_count > 25000:
    st.sidebar.warning(
        f"Search space: {estimated_count:,} portfolios. "
        "Consider max 1–2 interventions for large networks."
    )
else:
    st.sidebar.caption(
        f"Counterfactual portfolios: "
        f"{estimated_count:,}"
    )

run_btn = st.sidebar.button(
    "🚀 RUN COMMERCIAL SIMULATION",
    use_container_width=True,
)


# =========================================================
# 11. LANDING STATE
# =========================================================

if not run_btn:
    st.info(
        "Choose company, network, scenario and optimization "
        "settings, then press RUN COMMERCIAL SIMULATION."
    )

    st.markdown(
        """
### 🌟 Commercial MVP v2.3

**Core workflow**

`SHOCK → PROPAGATION → EXPOSURE → COUNTERFACTUAL SEARCH → MINIMUM INVESTMENT`

**Included**

- Multi-shock systemic propagation
- Custom company network CSV
- Partial interventions: 25 / 50 / 75 / 100%
- Budget-constrained optimization
- Reverse Resilience
- Pareto-efficient portfolios
- Clean investment curve
- Temporal T0 → T4 propagation replay
- Before / after systemic impact maps
- "Why this intervention?" explanation
- Executive command report

> The engine is a computational decision-support model. The product name
> contains "AI", but this version does not claim machine-learning training.
        """
    )

    st.stop()


# =========================================================
# 12. EXECUTION
# =========================================================

baseline_timeline = get_timeline_impacts(
    graph=graph,
    shock_sources_dict=shock_sources_dict,
    actions=[],
    steps=DEFAULT_STEPS,
)

baseline_impacts = final_impacts(
    baseline_timeline
)

baseline_damage = calculate_total_damage(
    baseline_impacts
)

critical_node = find_most_affected_node(
    baseline_impacts,
    shock_sources_dict,
)


with st.spinner(
    "Running systemic propagation and counterfactual search..."
):
    portfolios = generate_portfolios(
        graph=graph,
        intervention_levels=INTERVENTION_LEVELS,
        max_interventions=max_interventions,
    )

    results = [
        evaluate_portfolio(
            graph,
            shock_sources_dict,
            portfolio,
            baseline_damage,
        )
        for portfolio in portfolios
    ]


if not results:
    st.error(
        "No valid intervention portfolios were generated."
    )
    st.stop()


if opt_mode == "Budget Constraint":
    selected, feasible = select_budget_strategy(
        results,
        budget_limit,
    )

    if selected is None:
        st.error(
            "No intervention portfolio fits the selected budget."
        )
        st.stop()

else:
    selected = reverse_resilience(
        results,
        target_containment_val,
    )

    if selected is None:
        st.error(
            f"No portfolio reaches the "
            f"{target_containment_val}% containment target."
        )
        st.stop()

    feasible = [
        result
        for result in results
        if result["containment"]
        >= target_containment_val
    ]


# =========================================================
# 13. FINANCIAL METRICS
# =========================================================

avoided_loss_value = calculate_avoided_damage_value(
    baseline_damage=baseline_damage,
    final_damage=selected["final_damage"],
    financial_value_per_damage_unit=financial_conversion,
)

avoided_loss_ratio = calculate_avoided_damage_ratio(
    avoided_value=avoided_loss_value,
    investment=selected["cost"],
)

leverage_items = find_leverage_explanation(
    graph=graph,
    baseline_impacts=baseline_impacts,
    selected=selected,
)

frontier = pareto_frontier(results)
reverse_df = build_reverse_resilience_table(
    results
)
investment_curve = build_best_by_budget_curve(
    results
)

financial_df = build_financial_sensitivity(
    results=results,
    baseline_damage=baseline_damage,
    conversion_factor=financial_conversion,
)


# =========================================================
# 14. EXECUTIVE COMMAND CENTER
# =========================================================

st.header("📊 Executive Command Center")

st.markdown(
    f"**Company:** {company_name}  |  "
    f"**Industry:** {industry}  |  "
    f"**Scenario:** {shock_desc}"
)

c1, c2, c3, c4, c5 = st.columns(5)

risk_level = (
    "HIGH ⚠️"
    if baseline_damage > 200
    else "MODERATE"
)

c1.metric(
    "Risk Level",
    risk_level,
)

c2.metric(
    "Baseline Damage Units",
    f"{baseline_damage:.1f}",
)

c3.metric(
    "Most Affected Node",
    critical_node,
)

c4.metric(
    "Required Investment",
    f"${selected['cost']:.1f}k",
)

c5.metric(
    "Avoided Loss / Investment",
    f"{avoided_loss_ratio:.2f}x",
)

st.caption(
    "Avoided Loss / Investment is a modeled decision-support ratio, "
    "not an accounting ROI."
)

st.divider()


# =========================================================
# 15. BEFORE / AFTER
# =========================================================

col_a, col_b = st.columns(2)

with col_a:
    st.markdown(
        "### 🔴 Without Intervention"
    )

    modeled_exposure = (
        baseline_damage
        * financial_conversion
    )

    st.error(
        f"""
**Damage Units:** {baseline_damage:.1f}

**Modeled Financial Exposure:**
${modeled_exposure:.1f}k

**Most Affected Node:**
{critical_node}

**Systemic Status:**
Shock propagates through connected dependencies.
        """
    )


with col_b:
    st.markdown(
        "### 🟢 With Recommended Investment"
    )

    st.success(
        f"""
**Investment:**
${selected['cost']:.1f}k

**Containment:**
{selected['containment']:.1f}%

**Modeled Avoided Loss:**
${avoided_loss_value:.1f}k

**Avoided Loss / Investment:**
{avoided_loss_ratio:.2f}x
        """
    )


# =========================================================
# 16. WHY THIS INTERVENTION?
# =========================================================

st.divider()
st.header("🛠️ Why This Intervention?")

st.write(
    "The engine selected the portfolio through counterfactual "
    "simulation. The explanation below shows the baseline exposure "
    "at the source of each selected dependency and the dependency weight."
)

for item in leverage_items:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Dependency",
        f"{item['source']} → {item['target']}",
    )

    col2.metric(
        "Baseline Source Impact",
        f"{item['source_impact']:.1f}%",
    )

    col3.metric(
        "Dependency Weight",
        f"{item['dependency_weight']:.2f}",
    )

    col4.metric(
        "Intervention",
        f"{item['reduction']:.0f}%",
    )

    st.caption(
        f"Modeled intervention investment: "
        f"${item['investment']:.2f}k | "
        f"Leverage indicator: "
        f"{item['leverage_score']:.2f}"
    )


# =========================================================
# 17. RECOMMENDED PORTFOLIO
# =========================================================

st.divider()
st.header("🧰 Recommended Intervention Portfolio")

actions_df = pd.DataFrame(
    selected["actions"]
)

if not actions_df.empty:
    actions_df = actions_df.rename(
        columns={
            "source": "Source",
            "target": "Target",
            "reduction": "Reduction (%)",
            "cost": "Investment ($k)",
        }
    )

    actions_df["Reduction (%)"] = (
        actions_df["Reduction (%)"]
        .round(0)
    )

    actions_df["Investment ($k)"] = (
        actions_df["Investment ($k)"]
        .round(2)
    )

    st.dataframe(
        actions_df,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# 18. REVERSE RESILIENCE ENGINE
# =========================================================

st.divider()
st.header("🧬 Reverse Resilience Engine")

st.write(
    "For each target containment level, the engine searches for the "
    "lowest-cost modeled intervention portfolio that reaches the target."
)

st.dataframe(
    reverse_df,
    use_container_width=True,
    hide_index=True,
)

reverse_plot_df = reverse_df.dropna(
    subset=["Minimum Investment ($k)"]
)

if not reverse_plot_df.empty:
    reverse_fig = go.Figure()

    reverse_fig.add_trace(
        go.Scatter(
            x=reverse_plot_df[
                "Target Containment (%)"
            ],
            y=reverse_plot_df[
                "Minimum Investment ($k)"
            ],
            mode="lines+markers",
            hovertemplate=(
                "Target: %{x:.0f}%"
                "<br>Minimum Investment: %{y:.2f} $k"
                "<extra></extra>"
            ),
        )
    )

    reverse_fig.update_layout(
        title="Reverse Resilience Curve",
        xaxis_title="Target Containment (%)",
        yaxis_title="Minimum Investment ($k)",
        height=430,
    )

    st.plotly_chart(
        reverse_fig,
        use_container_width=True,
    )


# =========================================================
# 19. TARGET RESILIENCE CALCULATOR
# =========================================================

st.divider()
st.header("🎯 Target Resilience Calculator")

calculator_target = st.slider(
    "Required Containment Target (%)",
    min_value=20,
    max_value=95,
    value=(
        int(target_containment_val)
        if target_containment_val is not None
        else 80
    ),
)

calculator_solution = reverse_resilience(
    results,
    calculator_target,
)

if calculator_solution is None:
    st.warning(
        f"No modeled portfolio reaches "
        f"{calculator_target}% containment."
    )
else:
    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Target",
        f"{calculator_target:.0f}%",
    )

    r2.metric(
        "Minimum Investment",
        f"${calculator_solution['cost']:.1f}k",
    )

    r3.metric(
        "Achieved",
        f"{calculator_solution['containment']:.1f}%",
    )

    calc_ratio = calculate_avoided_damage_ratio(
        calculate_avoided_damage_value(
            baseline_damage,
            calculator_solution["final_damage"],
            financial_conversion,
        ),
        calculator_solution["cost"],
    )

    r4.metric(
        "Avoided Loss / Investment",
        f"{calc_ratio:.2f}x",
    )

    st.success(
        f"Minimum modeled investment: "
        f"${calculator_solution['cost']:.1f}k | "
        f"Achieved containment: "
        f"{calculator_solution['containment']:.1f}%"
    )


# =========================================================
# 20. CLEAN INVESTMENT ECONOMICS
# =========================================================

st.divider()
st.header("🔥 Investment Economics")

st.write(
    "This view keeps only improving investment points, making the "
    "economic relationship readable instead of plotting every "
    "counterfactual portfolio."
)

if not financial_df.empty:
    st.dataframe(
        financial_df,
        use_container_width=True,
        hide_index=True,
    )

if not investment_curve.empty:
    st.plotly_chart(
        clean_investment_figure(
            investment_curve,
            financial_conversion,
        ),
        use_container_width=True,
    )


# =========================================================
# 21. INVESTMENT DECISION
# =========================================================

st.divider()
st.header("🎯 Investment Decision")

d1, d2, d3 = st.columns(3)

d1.metric(
    "Required Investment",
    f"${selected['cost']:.1f}k",
)

d2.metric(
    "Modeled Avoided Loss",
    f"${avoided_loss_value:.1f}k",
)

d3.metric(
    "Avoided Loss / Investment",
    f"{avoided_loss_ratio:.2f}x",
)

if avoided_loss_ratio >= 3:
    decision_text = (
        "Under the current model assumptions, the modeled avoided "
        "loss is at least 3x the intervention investment."
    )
elif avoided_loss_ratio >= 1:
    decision_text = (
        "Under the current model assumptions, the modeled avoided "
        "loss exceeds the intervention investment."
    )
else:
    decision_text = (
        "Under the current model assumptions, the modeled avoided "
        "loss is below the intervention investment."
    )

st.info(decision_text)

st.caption(
    "This is a modeled scenario result. Financial conversion, network "
    "weights, intervention costs and shock magnitudes should be calibrated "
    "against company-specific operational and financial data."
)


# =========================================================
# 22. PARETO FRONTIER
# =========================================================

st.divider()
st.header("📈 Pareto Frontier")

st.plotly_chart(
    pareto_figure(frontier),
    use_container_width=True,
)

st.caption(
    "Pareto-efficient portfolios are those for which no other tested "
    "portfolio simultaneously provides lower investment and higher containment."
)


# =========================================================
# 23. SYSTEMIC IMPACT MAPS
# =========================================================

st.divider()
st.header("🗺️ Systemic Impact Maps")

highlight_edges = [
    (
        action["source"],
        action["target"],
    )
    for action in selected["actions"]
]

map1, map2 = st.columns(2)

with map1:
    st.plotly_chart(
        network_impact_figure(
            graph,
            baseline_impacts,
            "Baseline — Without Intervention",
        ),
        use_container_width=True,
    )

with map2:
    st.plotly_chart(
        network_impact_figure(
            graph,
            selected["final_impacts"],
            "Counterfactual — Recommended Plan",
            highlight_edges=highlight_edges,
        ),
        use_container_width=True,
    )


# =========================================================
# 24. PROPAGATION REPLAY
# =========================================================

st.divider()
st.header("⏱️ Propagation Replay")

st.write(
    "This table shows how the shock moves through the network from "
    "T0 to T4. Unlike a static snapshot, each period carries forward "
    "existing exposure and transmits it through outgoing dependencies."
)

replay_tab1, replay_tab2 = st.tabs(
    [
        "Baseline Replay",
        "Optimized Replay",
    ]
)

with replay_tab1:
    baseline_replay_df = timeline_to_dataframe(
        baseline_timeline,
        graph,
    )

    st.dataframe(
        baseline_replay_df,
        use_container_width=True,
        hide_index=True,
    )

with replay_tab2:
    optimized_replay_df = timeline_to_dataframe(
        selected["timeline"],
        graph,
    )

    st.dataframe(
        optimized_replay_df,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# 25. EXECUTIVE REPORT
# =========================================================

st.divider()
st.header("📄 Executive Report Generation")

report_text = generate_executive_report(
    company_name=company_name,
    industry=industry,
    revenue=annual_revenue,
    shock_desc=shock_desc,
    baseline_damage=baseline_damage,
    critical_node=critical_node,
    selected=selected,
    avoided_loss_ratio=avoided_loss_ratio,
    avoided_loss_value=avoided_loss_value,
    optimization_mode=opt_mode,
    target_or_budget=target_or_budget,
    leverage_items=leverage_items,
)

st.text_area(
    "Executive Decision Report",
    report_text,
    height=420,
)

safe_filename = (
    "".join(
        character
        if character.isalnum() or character in "-_"
        else "_"
        for character in company_name
    ).strip("_")
    or "company"
)

st.download_button(
    label="📥 Download Executive Report (TXT)",
    data=report_text,
    file_name=(
        f"{safe_filename}_ShockSim_Executive_Report.txt"
    ),
    mime="text/plain",
    use_container_width=True,
)


# =========================================================
# 26. MODEL ASSUMPTIONS & GOVERNANCE
# =========================================================

st.divider()

with st.expander(
    "ℹ️ Model Assumptions & Governance"
):
    st.markdown(
        """
### Propagation
The model uses a persistent temporal propagation mechanism:
existing exposure carries forward and each exposed node transmits
a weighted contribution through its outgoing dependencies.

### Multiple paths
Multiple contributions are combined with a bounded nonlinear aggregation
so modeled impact remains between 0% and 100%.

### Intervention
Intervention reduces the selected dependency weight by 25%, 50%, 75% or 100%.
The current cost model assumes intervention cost scales linearly with
the selected reduction.

### Financial conversion
`Damage Unit Value ($k)` converts modeled damage units into a modeled
financial exposure. It is an assumption unless calibrated using real
company data.

### Optimization
The current commercial MVP performs exhaustive counterfactual search
over the selected intervention levels and portfolio size. Larger networks
will eventually require pruning, branch-and-bound or a mathematical
optimization solver.

### AI terminology
"Shock Sim AI" is the product name. This version is primarily a
simulation + counterfactual optimization engine; it does not claim
machine-learning training or autonomous financial forecasting.
        """
    )

st.caption(
    "Shock Sim AI — Commercial MVP v2.3 | "
    "Systemic Shock Containment & Investment Decision Engine"
)
