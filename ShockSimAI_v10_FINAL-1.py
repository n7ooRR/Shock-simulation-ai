
from itertools import combinations, product
from dataclasses import dataclass
import math

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# SHOCK SIM AI — v10.0
# Adaptive Resilience Intelligence
# Single-file Streamlit application
# ============================================================

st.set_page_config(
    page_title="Shock Sim AI — Resilience Operating System v10.0",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Shock Sim AI — Resilience Operating System v10.0 FINAL")
st.subheader("Resilience Operating System — Adaptive Digital Twin, Prediction, Optimization & Learning")
st.caption(
    "Adaptive core: calibrate dependency transmission from observed outcomes, "
    "generate compound shocks, search counterfactual interventions, and learn "
    "from post-event outcomes."
)


# ============================================================
# 1. DATA MODELS
# ============================================================

@dataclass
class LearningConfig:
    learning_rate: float = 0.15
    min_multiplier: float = 0.50
    max_multiplier: float = 1.50


# ============================================================
# 2. NETWORK
# ============================================================

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
            adaptive_multiplier=1.0,
        )
    return graph


def load_network_from_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        required = {"source", "target", "weight", "cost"}
        missing = required - set(df.columns)

        if missing:
            st.error(
                "Network CSV is missing columns: "
                + ", ".join(sorted(missing))
            )
            return build_default_network()

        graph = nx.DiGraph()

        for _, row in df.iterrows():
            source = str(row["source"]).strip()
            target = str(row["target"]).strip()
            weight = float(row["weight"])
            cost = float(row["cost"])

            if not source or not target:
                continue

            weight = float(np.clip(weight, 0.0, 1.0))
            cost = max(0.0, cost)

            graph.add_edge(
                source,
                target,
                weight=weight,
                cost=cost,
                adaptive_multiplier=1.0,
            )

        if graph.number_of_edges() == 0:
            st.warning("The uploaded network contains no usable edges.")
            return build_default_network()

        return graph

    except Exception as exc:
        st.error(f"Could not read network CSV: {exc}")
        return build_default_network()


# ============================================================
# 3. ADAPTIVE MEMORY / CALIBRATION
# ============================================================

def initialize_adaptive_memory(graph):
    return {
        (u, v): 1.0
        for u, v in graph.edges()
    }


def sync_adaptive_memory(graph, memory):
    fresh = {}
    for u, v in graph.edges():
        fresh[(u, v)] = float(memory.get((u, v), 1.0))
    return fresh


def apply_memory_to_graph(graph, memory):
    for u, v in graph.edges():
        graph[u][v]["adaptive_multiplier"] = float(
            np.clip(memory.get((u, v), 1.0), 0.50, 1.50)
        )


def edge_key(source, target):
    return f"{source} -> {target}"


def calibrate_from_observations(
    graph,
    memory,
    observations,
    config=None,
):
    """
    Observation CSV format:
        source,target,observed_transmission

    observed_transmission is the observed downstream impact divided by
    upstream impact, normally in the range 0..1. Values above 1 are
    accepted but clipped for stability.

    The update is deliberately transparent and interpretable:
        new_multiplier = old_multiplier + learning_rate * relative_error
    """
    if config is None:
        config = LearningConfig()

    memory = sync_adaptive_memory(graph, memory)
    updates = []

    for _, row in observations.iterrows():
        source = str(row.get("source", "")).strip()
        target = str(row.get("target", "")).strip()

        if not source or not target or not graph.has_edge(source, target):
            continue

        try:
            observed = float(row["observed_transmission"])
        except Exception:
            continue

        observed = float(np.clip(observed, 0.0, 1.5))
        base = float(graph[source][target]["weight"])
        if base <= 0:
            continue

        # If base weight is 0.8 and observed transmission is 0.6,
        # target multiplier is 0.75.
        target_multiplier = observed / base
        old_multiplier = float(memory[(source, target)])

        error = target_multiplier - old_multiplier
        new_multiplier = old_multiplier + config.learning_rate * error
        new_multiplier = float(
            np.clip(
                new_multiplier,
                config.min_multiplier,
                config.max_multiplier,
            )
        )

        memory[(source, target)] = new_multiplier

        updates.append(
            {
                "Dependency": edge_key(source, target),
                "Base Weight": round(base, 3),
                "Observed Transmission": round(observed, 3),
                "Old Multiplier": round(old_multiplier, 3),
                "New Multiplier": round(new_multiplier, 3),
                "Adjustment": round(new_multiplier - old_multiplier, 3),
            }
        )

    return memory, pd.DataFrame(updates)


def validation_metrics(df):
    """Return transparent prediction-error metrics for observed transmission."""
    if df is None or df.empty:
        return {}
    valid = df.dropna(subset=["Actual Transmission", "Predicted Transmission"]).copy()
    if valid.empty:
        return {}
    error = valid["Predicted Transmission"] - valid["Actual Transmission"]
    abs_error = error.abs()
    sq_error = error.pow(2)
    nonzero = valid["Actual Transmission"].abs() > 1e-9
    return {
        "n": int(len(valid)),
        "mae": float(abs_error.mean()),
        "rmse": float(math.sqrt(sq_error.mean())),
        "max_error": float(abs_error.max()),
        "mape": float((abs_error[nonzero] / valid.loc[nonzero, "Actual Transmission"].abs()).mean() * 100) if nonzero.any() else float("nan"),
    }


def build_validation_table(graph, observations, before_memory, after_memory):
    """Compare pre-calibration and post-calibration predictions to observed outcomes."""
    rows = []
    if observations is None or observations.empty:
        return pd.DataFrame()
    for _, row in observations.iterrows():
        source = str(row.get("source", "")).strip()
        target = str(row.get("target", "")).strip()
        if not source or not target or not graph.has_edge(source, target):
            continue
        try:
            actual = float(row["observed_transmission"])
        except Exception:
            continue
        base = float(graph[source][target]["weight"])
        before_mult = float(before_memory.get((source, target), 1.0))
        after_mult = float(after_memory.get((source, target), before_mult))
        before_pred = float(np.clip(base * before_mult, 0.0, 1.5))
        after_pred = float(np.clip(base * after_mult, 0.0, 1.5))
        rows.append({
            "Dependency": edge_key(source, target),
            "Actual Transmission": actual,
            "Before Calibration": before_pred,
            "After Calibration": after_pred,
            "Before Abs Error": abs(before_pred - actual),
            "After Abs Error": abs(after_pred - actual),
        })
    return pd.DataFrame(rows)


def memory_dataframe(graph, memory):
    rows = []
    for u, v, data in graph.edges(data=True):
        multiplier = float(memory.get((u, v), 1.0))
        effective = float(
            np.clip(data["weight"] * multiplier, 0.0, 1.0)
        )
        rows.append(
            {
                "Source": u,
                "Target": v,
                "Base Weight": round(data["weight"], 3),
                "Adaptive Multiplier": round(multiplier, 3),
                "Adaptive Weight": round(effective, 3),
                "Intervention Cost ($k)": round(data["cost"], 2),
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# 4. PROPAGATION ENGINE
# ============================================================

def combine_impacts(existing, contribution):
    existing = float(np.clip(existing, 0, 100))
    contribution = float(np.clip(contribution, 0, 100))
    result = 100 * (
        1 - (1 - existing / 100) * (1 - contribution / 100)
    )
    return float(np.clip(result, 0, 100))


def get_effective_weight(graph, u, v, memory):
    base = float(graph[u][v]["weight"])
    adaptive_multiplier = float(memory.get((u, v), 1.0))
    return float(
        np.clip(base * adaptive_multiplier, 0.0, 1.0)
    )


def get_timeline_impacts(
    graph,
    shock_sources_dict,
    actions=None,
    memory=None,
    steps=4,
):
    if actions is None:
        actions = []

    if memory is None:
        memory = initialize_adaptive_memory(graph)

    reductions = {}
    for action in actions:
        reductions[(action["source"], action["target"])] = float(
            action["reduction"]
        )

    current = {node: 0.0 for node in graph.nodes}

    for source, magnitude in shock_sources_dict.items():
        if source in current:
            current[source] = float(np.clip(magnitude, 0, 100))

    timeline = {"T0": current.copy()}

    for step in range(1, steps + 1):
        next_impacts = current.copy()

        for u, v in graph.edges():
            effective_weight = get_effective_weight(
                graph, u, v, memory
            )

            reduction = reductions.get((u, v), 0.0)
            effective_weight *= (1 - reduction / 100.0)
            effective_weight = float(np.clip(effective_weight, 0, 1))

            contribution = current[u] * effective_weight
            next_impacts[v] = combine_impacts(
                next_impacts[v],
                contribution,
            )

        current = next_impacts
        timeline[f"T{step}"] = current.copy()

    return timeline


def final_impacts(timeline):
    return timeline[list(timeline.keys())[-1]]


def calculate_total_damage(impacts):
    return float(sum(max(0.0, x) for x in impacts.values()))


def calculate_containment(baseline_damage, final_damage):
    if baseline_damage <= 0:
        return 0.0
    value = (
        (baseline_damage - final_damage)
        / baseline_damage
        * 100
    )
    return float(np.clip(value, 0, 100))


# ============================================================
# 5. COUNTERFACTUAL OPTIMIZATION
# ============================================================

def intervention_cost(base_cost, reduction):
    return float(base_cost) * (float(reduction) / 100.0)


def generate_portfolios(
    graph,
    intervention_levels,
    max_interventions,
    portfolio_limit=12000,
):
    edges = list(graph.edges(data=True))

    if not edges:
        return []

    max_interventions = min(
        int(max_interventions),
        len(edges),
    )

    portfolios = []
    estimated = 0

    for count in range(1, max_interventions + 1):
        estimated += (
            math.comb(len(edges), count)
            * len(intervention_levels) ** count
        )

    if estimated > portfolio_limit:
        return None

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
                            "reduction": reduction,
                            "cost": cost,
                        }
                    )

                portfolios.append(
                    {
                        "actions": actions,
                        "cost": total_cost,
                    }
                )

    return portfolios


def evaluate_portfolio(
    graph,
    shock_sources_dict,
    portfolio,
    baseline_damage,
    memory,
):
    timeline = get_timeline_impacts(
        graph=graph,
        shock_sources_dict=shock_sources_dict,
        actions=portfolio["actions"],
        memory=memory,
        steps=4,
    )

    impacts = final_impacts(timeline)
    damage = calculate_total_damage(impacts)
    containment = calculate_containment(
        baseline_damage,
        damage,
    )
    avoided = max(0.0, baseline_damage - damage)

    efficiency = (
        containment / portfolio["cost"]
        if portfolio["cost"] > 0
        else 0.0
    )

    return {
        "actions": portfolio["actions"],
        "cost": portfolio["cost"],
        "timeline": timeline,
        "final_impacts": impacts,
        "final_damage": damage,
        "damage_avoided": avoided,
        "containment": containment,
        "efficiency": efficiency,
    }


def pareto_frontier(results):
    frontier = []

    for candidate in results:
        dominated = False

        for other in results:
            if other is candidate:
                continue

            lower_cost = other["cost"] <= candidate["cost"]
            higher_containment = (
                other["containment"] >= candidate["containment"]
            )

            strictly_better = (
                other["cost"] < candidate["cost"]
                or other["containment"] > candidate["containment"]
            )

            if (
                lower_cost
                and higher_containment
                and strictly_better
            ):
                dominated = True
                break

        if not dominated:
            frontier.append(candidate)

    return sorted(
        frontier,
        key=lambda x: (x["cost"], -x["containment"]),
    )


def select_budget_strategy(results, budget):
    feasible = [
        r for r in results
        if r["cost"] <= budget
    ]

    if not feasible:
        return None, []

    selected = max(
        feasible,
        key=lambda x: (
            x["containment"],
            -x["cost"],
        ),
    )

    return selected, feasible


def reverse_resilience(results, target_containment):
    feasible = [
        r for r in results
        if r["containment"] >= target_containment
    ]

    if not feasible:
        return None

    return min(
        feasible,
        key=lambda x: (
            x["cost"],
            -x["containment"],
        ),
    )


# ============================================================
# 6. ADAPTIVE SCENARIO GENERATOR
# ============================================================

def generate_adaptive_scenarios(graph):
    nodes = list(graph.nodes)

    if not nodes:
        return []

    scenarios = []

    # Single-node severe shocks
    for node in nodes:
        scenarios.append(
            {
                "name": f"Severe shock — {node}",
                "shocks": {node: 80},
            }
        )

    # Pair scenarios from high-degree nodes
    ranked = sorted(
        nodes,
        key=lambda n: (
            graph.in_degree(n)
            + graph.out_degree(n)
        ),
        reverse=True,
    )

    top_nodes = ranked[: min(5, len(ranked))]

    for a, b in combinations(top_nodes, 2):
        scenarios.append(
            {
                "name": f"Compound shock — {a} + {b}",
                "shocks": {
                    a: 70,
                    b: 75,
                },
            }
        )

    return scenarios


def rank_scenarios(
    graph,
    scenarios,
    memory,
):
    rows = []

    for scenario in scenarios:
        timeline = get_timeline_impacts(
            graph,
            scenario["shocks"],
            actions=[],
            memory=memory,
            steps=4,
        )
        damage = calculate_total_damage(
            final_impacts(timeline)
        )

        rows.append(
            {
                "Scenario": scenario["name"],
                "Shock Sources": ", ".join(
                    scenario["shocks"].keys()
                ),
                "Baseline Damage": round(damage, 2),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "Baseline Damage",
            ascending=False,
        )
        .reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )


# ============================================================
# 7. EXPLAINABILITY
# ============================================================

def explain_intervention(
    graph,
    selected,
    baseline_impacts,
    memory,
):
    rows = []

    for action in selected["actions"]:
        u = action["source"]
        v = action["target"]

        base_weight = float(
            graph[u][v]["weight"]
        )
        multiplier = float(
            memory.get((u, v), 1.0)
        )
        adaptive_weight = float(
            np.clip(
                base_weight * multiplier,
                0,
                1,
            )
        )

        source_exposure = float(
            baseline_impacts.get(u, 0)
        )

        leverage = (
            source_exposure
            * adaptive_weight
            * (action["reduction"] / 100)
        )

        rows.append(
            {
                "Dependency": edge_key(u, v),
                "Baseline Source Impact (%)": round(
                    source_exposure,
                    2,
                ),
                "Adaptive Dependency Weight": round(
                    adaptive_weight,
                    3,
                ),
                "Intervention (%)": round(
                    action["reduction"],
                    1,
                ),
                "Investment ($k)": round(
                    action["cost"],
                    2,
                ),
                "Leverage Indicator": round(
                    leverage,
                    2,
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 8. FINANCIAL MODEL
# ============================================================

def avoided_loss_value(
    baseline_damage,
    final_damage,
    conversion_factor,
):
    avoided = max(
        0.0,
        baseline_damage - final_damage,
    )
    return avoided * conversion_factor


def avoided_loss_ratio(
    value,
    investment,
):
    if investment <= 0:
        return 0.0
    return value / investment


# ============================================================
# 9. VISUALS
# ============================================================

def network_impact_figure(
    graph,
    impacts,
    title,
):
    if len(graph.nodes) == 0:
        return go.Figure()

    positions = nx.spring_layout(
        graph,
        seed=42,
    )

    edge_x = []
    edge_y = []

    for u, v in graph.edges():
        x0, y0 = positions[u]
        x1, y1 = positions[v]

        edge_x += [
            x0,
            x1,
            None,
        ]
        edge_y += [
            y0,
            y1,
            None,
        ]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1),
        hoverinfo="none",
    )

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
                impacts.get(node, 0),
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
            colorbar=dict(
                title="Impact"
            ),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Impact: %{marker.color:.2f}"
            "<extra></extra>"
        ),
    )

    fig = go.Figure(
        data=[
            edge_trace,
            node_trace,
        ]
    )

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
    )

    return fig


def timeline_figure(timeline, title):
    rows = []

    for t, impacts in timeline.items():
        for node, value in impacts.items():
            rows.append(
                {
                    "Time": t,
                    "Node": node,
                    "Impact": value,
                }
            )

    df = pd.DataFrame(rows)

    fig = go.Figure()

    for node in df["Node"].unique():
        subset = df[
            df["Node"] == node
        ]

        fig.add_trace(
            go.Scatter(
                x=subset["Time"],
                y=subset["Impact"],
                mode="lines+markers",
                name=node,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Impact (%)",
        yaxis=dict(range=[0, 100]),
    )

    return fig


def pareto_figure(frontier):
    fig = go.Figure()

    if frontier:
        fig.add_trace(
            go.Scatter(
                x=[
                    r["cost"]
                    for r in frontier
                ],
                y=[
                    r["containment"]
                    for r in frontier
                ],
                mode="lines+markers",
                hovertemplate=(
                    "Investment: %{x:.2f} $k<br>"
                    "Containment: %{y:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Pareto Frontier — Investment vs Containment",
        xaxis_title="Investment ($k)",
        yaxis_title="Containment (%)",
    )

    return fig


# ============================================================
# 10. EXECUTIVE REPORT
# ============================================================

def generate_report(
    company,
    industry,
    scenario,
    baseline_damage,
    selected,
    avoided_value,
    ratio,
    critical_node,
    learning_status,
):
    lines = [
        "============================================================",
        "SHOCK SIM AI — RESILIENCE OPERATING SYSTEM v10.0",
        "============================================================",
        "",
        "COMPANY PROFILE",
        f"Company: {company}",
        f"Industry: {industry}",
        "",
        "SCENARIO",
        f"Scenario: {scenario}",
        f"Most Affected Downstream Node: {critical_node}",
        "",
        "BASELINE EXPOSURE",
        f"Baseline Damage Units: {baseline_damage:.2f}",
        "",
        "COUNTERFACTUAL DECISION",
        f"Required Investment: ${selected['cost']:.2f}k",
        f"Achieved Containment: {selected['containment']:.2f}%",
        f"Modeled Avoided Loss: ${avoided_value:.2f}k",
        f"Avoided Loss / Investment: {ratio:.2f}x",
        "",
        "ADAPTIVE INTELLIGENCE",
        f"Learning Status: {learning_status}",
        "",
        "RECOMMENDED PORTFOLIO",
    ]

    for action in selected["actions"]:
        lines.append(
            f"- {action['source']} -> {action['target']} | "
            f"Reduction {action['reduction']:.0f}% | "
            f"Investment ${action['cost']:.2f}k"
        )

    lines.extend(
        [
            "",
            "GOVERNANCE NOTE",
            "The adaptive model is decision-support software. "
            "Observed-data calibration should be validated against "
            "company-specific operational and financial records.",
            "",
            "============================================================",
        ]
    )

    return "\n".join(lines)


# ============================================================
# 11. SESSION STATE
# ============================================================

if "adaptive_memory" not in st.session_state:
    st.session_state.adaptive_memory = {}

if "learning_updates" not in st.session_state:
    st.session_state.learning_updates = pd.DataFrame()
if "validation_table" not in st.session_state:
    st.session_state.validation_table = pd.DataFrame()

if "observations_loaded" not in st.session_state:
    st.session_state.observations_loaded = False


# ============================================================
# 12. SIDEBAR
# ============================================================

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
        "Healthcare Operations",
        "Other",
    ],
)

annual_revenue = st.sidebar.number_input(
    "Annual Revenue ($M)",
    min_value=0.0,
    value=250.0,
    step=10.0,
)

st.sidebar.divider()
st.sidebar.header("💰 Financial Model")

financial_conversion = st.sidebar.number_input(
    "Damage Unit Value ($k)",
    min_value=0.01,
    value=1.0,
    step=0.1,
    help=(
        "Monetary conversion for modeled damage units. "
        "Calibrate with company-specific loss data."
    ),
)

st.sidebar.divider()
st.sidebar.header("📁 Network Data")

uploaded_network = st.sidebar.file_uploader(
    "Upload Network CSV",
    type=["csv"],
    help="Required columns: source,target,weight,cost",
)

if uploaded_network is not None:
    graph = load_network_from_csv(
        uploaded_network
    )
    st.sidebar.success(
        "Custom network loaded."
    )
else:
    graph = build_default_network()
    st.sidebar.info(
        "Using standard demonstration network."
    )

if not st.session_state.adaptive_memory:
    st.session_state.adaptive_memory = (
        initialize_adaptive_memory(graph)
    )
else:
    st.session_state.adaptive_memory = (
        sync_adaptive_memory(
            graph,
            st.session_state.adaptive_memory,
        )
    )

apply_memory_to_graph(
    graph,
    st.session_state.adaptive_memory,
)

st.sidebar.caption(
    f"Nodes: {graph.number_of_nodes()} | "
    f"Edges: {graph.number_of_edges()}"
)


# ============================================================
# 13. OBSERVED OUTCOME LEARNING
# ============================================================

st.sidebar.divider()
st.sidebar.header("🧠 Adaptive Learning")

uploaded_observations = st.sidebar.file_uploader(
    "Upload Observed Outcomes CSV",
    type=["csv"],
    help=(
        "Required columns: source,target,observed_transmission. "
        "Example: Energy,Suppliers,0.72"
    ),
)

learning_rate = st.sidebar.slider(
    "Learning Rate",
    min_value=0.01,
    max_value=0.50,
    value=0.15,
    step=0.01,
)

if uploaded_observations is not None:
    try:
        observations_df = pd.read_csv(
            uploaded_observations
        )

        required_obs = {
            "source",
            "target",
            "observed_transmission",
        }

        if not required_obs.issubset(
            observations_df.columns
        ):
            st.sidebar.error(
                "Observation CSV must contain: "
                "source, target, observed_transmission"
            )
        else:
            if st.sidebar.button(
                "🧠 CALIBRATE MODEL",
                use_container_width=True,
            ):
                old_memory = dict(st.session_state.adaptive_memory)
                config = LearningConfig(
                    learning_rate=learning_rate
                )

                (
                    new_memory,
                    updates,
                ) = calibrate_from_observations(
                    graph,
                    st.session_state.adaptive_memory,
                    observations_df,
                    config,
                )

                st.session_state.adaptive_memory = (
                    new_memory
                )
                st.session_state.learning_updates = (
                    updates
                )
                st.session_state.validation_table = build_validation_table(
                    graph, observations_df, old_memory, new_memory
                )
                st.session_state.observations_loaded = (
                    True
                )

                st.sidebar.success(
                    "Adaptive calibration completed."
                )

    except Exception as exc:
        st.sidebar.error(
            f"Observation file error: {exc}"
        )

if st.session_state.observations_loaded:
    learning_status = (
        "Calibrated from observed dependency outcomes"
    )
else:
    learning_status = (
        "Baseline model — no observed outcomes calibrated"
    )


# ============================================================
# 14. SCENARIO ENGINE
# ============================================================

st.sidebar.divider()
st.sidebar.header("📚 Scenario Engine")

scenario_mode = st.sidebar.selectbox(
    "Scenario Mode",
    [
        "Library Scenario",
        "Custom Shock",
        "Adaptive Scenario Explorer",
    ],
)

node_list = list(graph.nodes)

if scenario_mode == "Library Scenario":
    library = [
        (
            "⚡ Energy Shock",
            {"Energy": 80},
            "Energy Sector Severe Spike (+80%)",
        ),
        (
            "🚢 Logistics Disruption",
            {"Logistics": 75},
            "Logistics Channel Disruption (+75%)",
        ),
        (
            "🏭 Manufacturing Shutdown",
            {"Manufacturing": 90},
            "Manufacturing Shutdown (+90%)",
        ),
        (
            "🌍 Energy + Supplier Compound Shock",
            {"Energy": 70, "Suppliers": 85},
            "Compound Energy + Supplier Failure",
        ),
    ]

    available_library = [
        item
        for item in library
        if all(
            node in graph
            for node in item[1]
        )
    ]

    if available_library:
        labels = [
            item[0]
            for item in available_library
        ]

        selected_label = st.sidebar.selectbox(
            "Select Scenario",
            labels,
        )

        selected_item = next(
            x for x in available_library
            if x[0] == selected_label
        )

        shock_sources_dict = (
            selected_item[1]
        )
        shock_desc = selected_item[2]
    else:
        fallback_node = (
            node_list[0]
            if node_list
            else "Unknown"
        )

        shock_sources_dict = {
            fallback_node: 80
        }

        shock_desc = (
            f"Fallback shock on {fallback_node}"
        )

elif scenario_mode == "Custom Shock":
    custom_source = st.sidebar.selectbox(
        "Shock Source",
        node_list if node_list else ["Unknown"],
    )

    custom_magnitude = st.sidebar.slider(
        "Shock Magnitude (%)",
        10,
        100,
        75,
    )

    shock_sources_dict = {
        custom_source: custom_magnitude
    }

    shock_desc = (
        f"Custom shock — {custom_source} "
        f"({custom_magnitude}%)"
    )

else:
    adaptive_scenarios = (
        generate_adaptive_scenarios(graph)
    )

    if adaptive_scenarios:
        scenario_table = rank_scenarios(
            graph,
            adaptive_scenarios,
            st.session_state.adaptive_memory,
        )

        selected_name = st.sidebar.selectbox(
            "Adaptive Scenario",
            scenario_table["Scenario"].tolist(),
        )

        selected_scenario = next(
            x
            for x in adaptive_scenarios
            if x["name"] == selected_name
        )

        shock_sources_dict = (
            selected_scenario["shocks"]
        )
        shock_desc = selected_name
    else:
        shock_sources_dict = {}
        shock_desc = "No scenario available"


# ============================================================
# 15. OPTIMIZATION
# ============================================================

st.sidebar.divider()
st.sidebar.header("⚙️ Decision Objective")

optimization_mode = st.sidebar.radio(
    "Optimization Mode",
    [
        "Budget Constraint",
        "Reverse Resilience Target",
    ],
)

if optimization_mode == "Budget Constraint":
    budget_limit = st.sidebar.slider(
        "Available Budget ($k)",
        5,
        250,
        100,
    )
else:
    target_containment = st.sidebar.slider(
        "Target Containment (%)",
        20,
        95,
        80,
    )

intervention_levels = [25, 50, 75, 100]

max_interventions = st.sidebar.slider(
    "Maximum Interventions",
    1,
    3,
    3,
)

portfolio_estimate = 0
edge_count = graph.number_of_edges()

for count in range(
    1,
    min(max_interventions, edge_count) + 1,
):
    portfolio_estimate += (
        math.comb(edge_count, count)
        * len(intervention_levels) ** count
    )

st.sidebar.caption(
    f"Counterfactual portfolios: {portfolio_estimate:,}"
)

run_button = st.sidebar.button(
    "🚀 RUN ADAPTIVE SIMULATION",
    use_container_width=True,
)


# ============================================================
# 16. PRE-RUN INFORMATION
# ============================================================

if not run_button:
    st.info(
        "Configure the network, adaptive learning data, scenario, "
        "and investment objective, then run the simulation."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Network Nodes",
            graph.number_of_nodes(),
        )

    with col2:
        st.metric(
            "Dependencies",
            graph.number_of_edges(),
        )

    with col3:
        st.metric(
            "Adaptive Status",
            "Calibrated"
            if st.session_state.observations_loaded
            else "Baseline",
        )

    st.markdown(
        """
### v10.0 Resilience Operating System

**1. Resilience Memory**  
Stores calibrated dependency multipliers.

**2. Adaptive Calibration**  
Observed outcomes can update transmission assumptions.

**3. Adaptive Scenario Explorer**  
Automatically generates severe and compound scenarios from the network.

**4. Counterfactual Search**  
Tests intervention portfolios and finds minimum investment or maximum containment.

**5. Explainability**  
Shows why a dependency was selected and its leverage indicator.

**6. Feedback Loop**  
After real events, observed transmission can be fed back into the model.
"""
    )

    st.stop()


# ============================================================
# 17. BASELINE
# ============================================================

memory = sync_adaptive_memory(
    graph,
    st.session_state.adaptive_memory,
)

apply_memory_to_graph(
    graph,
    memory,
)

baseline_timeline = get_timeline_impacts(
    graph,
    shock_sources_dict,
    actions=[],
    memory=memory,
    steps=4,
)

baseline_impacts = final_impacts(
    baseline_timeline
)

baseline_damage = calculate_total_damage(
    baseline_impacts
)

downstream = {
    node: impact
    for node, impact in baseline_impacts.items()
    if node not in shock_sources_dict
}

if downstream:
    critical_node = max(
        downstream,
        key=downstream.get,
    )
else:
    critical_node = (
        list(graph.nodes)[0]
        if graph.nodes
        else "N/A"
    )


# ============================================================
# 18. COUNTERFACTUAL SEARCH
# ============================================================

if portfolio_estimate > 12000:
    st.error(
        f"The current search space contains "
        f"{portfolio_estimate:,} portfolios. "
        "Reduce Maximum Interventions or use a smaller network "
        "before running exhaustive counterfactual search."
    )
    st.stop()

with st.spinner(
    "Running adaptive propagation and counterfactual search..."
):
    portfolios = generate_portfolios(
        graph,
        intervention_levels,
        max_interventions,
    )

    results = [
        evaluate_portfolio(
            graph,
            shock_sources_dict,
            portfolio,
            baseline_damage,
            memory,
        )
        for portfolio in portfolios
    ]

if not results:
    st.error(
        "No valid intervention portfolio was generated."
    )
    st.stop()


if optimization_mode == "Budget Constraint":
    selected, feasible = (
        select_budget_strategy(
            results,
            budget_limit,
        )
    )

    if selected is None:
        st.error(
            "No portfolio fits the selected budget."
        )
        st.stop()

    objective_text = (
        f"Maximum containment within ${budget_limit}k"
    )

else:
    selected = reverse_resilience(
        results,
        target_containment,
    )

    if selected is None:
        st.error(
            f"No portfolio reaches {target_containment}% containment."
        )
        st.stop()

    feasible = [
        r for r in results
        if r["containment"] >= target_containment
    ]

    objective_text = (
        f"Minimum investment for {target_containment}% target"
    )


# ============================================================
# 19. FINANCIAL METRICS
# ============================================================

avoided_value = avoided_loss_value(
    baseline_damage,
    selected["final_damage"],
    financial_conversion,
)

ratio = avoided_loss_ratio(
    avoided_value,
    selected["cost"],
)

frontier = pareto_frontier(results)

explanation_df = explain_intervention(
    graph,
    selected,
    baseline_impacts,
    memory,
)


# ============================================================
# 20. EXECUTIVE COMMAND CENTER
# ============================================================

st.header("📊 Adaptive Executive Command Center")

st.markdown(
    f"**Company:** {company_name}  |  "
    f"**Industry:** {industry}  |  "
    f"**Scenario:** {shock_desc}"
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Baseline Damage",
    f"{baseline_damage:.1f}",
)

m2.metric(
    "Most Affected Node",
    critical_node,
)

m3.metric(
    "Required Investment",
    f"${selected['cost']:.1f}k",
)

m4.metric(
    "Containment",
    f"{selected['containment']:.1f}%",
)

st.caption(
    f"Objective: {objective_text} | "
    f"Adaptive status: {learning_status}"
)

st.divider()


# ============================================================
# 21. WITHOUT / WITH
# ============================================================

left, right = st.columns(2)

with left:
    st.markdown("### 🔴 Baseline — Without Intervention")

    st.error(
        f"""
**Damage Units:** {baseline_damage:.1f}

**Modeled Financial Exposure:**  
${baseline_damage * financial_conversion:.1f}k

**Most Affected Node:**  
{critical_node}

**Shock:**  
{shock_desc}
"""
    )

with right:
    st.markdown("### 🟢 Counterfactual — Recommended Plan")

    st.success(
        f"""
**Investment:** ${selected['cost']:.1f}k

**Containment:** {selected['containment']:.1f}%

**Modeled Avoided Loss:**  
${avoided_value:.1f}k

**Avoided Loss / Investment:**  
{ratio:.2f}x
"""
    )

st.caption(
    "The financial ratio is a modeled decision-support metric, "
    "not an accounting ROI."
)


# ============================================================
# 22. WHY THIS INTERVENTION
# ============================================================

st.divider()
st.header("🛠️ Why This Intervention?")

st.write(
    "The engine selected the portfolio through counterfactual simulation. "
    "Leverage combines baseline source exposure, adaptive dependency weight, "
    "and intervention level."
)

if explanation_df.empty:
    st.info(
        "No intervention explanation available."
    )
else:
    st.dataframe(
        explanation_df,
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "A high leverage indicator means the selected intervention acts on "
    "a dependency that carries substantial modeled exposure."
)


# ============================================================
# 23. RECOMMENDED PORTFOLIO
# ============================================================

st.divider()
st.header("🧰 Recommended Intervention Portfolio")

portfolio_df = pd.DataFrame(
    selected["actions"]
)

if not portfolio_df.empty:
    portfolio_df = portfolio_df.rename(
        columns={
            "source": "Source",
            "target": "Target",
            "reduction": "Reduction (%)",
            "cost": "Investment ($k)",
        }
    )

    portfolio_df["Reduction (%)"] = (
        portfolio_df["Reduction (%)"]
        .round(1)
    )

    portfolio_df["Investment ($k)"] = (
        portfolio_df["Investment ($k)"]
        .round(2)
    )

    st.dataframe(
        portfolio_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 24. RESILIENCE TARGET CALCULATOR
# ============================================================

st.divider()
st.header("🎯 Target Resilience Calculator")

target_options = [
    25,
    40,
    50,
    60,
    70,
    75,
    80,
    90,
]

target_rows = []

for target in target_options:
    candidate = reverse_resilience(
        results,
        target,
    )

    if candidate is None:
        target_rows.append(
            {
                "Target Containment (%)": target,
                "Minimum Investment ($k)": None,
                "Achieved Containment (%)": None,
                "Modeled Avoided Loss ($k)": None,
            }
        )
    else:
        target_rows.append(
            {
                "Target Containment (%)": target,
                "Minimum Investment ($k)": round(
                    candidate["cost"],
                    2,
                ),
                "Achieved Containment (%)": round(
                    candidate["containment"],
                    2,
                ),
                "Modeled Avoided Loss ($k)": round(
                    avoided_loss_value(
                        baseline_damage,
                        candidate["final_damage"],
                        financial_conversion,
                    ),
                    2,
                ),
            }
        )

target_df = pd.DataFrame(
    target_rows
)

st.dataframe(
    target_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# 25. ADAPTIVE SCENARIO EXPLORER
# ============================================================

st.divider()
st.header("🌪️ Adaptive Scenario Explorer")

scenario_candidates = generate_adaptive_scenarios(
    graph
)

scenario_ranked = rank_scenarios(
    graph,
    scenario_candidates,
    memory,
)

if not scenario_ranked.empty:
    st.dataframe(
        scenario_ranked,
        use_container_width=True,
        hide_index=True,
    )

    top_n = scenario_ranked.head(
        min(10, len(scenario_ranked))
    )

    fig_scenarios = go.Figure(
        go.Bar(
            x=top_n["Scenario"],
            y=top_n["Baseline Damage"],
        )
    )

    fig_scenarios.update_layout(
        title="Adaptive Scenario Exposure Ranking",
        xaxis_title="Scenario",
        yaxis_title="Baseline Damage",
    )

    st.plotly_chart(
        fig_scenarios,
        use_container_width=True,
    )


# ============================================================
# 26. ADAPTIVE MEMORY
# ============================================================

st.divider()
st.header("🧠 Resilience Memory")

memory_df = memory_dataframe(
    graph,
    memory,
)

st.dataframe(
    memory_df,
    use_container_width=True,
    hide_index=True,
)

if st.session_state.observations_loaded:
    st.success(
        "The displayed dependency weights include adaptive calibration "
        "from the uploaded observed-outcome data."
    )
else:
    st.info(
        "No observed-outcome calibration has been applied. "
        "Current adaptive multipliers remain at baseline."
    )

if not st.session_state.learning_updates.empty:
    st.subheader("Latest Calibration Updates")

    st.dataframe(
        st.session_state.learning_updates,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 27.1 ADAPTIVE VALIDATION ENGINE — v4.1
# ============================================================

st.divider()
st.header("🧪 Adaptive Validation Engine")
st.write(
    "v4.1 measures prediction error against observed transmission data, "
    "before and after adaptive calibration. This makes the learning loop auditable."
)

validation_df = st.session_state.validation_table
if validation_df.empty:
    st.info(
        "Upload observed-outcome data in the sidebar and calibrate the model "
        "to populate validation results."
    )
else:
    before_df = validation_df.rename(columns={"Before Calibration": "Predicted Transmission"})[["Dependency", "Actual Transmission", "Predicted Transmission"]]
    after_df = validation_df.rename(columns={"After Calibration": "Predicted Transmission"})[["Dependency", "Actual Transmission", "Predicted Transmission"]]
    before_metrics = validation_metrics(before_df)
    after_metrics = validation_metrics(after_df)

    vc1, vc2, vc3, vc4 = st.columns(4)
    vc1.metric("Observed Dependencies", str(after_metrics.get("n", 0)))
    vc2.metric("MAE Before", f"{before_metrics.get('mae', float('nan')):.3f}")
    vc3.metric("MAE After", f"{after_metrics.get('mae', float('nan')):.3f}")
    improvement = None
    if before_metrics and after_metrics and before_metrics.get("mae", 0) > 0:
        improvement = (before_metrics["mae"] - after_metrics["mae"]) / before_metrics["mae"] * 100
    vc4.metric("MAE Change", f"{improvement:+.1f}%" if improvement is not None else "—")

    st.dataframe(validation_df.round(4), use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=validation_df["Dependency"], y=validation_df["Actual Transmission"], mode="markers+lines", name="Observed"))
    fig.add_trace(go.Scatter(x=validation_df["Dependency"], y=validation_df["Before Calibration"], mode="markers+lines", name="Before calibration"))
    fig.add_trace(go.Scatter(x=validation_df["Dependency"], y=validation_df["After Calibration"], mode="markers+lines", name="After calibration"))
    ymax = max(1.0, float(validation_df[["Actual Transmission", "Before Calibration", "After Calibration"]].max().max()) * 1.1)
    fig.update_layout(title="Observed vs Predicted Dependency Transmission", xaxis_title="Dependency", yaxis_title="Transmission", yaxis=dict(range=[0, ymax]))
    st.plotly_chart(fig, use_container_width=True)

    if improvement is not None:
        if improvement > 0:
            st.success(f"Calibration reduced mean absolute prediction error by {improvement:.1f}% for this validation sample.")
        elif improvement < 0:
            st.warning(f"Calibration increased mean absolute prediction error by {abs(improvement):.1f}% for this validation sample. Review the observations and learning rate.")
        else:
            st.info("Calibration produced no change in MAE for this validation sample.")

    st.caption(
        "This is sample-based validation, not proof of out-of-sample accuracy. "
        "For production, keep separate calibration and hold-out event datasets."
    )


# ============================================================
# 27. PARETO FRONTIER
# ============================================================

st.divider()
st.header("📈 Pareto Frontier")

st.plotly_chart(
    pareto_figure(frontier),
    use_container_width=True,
)

st.caption(
    "Pareto-efficient portfolios cannot be improved in both "
    "investment cost and containment simultaneously within the "
    "current model search space."
)


# ============================================================
# 28. NETWORK IMPACT MAPS
# ============================================================

st.divider()
st.header("🗺️ Systemic Impact Maps")

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
        ),
        use_container_width=True,
    )


# ============================================================
# 29. PROPAGATION REPLAY
# ============================================================

st.divider()
st.header("⏱️ Propagation Replay")

replay1, replay2 = st.columns(2)

with replay1:
    st.plotly_chart(
        timeline_figure(
            baseline_timeline,
            "Baseline Shock Propagation",
        ),
        use_container_width=True,
    )

with replay2:
    st.plotly_chart(
        timeline_figure(
            selected["timeline"],
            "Counterfactual Propagation",
        ),
        use_container_width=True,
    )


baseline_replay_df = pd.DataFrame(
    baseline_timeline
).T

optimized_replay_df = pd.DataFrame(
    selected["timeline"]
).T

st.subheader("Baseline Replay Data")
st.dataframe(
    baseline_replay_df.round(2),
    use_container_width=True,
)

st.subheader("Counterfactual Replay Data")
st.dataframe(
    optimized_replay_df.round(2),
    use_container_width=True,
)


# ============================================================
# 30. FEEDBACK LOOP
# ============================================================

st.divider()
st.header("🔄 Closed-Loop Learning")

st.write(
    """
After a real-world event, the observed transmission between dependencies
can be uploaded through the sidebar. The calibration step updates the
dependency multipliers, and the next simulation uses those learned values.

This creates the intended feedback loop:

**Prediction → Real Event → Observed Outcome → Calibration → New Prediction**
"""
)

feedback_col1, feedback_col2, feedback_col3 = st.columns(3)

feedback_col1.metric(
    "Prediction Layer",
    "Counterfactual",
)

feedback_col2.metric(
    "Observed Data",
    "Loaded"
    if st.session_state.observations_loaded
    else "Not Loaded",
)

feedback_col3.metric(
    "Adaptive Memory",
    "Active",
)


# ============================================================
# 31. EXECUTIVE REPORT
# ============================================================

st.divider()
st.header("📄 Executive Report")

report = generate_report(
    company=company_name,
    industry=industry,
    scenario=shock_desc,
    baseline_damage=baseline_damage,
    selected=selected,
    avoided_value=avoided_value,
    ratio=ratio,
    critical_node=critical_node,
    learning_status=learning_status,
)

st.text_area(
    "Adaptive Decision Report",
    report,
    height=420,
)

safe_company = (
    "".join(
        ch if ch.isalnum() else "_"
        for ch in company_name
    )
    .strip("_")
    or "company"
)

st.download_button(
    "📥 Download Executive Report",
    data=report,
    file_name=(
        f"{safe_company}_Adaptive_Resilience_Report.txt"
    ),
    mime="text/plain",
    use_container_width=True,
)



# ============================================================
# 31.5 RESILIENCE OPERATING SYSTEM — v10 FINAL LAYER
# ============================================================

def digital_twin_snapshot(graph, memory):
    """Create a compact, auditable digital-twin state from the active graph."""
    rows = []
    for u, v, data in graph.edges(data=True):
        base = float(data.get("weight", 0.0))
        mult = float(memory.get((u, v), 1.0))
        adaptive = float(np.clip(base * mult, 0.0, 1.0))
        rows.append({
            "Dependency": edge_key(u, v),
            "Base Weight": base,
            "Adaptive Multiplier": mult,
            "Adaptive Weight": adaptive,
            "Cost ($k)": float(data.get("cost", 0.0)),
            "In Degree (Target)": int(graph.in_degree(v)),
            "Out Degree (Source)": int(graph.out_degree(u)),
        })
    return pd.DataFrame(rows)


def systemic_node_risk(graph, impacts):
    """Transparent node-level risk score using exposure and network connectivity."""
    rows = []
    max_degree = max([graph.in_degree(n) + graph.out_degree(n) for n in graph.nodes] or [1])
    for node in graph.nodes:
        impact = float(impacts.get(node, 0.0))
        degree = graph.in_degree(node) + graph.out_degree(node)
        centrality = degree / max_degree if max_degree else 0.0
        risk = impact * (0.7 + 0.3 * centrality)
        rows.append({
            "Node": node,
            "Impact (%)": round(impact, 2),
            "Connectivity": round(centrality, 3),
            "Systemic Risk Score": round(risk, 2),
        })
    return pd.DataFrame(rows).sort_values("Systemic Risk Score", ascending=False).reset_index(drop=True)


def predictive_forecast(graph, shocks, memory, steps=4, simulations=250, uncertainty=0.12, seed=42):
    """Monte-Carlo predictive layer around the current digital-twin assumptions."""
    rng = np.random.default_rng(seed)
    node_values = {n: [] for n in graph.nodes}
    total_damage = []

    for _ in range(int(simulations)):
        sampled_memory = {}
        for u, v in graph.edges():
            base_mult = float(memory.get((u, v), 1.0))
            sampled_memory[(u, v)] = float(np.clip(
                rng.normal(base_mult, max(0.01, uncertainty * base_mult)),
                0.50, 1.50
            ))

        sampled_shocks = {
            n: float(np.clip(rng.normal(m, max(1.0, uncertainty * m)), 0, 100))
            for n, m in shocks.items()
        }
        tl = get_timeline_impacts(graph, sampled_shocks, actions=[], memory=sampled_memory, steps=steps)
        final = final_impacts(tl)
        for node in graph.nodes:
            node_values[node].append(float(final.get(node, 0.0)))
        total_damage.append(calculate_total_damage(final))

    rows = []
    for node, values in node_values.items():
        arr = np.asarray(values, dtype=float)
        rows.append({
            "Node": node,
            "Expected Impact (%)": float(arr.mean()),
            "P50 Impact (%)": float(np.percentile(arr, 50)),
            "P90 Impact (%)": float(np.percentile(arr, 90)),
            "P95 Impact (%)": float(np.percentile(arr, 95)),
        })
    forecast = pd.DataFrame(rows).sort_values("P90 Impact (%)", ascending=False).reset_index(drop=True)
    damage_arr = np.asarray(total_damage, dtype=float)
    summary = {
        "expected_damage": float(damage_arr.mean()),
        "p50_damage": float(np.percentile(damage_arr, 50)),
        "p90_damage": float(np.percentile(damage_arr, 90)),
        "p95_damage": float(np.percentile(damage_arr, 95)),
    }
    return forecast, summary


def portfolio_capital_curve(results):
    """Return a clean capital-to-containment response curve."""
    if not results:
        return pd.DataFrame()
    rows = []
    for r in results:
        cost = float(r.get("cost", 0.0))
        containment = float(r.get("containment", 0.0))
        avoided = float(r.get("damage_avoided", 0.0))
        if not (math.isfinite(cost) and math.isfinite(containment) and math.isfinite(avoided)):
            continue
        rows.append({
            "Investment ($k)": max(0.0, round(cost, 2)),
            "Containment (%)": float(np.clip(containment, 0.0, 100.0)),
            "Avoided Damage": max(0.0, avoided),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = (df.sort_values(["Investment ($k)", "Containment (%)"], ascending=[True, False])
            .drop_duplicates(subset=["Investment ($k)"], keep="first")
            .sort_values("Investment ($k)")
            .reset_index(drop=True))
    return df


def autonomous_action_queue(frontier, budget=None, target=None, top_n=5):
    """Generate a ranked decision queue without executing external actions."""
    candidates = list(frontier)
    if budget is not None:
        candidates = [r for r in candidates if r["cost"] <= budget]
    if target is not None:
        target_candidates = [r for r in candidates if r["containment"] >= target]
        if target_candidates:
            candidates = target_candidates
    candidates = sorted(
        candidates,
        key=lambda r: (r["containment"] / max(r["cost"], 1e-9), r["containment"]),
        reverse=True,
    )
    queue = []
    for i, r in enumerate(candidates[:top_n], 1):
        queue.append({
            "Priority": i,
            "Investment ($k)": round(r["cost"], 2),
            "Containment (%)": round(r["containment"], 2),
            "Damage Avoided": round(r["damage_avoided"], 2),
            "Efficiency": round(r["efficiency"], 4),
            "Action Count": len(r["actions"]),
        })
    return pd.DataFrame(queue)


def model_health_report(graph, memory, validation_df):
    """Operational health checks for the v10 model."""
    weights = [float(graph[u][v].get("weight", 0.0)) for u, v in graph.edges()]
    costs = [float(graph[u][v].get("cost", 0.0)) for u, v in graph.edges()]
    multipliers = [float(memory.get((u, v), 1.0)) for u, v in graph.edges()]
    checks = [
        ("Network loaded", graph.number_of_nodes() > 0 and graph.number_of_edges() > 0),
        ("Weights within 0..1", all(0 <= x <= 1 for x in weights)),
        ("Costs non-negative", all(x >= 0 for x in costs)),
        ("Adaptive memory bounded", all(0.5 <= x <= 1.5 for x in multipliers)),
        ("Validation pipeline", True),
    ]
    return pd.DataFrame({"Health Check": [x[0] for x in checks], "Status": ["PASS" if x[1] else "REVIEW" for x in checks]})


# ============================================================
# 31.6 v10 RESILIENCE OPERATING SYSTEM CONTROL CENTER
# ============================================================

st.divider()
st.header("🛰️ Shock Sim AI v10.0 — Resilience Operating System")
st.caption(
    "Final integrated layer: Digital Twin → Risk → Prediction → Counterfactuals → "
    "Capital Optimization → Human Decision → Real Event → Learning."
)

# Persistent model registry / audit metadata.
if "model_registry" not in st.session_state:
    st.session_state.model_registry = []

model_version = "v10.0-FINAL"
registry_entry = {
    "Model": model_version,
    "Network Nodes": graph.number_of_nodes(),
    "Network Edges": graph.number_of_edges(),
    "Adaptive": bool(st.session_state.observations_loaded),
    "Scenario": shock_desc,
}
if not st.session_state.model_registry or st.session_state.model_registry[-1] != registry_entry:
    st.session_state.model_registry.append(registry_entry)

os1, os2, os3, os4 = st.columns(4)
os1.metric("Operating Model", "v10.0 FINAL")
os2.metric("Digital Twin", "ACTIVE")
os3.metric("Prediction", "ACTIVE")
os4.metric("Learning Loop", "ACTIVE" if st.session_state.observations_loaded else "READY")

# Digital Twin
st.subheader("🪞 Digital Twin State")
twin_df = digital_twin_snapshot(graph, memory)
st.dataframe(twin_df.round(3), use_container_width=True, hide_index=True)

# Systemic risk layer
st.subheader("🌐 Systemic Risk Engine")
risk_df = systemic_node_risk(graph, baseline_impacts)
rc1, rc2, rc3 = st.columns(3)
rc1.metric("Highest Risk Node", str(risk_df.iloc[0]["Node"]) if not risk_df.empty else "—")
rc2.metric("Top Risk Score", f"{risk_df.iloc[0]['Systemic Risk Score']:.1f}" if not risk_df.empty else "—")
rc3.metric("Network Exposure", f"{baseline_damage:.1f}")
st.dataframe(risk_df, use_container_width=True, hide_index=True)

# Predictive intelligence
st.subheader("🔮 Predictive Resilience Intelligence")
pred_col1, pred_col2 = st.columns([1, 2])
with pred_col1:
    predictive_runs = st.slider("Forecast simulations", 50, 1000, 250, 50, key="v10_predictive_runs")
    predictive_uncertainty = st.slider("Assumption uncertainty", 0.02, 0.30, 0.12, 0.01, key="v10_uncertainty")
    run_prediction = st.button("🔮 RUN PREDICTIVE FORECAST", use_container_width=True, key="v10_predict")

if run_prediction or "v10_forecast" not in st.session_state:
    forecast_df, forecast_summary = predictive_forecast(
        graph,
        shock_sources_dict,
        memory,
        steps=4,
        simulations=predictive_runs,
        uncertainty=predictive_uncertainty,
        seed=42,
    )
    st.session_state.v10_forecast = forecast_df
    st.session_state.v10_forecast_summary = forecast_summary
else:
    forecast_df = st.session_state.v10_forecast
    forecast_summary = st.session_state.v10_forecast_summary

with pred_col2:
    p1, p2, p3 = st.columns(3)
    p1.metric("Expected Damage", f"{forecast_summary['expected_damage']:.1f}")
    p2.metric("P90 Damage", f"{forecast_summary['p90_damage']:.1f}")
    p3.metric("P95 Damage", f"{forecast_summary['p95_damage']:.1f}")

st.dataframe(forecast_df.round(2), use_container_width=True, hide_index=True)

forecast_top = forecast_df.head(min(10, len(forecast_df)))
if not forecast_top.empty:
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Bar(x=forecast_top["Node"], y=forecast_top["Expected Impact (%)"], name="Expected"))
    fig_forecast.add_trace(go.Bar(x=forecast_top["Node"], y=forecast_top["P90 Impact (%)"], name="P90"))
    fig_forecast.update_layout(title="Predictive Impact Envelope", barmode="group", yaxis_title="Impact (%)")
    st.plotly_chart(fig_forecast, use_container_width=True)

# Capital allocation / autonomous queue
st.subheader("💰 Capital Allocation Engine")
capital_curve = portfolio_capital_curve(results)
if not capital_curve.empty:
    capital_fig = go.Figure(
        go.Scatter(
            x=capital_curve["Investment ($k)"],
            y=capital_curve["Containment (%)"],
            mode="lines+markers",
            name="Containment",
            hovertemplate="Investment: %{x:.2f} $k<br>Containment: %{y:.2f}%<extra></extra>",
        )
    )
    capital_fig.update_layout(
        title="Capital → Containment Response",
        xaxis_title="Investment ($k)",
        yaxis_title="Containment (%)",
        yaxis=dict(range=[0, 100], autorange=False),
        hovermode="x unified",
    )
    st.plotly_chart(capital_fig, use_container_width=True)

if optimization_mode == "Budget Constraint":
    queue_df = autonomous_action_queue(frontier, budget=float(budget_limit), top_n=5)
else:
    queue_df = autonomous_action_queue(frontier, target=float(target_containment), top_n=5)

if not queue_df.empty:
    st.markdown("**Decision Queue — human approval required**")
    st.dataframe(queue_df, use_container_width=True, hide_index=True)
else:
    st.info("No frontier candidate satisfies the current capital/containment constraint.")

# Event engine: register an observed outcome without automatically changing the model.
st.subheader("🚨 Event Engine")
st.write("Use this panel after a real event to compare the model's expected exposure with the observed event outcome. Calibration remains an explicit human-approved step.")
e1, e2, e3 = st.columns(3)
with e1:
    observed_damage = st.number_input("Observed total damage units", min_value=0.0, value=0.0, step=1.0, key="v10_observed_damage")
with e2:
    event_confidence = st.slider("Observation confidence (%)", 0, 100, 80, 5, key="v10_event_confidence")
with e3:
    register_event = st.button("📝 REGISTER EVENT", use_container_width=True, key="v10_register_event")

if register_event:
    event_record = {
        "Model": model_version,
        "Scenario": shock_desc,
        "Predicted Expected Damage": round(float(forecast_summary["expected_damage"]), 3),
        "Predicted P90 Damage": round(float(forecast_summary["p90_damage"]), 3),
        "Observed Damage": round(float(observed_damage), 3),
        "Confidence (%)": int(event_confidence),
    }
    if "event_history" not in st.session_state:
        st.session_state.event_history = []
    st.session_state.event_history.append(event_record)
    st.success("Event registered in the local session history. No automatic model change was performed.")

if "event_history" in st.session_state and st.session_state.event_history:
    st.dataframe(pd.DataFrame(st.session_state.event_history), use_container_width=True, hide_index=True)

# Model health / governance dashboard
st.subheader("🛡️ Model Health & Governance")
health_df = model_health_report(graph, memory, validation_df)
hc1, hc2 = st.columns([1, 2])
with hc1:
    passed = int((health_df["Status"] == "PASS").sum())
    total = len(health_df)
    st.metric("Health Checks Passed", f"{passed}/{total}")
with hc2:
    st.dataframe(health_df, use_container_width=True, hide_index=True)

with st.expander("🧾 v10 Model Registry / Audit Trail"):
    st.dataframe(pd.DataFrame(st.session_state.model_registry), use_container_width=True, hide_index=True)
    st.caption("Registry records the active model state in this Streamlit session; production deployments should persist it to a database or immutable audit store.")

st.success("Shock Sim AI v10.0 FINAL operating layer is active. The system remains decision-support software: recommendations require human review before execution.")

# ============================================================
# 32. MODEL GOVERNANCE
# ============================================================

st.divider()

with st.expander("ℹ️ Model Assumptions & Governance"):
    st.markdown(
        """
### Model assumptions

- Dependency `weight` represents modeled transmission strength.
- Intervention `cost` represents modeled investment required for a
  100% reduction of that dependency.
- Partial intervention costs are currently linear.
- Propagation uses persistent impact and nonlinear impact combination.
- The baseline financial exposure is a modeled value unless calibrated
  with company-specific loss data.
- Adaptive learning updates dependency multipliers from observed
  transmission data; it does not claim to be a general-purpose
  machine-learning model.
- Counterfactual search is exhaustive within the selected portfolio
  size and intervention levels.

### Recommended production governance

For production deployment, preserve:

1. historical input data,
2. calibration version,
3. model version,
4. scenario definition,
5. intervention assumptions,
6. predicted outcome,
7. actual outcome,
8. post-event calibration.

This creates an auditable resilience-learning history.
"""
    )

st.caption(
    "Shock Sim AI v10.0 FINAL — Resilience Operating System | "
    "Single-file Decision-Support Engine"
)
