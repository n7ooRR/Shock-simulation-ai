
from itertools import combinations, product
from dataclasses import dataclass
import math

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# SHOCK SIM AI — v4.0
# Adaptive Resilience Intelligence
# Single-file Streamlit application
# ============================================================

st.set_page_config(
    page_title="Shock Sim AI — Adaptive Resilience Intelligence v4.0",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Shock Sim AI — Adaptive Resilience Intelligence v4.0")
st.subheader("Adaptive Resilience Intelligence & Counterfactual Investment Engine")
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
        "SHOCK SIM AI — ADAPTIVE RESILIENCE INTELLIGENCE v4.0",
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
### v4.0 Adaptive Intelligence

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
    "Shock Sim AI v4.0 — Adaptive Resilience Intelligence | "
    "Single-file Decision-Support Engine"
)
