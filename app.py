
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
                    for r
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
                    for r
