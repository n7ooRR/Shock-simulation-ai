from itertools import combinations, product
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# SHOCK SIM AI — COMMERCIAL MVP v2.2
# Single-file Streamlit application
# =========================================================

st.set_page_config(
    page_title="Shock Sim AI — Commercial MVP",
    page_icon="🧠",
    layout="wide",
)


# =========================================================
# 1. PAGE HEADER
# =========================================================

st.title("🧠 Shock Sim AI — Commercial MVP")
st.subheader("Systemic Shock Containment & Investment Decision Engine")
st.caption(
    "Core IP: determine the minimum intervention investment required "
    "to achieve a target resilience/containment level."
)


# =========================================================
# 2. DEFAULT NETWORK
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
    required_cols = {"source", "target", "weight", "cost"}

    try:
        df = pd.read_csv(uploaded_file)

        if not required_cols.issubset(df.columns):
            st.error(
                "CSV must contain: source, target, weight, cost"
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

            if weight < 0:
                raise ValueError(
                    f"Negative weight found for {source} -> {target}"
                )

            if cost < 0:
                raise ValueError(
                    f"Negative cost found for {source} -> {target}"
                )

            graph.add_edge(
                source,
                target,
                weight=weight,
                cost=cost,
            )

        if graph.number_of_edges() == 0:
            st.warning(
                "Uploaded CSV produced an empty network. "
                "Using the default network."
            )
            return build_default_network()

        return graph

    except Exception as exc:
        st.error(f"CSV loading error: {exc}")
        return build_default_network()


# =========================================================
# 3. PROPAGATION ENGINE
# =========================================================

def combine_impacts(existing, contribution):
    """
    Nonlinear capped combination of multiple incoming impacts.
    Values remain in the 0-100 range.
    """
    existing = np.clip(existing, 0, 100)
    contribution = np.clip(contribution, 0, 100)

    result = 100 * (
        1
        - (1 - existing / 100)
        * (1 - contribution / 100)
    )

    return float(np.clip(result, 0, 100))


def get_timeline_impacts(
    graph,
    shock_sources_dict,
    actions=None,
    steps=4,
):
    """
    Persistent-impact temporal propagation.

    T0 = initial shock.
    Each subsequent step retains current impact and propagates
    part of it through network edges.
    """
    if actions is None:
        actions = []

    reductions = {}

    for action in actions:
        key = (
            action["source"],
            action["target"],
        )
        reductions[key] = action["reduction"]

    current = {
        node: 0.0
        for node in graph.nodes
    }

    for source, magnitude in shock_sources_dict.items():
        if source in current:
            current[source] = float(
                np.clip(magnitude, 0, 100)
            )

    timeline = {
        "T0": current.copy()
    }

    for step in range(1, steps + 1):
        next_impacts = current.copy()

        for u, v, data in graph.edges(data=True):
            base_weight = float(data.get("weight", 0))
            reduction = reductions.get(
                (u, v),
                0,
            )

            effective_weight = (
                base_weight
                * (1 - reduction / 100)
            )

            contribution = (
                current[u]
                * effective_weight
            )

            next_impacts[v] = combine_impacts(
                next_impacts[v],
                contribution,
            )

        current = {
            node: float(
                np.clip(value, 0, 100)
            )
            for node, value in next_impacts.items()
        }

        timeline[f"T{step}"] = current.copy()

    return timeline


def final_impacts(timeline):
    if not timeline:
        return {}

    last_key = list(timeline.keys())[-1]
    return timeline[last_key]


def calculate_total_damage(impacts):
    return float(sum(impacts.values()))


# =========================================================
# 4. OPTIMIZATION ENGINE
# =========================================================

def intervention_cost(base_cost, reduction):
    return float(
        base_cost
        * (reduction / 100.0)
    )


def generate_portfolios(
    graph,
    intervention_levels,
    max_interventions,
):
    """
    Exhaustive portfolio generator.

    For each selected edge, the engine tries every allowed
    intervention level.
    """
    edges = list(
        graph.edges(data=True)
    )

    if not edges:
        return []

    max_interventions = min(
        max_interventions,
        len(edges),
    )

    portfolios = []

    for count in range(
        1,
        max_interventions + 1,
    ):
        for selected_edges in combinations(
            edges,
            count,
        ):
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


def calculate_containment(
    baseline_damage,
    final_damage,
):
    if baseline_damage <= 0:
        return 0.0

    containment = (
        (baseline_damage - final_damage)
        / baseline_damage
    ) * 100

    return float(
        np.clip(
            containment,
            0,
            100,
        )
    )


def calculate_efficiency(
    containment,
    cost,
):
    if cost <= 0:
        return 0.0

    return containment / cost


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
        steps=4,
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
        "cost": portfolio["cost"],
        "timeline": timeline,
        "final_impacts": impacts,
        "final_damage": damage,
        "damage_avoided": damage_avoided,
        "containment": containment,
        "efficiency": efficiency,
    }


def pareto_frontier(results):
    frontier = []

    for candidate in results:
        dominated = False

        for other in results:
            if candidate is other:
                continue

            better_or_equal = (
                other["cost"]
                <= candidate["cost"]
                and other["containment"]
                >= candidate["containment"]
            )

            strictly_better = (
                other["cost"]
                < candidate["cost"]
                or other["containment"]
                > candidate["containment"]
            )

            if (
                better_or_equal
                and strictly_better
            ):
                dominated = True
                break

        if not dominated:
            frontier.append(candidate)

    return sorted(
        frontier,
        key=lambda x: (
            x["cost"],
            -x["containment"],
        ),
    )


def select_budget_strategy(
    results,
    budget,
):
    feasible = [
        result
        for result in results
        if result["cost"] <= budget
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


def reverse_resilience(
    results,
    target_containment,
):
    feasible = [
        result
        for result in results
        if result["containment"]
        >= target_containment
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


def build_reverse_resilience_curve(
    results,
    targets=None,
):
    if targets is None:
        targets = [
            25,
            40,
            50,
            60,
            70,
            75,
            80,
            90,
        ]

    rows = []

    for target in targets:
        solution = reverse_resilience(
            results,
            target,
        )

        if solution is None:
            rows.append(
                {
                    "Target Containment (%)": target,
                    "Minimum Investment ($k)": np.nan,
                    "Achieved Containment (%)": np.nan,
                    "Avoided Damage Units": np.nan,
                }
            )
        else:
            rows.append(
                {
                    "Target Containment (%)": target,
                    "Minimum Investment ($k)": round(
                        solution["cost"],
                        2,
                    ),
                    "Achieved Containment (%)": round(
                        solution["containment"],
                        2,
                    ),
                    "Avoided Damage Units": round(
                        solution["damage_avoided"],
                        2,
                    ),
                }
            )

    return pd.DataFrame(rows)


# =========================================================
# 5. FINANCIAL MODEL
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
    rows = []

    for result in results:
        avoided_value = (
            calculate_avoided_damage_value(
                baseline_damage=baseline_damage,
                final_damage=result[
                    "final_damage"
                ],
                financial_value_per_damage_unit=conversion_factor,
            )
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

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Collapse duplicate investment levels by retaining
    # the highest containment at each cost.
    df = (
        df.sort_values(
            [
                "Investment ($k)",
                "Containment (%)",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop_duplicates(
            subset=["Investment ($k)"],
            keep="first",
        )
        .sort_values("Investment ($k)")
    )

    return df


# =========================================================
# 6. VISUALIZATION ENGINE
# =========================================================

def network_impact_figure(
    graph,
    impacts,
    title,
):
    if graph.number_of_nodes() == 0:
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
            size=35,
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
            "<b>%{text}</b>"
            "<br>Impact: %{marker.color:.2f}"
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


def pareto_figure(frontier):
    fig = go.Figure()

    if not frontier:
        return fig

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
            text=[
                f"{r['containment']:.1f}%"
                for r in frontier
            ],
            hovertemplate=(
                "Investment: $%{x:.1f}k"
                "<br>Containment: %{y:.1f}%"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Pareto Frontier — Investment vs Containment",
        xaxis_title="Intervention Cost ($k)",
        yaxis_title="Containment (%)",
    )

    return fig


def reverse_resilience_figure(reverse_df):
    valid = reverse_df.dropna(
        subset=["Minimum Investment ($k)"]
    )

    fig = go.Figure()

    if valid.empty:
        return fig

    fig.add_trace(
        go.Scatter(
            x=valid[
                "Target Containment (%)"
            ],
            y=valid[
                "Minimum Investment ($k)"
            ],
            mode="lines+markers",
            name="Required Investment",
            hovertemplate=(
                "Target Containment: %{x:.0f}%"
                "<br>Minimum Investment: $%{y:.1f}k"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Reverse Resilience Curve",
        xaxis_title="Target Containment (%)",
        yaxis_title="Minimum Required Investment ($k)",
    )

    return fig


def financial_figure(financial_df):
    fig = go.Figure()

    if financial_df.empty:
        return fig

    fig.add_trace(
        go.Scatter(
            x=financial_df[
                "Investment ($k)"
            ],
            y=financial_df[
                "Avoided Loss ($k)"
            ],
            mode="lines+markers",
            name="Avoided Loss ($k)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=financial_df[
                "Investment ($k)"
            ],
            y=financial_df[
                "Containment (%)"
            ],
            mode="lines+markers",
            name="Containment (%)",
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Investment vs Avoided Loss & Containment",
        xaxis_title="Investment ($k)",
        yaxis=dict(
            title="Avoided Loss ($k)"
        ),
        yaxis2=dict(
            title="Containment (%)",
            overlaying="y",
            side="right",
        ),
        legend=dict(
            orientation="h"
        ),
    )

    return fig


# =========================================================
# 7. EXECUTIVE REPORT
# =========================================================

def generate_executive_report(
    company_name,
    industry,
    revenue,
    shock_desc,
    baseline_damage,
    baseline_financial_exposure,
    critical_node,
    selected,
    avoided_loss_ratio,
    avoided_loss_value,
    optimization_mode,
):
    report = f"""
============================================================
SHOCK SIM AI — EXECUTIVE COMMAND REPORT
Commercial MVP v2.2
============================================================

COMPANY PROFILE
------------------------------------------------------------
Company Name: {company_name}
Industry: {industry}
Annual Revenue: ${revenue:.2f}M

SCENARIO
------------------------------------------------------------
Scenario: {shock_desc}
Optimization Mode: {optimization_mode}

BASELINE SYSTEMIC EXPOSURE
------------------------------------------------------------
Baseline Damage Units: {baseline_damage:.2f}
Modeled Baseline Financial Exposure:
${baseline_financial_exposure:.2f}k

Most Affected Downstream Node:
{critical_node}

RECOMMENDED INVESTMENT
------------------------------------------------------------
Required Investment:
${selected["cost"]:.2f}k

Expected Containment:
{selected["containment"]:.2f}%

Final Damage Units:
{selected["final_damage"]:.2f}

Modeled Avoided Loss:
${avoided_loss_value:.2f}k

Avoided Loss / Investment:
{avoided_loss_ratio:.2f}x

RECOMMENDED INTERVENTION PORTFOLIO
------------------------------------------------------------
"""

    for action in selected["actions"]:
        report += (
            f"- {action['source']} -> {action['target']} | "
            f"Reduction: {action['reduction']}% | "
            f"Investment: ${action['cost']:.2f}k\n"
        )

    report += """
MODEL GOVERNANCE NOTE
------------------------------------------------------------
This report is a decision-support output based on the network,
shock magnitudes, propagation rules, intervention levels, and
financial conversion assumptions supplied to the model.

The Avoided Loss / Investment ratio is NOT an accounting ROI.
Financial assumptions should be calibrated and validated against
company-specific operational and financial data before use in
formal investment decisions.

============================================================
Shock Sim AI — Systemic Shock Containment & Investment
Decision Engine
============================================================
"""

    return report


# =========================================================
# 8. SIDEBAR — COMPANY
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
    ],
)

annual_revenue = st.sidebar.number_input(
    "Annual Revenue ($M)",
    min_value=0.0,
    value=250.0,
    step=10.0,
)


# =========================================================
# 9. SIDEBAR — FINANCIAL MODEL
# =========================================================

st.sidebar.subheader("💰 Financial Model")

financial_conversion = st.sidebar.number_input(
    "Damage Unit Value ($k)",
    min_value=0.1,
    value=1.0,
    step=0.1,
    help=(
        "Monetary value assigned to one modeled damage unit. "
        "Calibrate this using company data."
    ),
)


# =========================================================
# 10. SIDEBAR — NETWORK
# =========================================================

st.sidebar.divider()
st.sidebar.header("📁 Network Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload Custom Network CSV",
    type=["csv"],
    help=(
        "Required columns: source, target, weight, cost"
    ),
)

if uploaded_file is not None:
    graph = load_network_from_csv(
        uploaded_file
    )

    st.sidebar.success(
        "Custom company network loaded."
    )
else:
    graph = build_default_network()

    st.sidebar.info(
        "Using standard demo network."
    )


# =========================================================
# 11. SIDEBAR — NETWORK SUMMARY
# =========================================================

st.sidebar.caption(
    f"Nodes: {graph.number_of_nodes()} | "
    f"Edges: {graph.number_of_edges()}"
)


# =========================================================
# 12. SIDEBAR — SCENARIO LIBRARY
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


if "Energy Shock" in scenario_option:
    if "Energy" not in graph.nodes:
        st.sidebar.warning(
            "Energy node not found in current network."
        )
        shock_sources_dict = {}
        shock_desc = (
            "Invalid Energy Shock — node not found"
        )
    else:
        shock_sources_dict = {
            "Energy": 80
        }
        shock_desc = (
            "Energy Sector Severe Spike (+80%)"
        )

elif "Logistics Disruption" in scenario_option:
    if "Logistics" not in graph.nodes:
        st.sidebar.warning(
            "Logistics node not found in current network."
        )
        shock_sources_dict = {}
        shock_desc = (
            "Invalid Logistics Shock — node not found"
        )
    else:
        shock_sources_dict = {
            "Logistics": 75
        }
        shock_desc = (
            "Logistics Channel Disruption (+75%)"
        )

elif "Factory Shutdown" in scenario_option:
    if "Manufacturing" not in graph.nodes:
        st.sidebar.warning(
            "Manufacturing node not found in current network."
        )
        shock_sources_dict = {}
        shock_desc = (
            "Invalid Manufacturing Shock — node not found"
        )
    else:
        shock_sources_dict = {
            "Manufacturing": 90
        }
        shock_desc = (
            "Manufacturing Plant Shutdown / Failure (+90%)"
        )

elif "Multi-Shock" in scenario_option:
    available_shocks = {}

    if "Energy" in graph.nodes:
        available_shocks["Energy"] = 70

    if "Suppliers" in graph.nodes:
        available_shocks["Suppliers"] = 85

    if not available_shocks:
        st.sidebar.warning(
            "No Energy/Suppliers nodes found."
        )
        shock_sources_dict = {}
        shock_desc = (
            "Invalid Multi-Shock — nodes not found"
        )
    else:
        shock_sources_dict = available_shocks
        shock_desc = (
            "Compound Systemic Shock "
            "(Energy + Supplier Failure)"
        )

else:
    st.sidebar.subheader(
        "Custom Shock Parameters"
    )

    if node_list:
        custom_source = st.sidebar.selectbox(
            "Primary Shock Source",
            node_list,
        )
    else:
        custom_source = None

    custom_magnitude = st.sidebar.slider(
        "Shock Magnitude (%)",
        min_value=30,
        max_value=100,
        value=75,
    )

    if custom_source:
        shock_sources_dict = {
            custom_source: custom_magnitude
        }
        shock_desc = (
            f"Custom Shock on "
            f"{custom_source} "
            f"({custom_magnitude}%)"
        )


# =========================================================
# 13. SIDEBAR — OPTIMIZATION
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
        min_value=10,
        max_value=250,
        value=100,
    )

else:
    target_containment_val = st.sidebar.slider(
        "Target Containment (%)",
        min_value=20,
        max_value=95,
        value=80,
    )


intervention_levels = [
    25,
    50,
    75,
    100,
]

max_interventions = st.sidebar.slider(
    "Maximum Interventions",
    min_value=1,
    max_value=3,
    value=3,
)


run_btn = st.sidebar.button(
    "🚀 RUN COMMERCIAL SIMULATION",
    use_container_width=True,
)


# =========================================================
# 14. LANDING STATE
# =========================================================

if not run_btn:
    st.info(
        "Configure the company, network, scenario and "
        "optimization objective, then run the simulation."
    )

    st.markdown(
        """
        ### 🌟 Commercial MVP Capabilities

        - **Systemic Shock Propagation**
        - **Multi-Shock Scenarios**
        - **Counterfactual Intervention Search**
        - **Budget-Constrained Optimization**
        - **Reverse Resilience**
        - **Minimum Investment Calculation**
        - **Avoided Loss Modeling**
        - **Pareto Frontier**
        - **Reverse Resilience Curve**
        - **Executive Command Center**
        - **Custom Network CSV**
        - **Executive TXT Report**
        """
    )

    st.stop()


# =========================================================
# 15. VALIDATION
# =========================================================

if graph.number_of_nodes() == 0:
    st.error(
        "Network contains no nodes."
    )
    st.stop()

if not shock_sources_dict:
    st.error(
        "No valid shock source exists for the selected scenario."
    )
    st.stop()


# =========================================================
# 16. BASELINE
# =========================================================

with st.spinner(
    "Calculating baseline systemic propagation..."
):
    baseline_timeline = get_timeline_impacts(
        graph=graph,
        shock_sources_dict=shock_sources_dict,
        actions=[],
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
    most_affected_node = max(
        downstream,
        key=downstream.get,
    )
else:
    most_affected_node = max(
        baseline_impacts,
        key=baseline_impacts.get,
    )


# =========================================================
# 17. PORTFOLIO SEARCH
# =========================================================

with st.spinner(
    "Searching counterfactual intervention portfolios..."
):
    portfolios = generate_portfolios(
        graph=graph,
        intervention_levels=intervention_levels,
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
        "No valid intervention portfolio was generated."
    )
    st.stop()


# =========================================================
# 18. OPTIMIZATION SELECTION
# =========================================================

if opt_mode == "Budget Constraint":

    selected, feasible = (
        select_budget_strategy(
            results,
            budget_limit,
        )
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

    feasible = [
        result
        for result in results
        if result["containment"]
        >= target_containment_val
    ]

    if selected is None:
        st.error(
            f"No portfolio can achieve "
            f"{target_containment_val}% containment "
            f"within the current intervention search space."
        )
        st.stop()


# =========================================================
# 19. FINANCIAL METRICS
# =========================================================

baseline_financial_exposure = (
    baseline_damage
    * financial_conversion
)

avoided_loss_value = (
    calculate_avoided_damage_value(
        baseline_damage=baseline_damage,
        final_damage=selected[
            "final_damage"
        ],
        financial_value_per_damage_unit=financial_conversion,
    )
)

avoided_loss_ratio = (
    calculate_avoided_damage_ratio(
        avoided_value=avoided_loss_value,
        investment=selected["cost"],
    )
)


# =========================================================
# 20. EXECUTIVE COMMAND CENTER
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
    most_affected_node,
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
    "Avoided Loss / Investment is a modeled "
    "decision-support ratio, not an accounting ROI."
)


# =========================================================
# 21. DO-NOTHING VS INTERVENTION
# =========================================================

st.divider()

col_a, col_b = st.columns(2)

with col_a:
    st.markdown(
        "### 🔴 Without Intervention"
    )

    st.error(
        f"""
        **Damage Units:** {baseline_damage:.1f}

        **Modeled Financial Exposure:**
        ${baseline_financial_exposure:.1f}k

        **Most Affected Node:**
        {most_affected_node}

        **Systemic Status:**
        Shock propagates through connected dependencies.
        """
    )


with col_b:
    st.markdown(
        f"### 🟢 With Recommended Investment"
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
# 22. INVESTMENT DECISION
# =========================================================

st.divider()
st.header("🎯 Investment Decision")

d1, d2, d3, d4 = st.columns(4)

d1.metric(
    "Required Investment",
    f"${selected['cost']:.1f}k",
)

d2.metric(
    "Target / Achieved Containment",
    f"{selected['containment']:.1f}%",
)

d3.metric(
    "Modeled Avoided Loss",
    f"${avoided_loss_value:.1f}k",
)

d4.metric(
    "Avoided Loss / Investment",
    f"{avoided_loss_ratio:.2f}x",
)


if avoided_loss_ratio >= 3:
    decision_text = (
        "Under the current model assumptions, "
        "the modeled avoided loss is approximately "
        "3x or more than the intervention investment."
    )
elif avoided_loss_ratio >= 1:
    decision_text = (
        "Under the current model assumptions, "
        "the modeled avoided loss exceeds the "
        "intervention investment."
    )
else:
    decision_text = (
        "Under the current model assumptions, "
        "the modeled avoided loss does not exceed "
        "the intervention investment."
    )

st.info(decision_text)

st.caption(
    "Financial results depend on the supplied damage-unit "
    "conversion factor and should be calibrated using "
    "company-specific data."
)


# =========================================================
# 23. SELECTED PORTFOLIO
# =========================================================

st.divider()
st.header("🛠 Recommended Intervention Portfolio")

actions_df = pd.DataFrame(
    selected["actions"]
)

if not actions_df.empty:
    display_actions = actions_df.copy()

    display_actions["reduction"] = (
        display_actions["reduction"]
        .astype(str)
        + "%"
    )

    display_actions["cost"] = (
        display_actions["cost"]
        .round(2)
    )

    display_actions = display_actions.rename(
        columns={
            "source": "Source",
            "target": "Target",
            "reduction": "Reduction",
            "cost": "Investment ($k)",
        }
    )

    st.dataframe(
        display_actions,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# 24. REVERSE RESILIENCE CURVE
# =========================================================

st.divider()
st.header("🧬 Reverse Resilience Engine")

st.write(
    "The engine searches for the minimum investment "
    "required to achieve each target containment level."
)

reverse_df = build_reverse_resilience_curve(
    results
)

st.dataframe(
    reverse_df,
    use_container_width=True,
    hide_index=True,
)

st.plotly_chart(
    reverse_resilience_figure(
        reverse_df
    ),
    use_container_width=True,
)


# =========================================================
# 25. TARGET RESILIENCE CALCULATOR
# =========================================================

st.subheader(
    "🎯 Target Resilience Calculator"
)

target_input = st.slider(
    "Required Containment Target (%)",
    min_value=10,
    max_value=95,
    value=80,
    step=5,
)

target_solution = reverse_resilience(
    results,
    target_input,
)

if target_solution is None:

    st.warning(
        f"No available intervention portfolio can "
        f"achieve {target_input}% containment."
    )

else:

    target_avoided_loss = (
        calculate_avoided_damage_value(
            baseline_damage=baseline_damage,
            final_damage=target_solution[
                "final_damage"
            ],
            financial_value_per_damage_unit=financial_conversion,
        )
    )

    target_ratio = (
        calculate_avoided_damage_ratio(
            avoided_value=target_avoided_loss,
            investment=target_solution[
                "cost"
            ],
        )
    )

    t1, t2, t3, t4 = st.columns(4)

    t1.metric(
        "Target",
        f"{target_input}%",
    )

    t2.metric(
        "Minimum Investment",
        f"${target_solution['cost']:.1f}k",
    )

    t3.metric(
        "Achieved",
        f"{target_solution['containment']:.1f}%",
    )

    t4.metric(
        "Avoided Loss / Investment",
        f"{target_ratio:.2f}x",
    )

    st.success(
        f"""
        **Minimum modeled investment:**
        ${target_solution['cost']:.1f}k

        **Achieved containment:**
        {target_solution['containment']:.1f}%
        """
    )


# =========================================================
# 26. FINANCIAL SENSITIVITY
# =========================================================

st.divider()
st.header(
    "🔥 Investment Economics"
)

financial_df = build_financial_sensitivity(
    results=results,
    baseline_damage=baseline_damage,
    conversion_factor=financial_conversion,
)

if not financial_df.empty:

    st.dataframe(
        financial_df,
        use_container_width=True,
        hide_index=True,
    )

    st.plotly_chart(
        financial_figure(
            financial_df
        ),
        use_container_width=True,
    )


# =========================================================
# 27. PARETO FRONTIER
# =========================================================

st.divider()
st.header(
    "📈 Pareto Frontier"
)

frontier = pareto_frontier(
    results
)

st.plotly_chart(
    pareto_figure(frontier),
    use_container_width=True,
)

st.caption(
    "Pareto-efficient portfolios represent cases where "
    "another portfolio cannot simultaneously provide "
    "lower investment and higher containment."
)


# =========================================================
# 28. NETWORK IMPACT MAPS
# =========================================================

st.divider()
st.header(
    "🗺️ Systemic Impact Maps"
)

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


# =========================================================
# 29. TEMPORAL PROPAGATION REPLAY
# =========================================================

st.divider()
st.header(
    "⏱️ Propagation Replay"
)

timeline_rows = []

for time_key, impacts in selected[
    "timeline"
].items():

    timeline_rows.append(
        {
            "Time": time_key,
            **{
                node: round(value, 2)
                for node, value
                in impacts.items()
            },
        }
    )

timeline_df = pd.DataFrame(
    timeline_rows
)

st.dataframe(
    timeline_df,
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# 30. EXECUTIVE REPORT
# =========================================================

st.divider()
st.header(
    "📄 Executive Report Generation"
)

report_text = generate_executive_report(
    company_name=company_name,
    industry=industry,
    revenue=annual_revenue,
    shock_desc=shock_desc,
    baseline_damage=baseline_damage,
    baseline_financial_exposure=baseline_financial_exposure,
    critical_node=most_affected_node,
    selected=selected,
    avoided_loss_ratio=avoided_loss_ratio,
    avoided_loss_value=avoided_loss_value,
    optimization_mode=opt_mode,
)

st.text_area(
    "Executive Decision Report",
    report_text,
    height=430,
)

safe_filename = (
    "".join(
        char
        if char.isalnum()
        else "_"
        for char in company_name
    )
    .strip("_")
    or "company"
)

st.download_button(
    label="📥 Download Executive Report (TXT)",
    data=report_text,
    file_name=(
        f"{safe_filename}_"
        "Containment_Report.txt"
    ),
    mime="text/plain",
    use_container_width=True,
)


# =========================================================
# 31. MODEL GOVERNANCE
# =========================================================

st.divider()

with st.expander(
    "ℹ️ Model Assumptions & Governance"
):
    st.markdown(
        """
        **Propagation model**
        - Edge weights determine shock transmission.
        - Multiple impacts are combined using a capped nonlinear rule.
        - Impacts persist across the simulated time steps.

        **Intervention model**
        - Available reductions are 25%, 50%, 75%, and 100%.
        - Intervention cost is currently linear with reduction percentage.

        **Optimization**
        - The engine exhaustively searches portfolios up to the selected
          maximum intervention count.
        - Reverse Resilience selects the minimum-cost portfolio meeting
          a containment target.

        **Financial model**
        - Damage units are not automatically real currency.
        - The Damage Unit Value must be calibrated against company data.
        - Avoided Loss / Investment is a modeled decision-support ratio,
          not accounting ROI.

        **Network data**
        - CSV inputs are treated as model assumptions unless validated
          against operational/company data.
        """
    )


# =========================================================
# 32. FOOTER
# =========================================================

st.caption(
    "Shock Sim AI — Commercial MVP v2.2 | "
    "Systemic Shock Containment & Investment Decision Engine"
)
