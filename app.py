from itertools import combinations, product
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# 1. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Shock Sim AI", page_icon="🧠", layout="wide"
)


st.title("🧠 Shock Sim AI")

st.subheader(
    "Counterfactual Containment & Reverse Resilience Engine"
)

st.caption(
    "Shock → Propagation → Intervention → Optimization → Resilience"
)


# =========================================================
# 2. NETWORK BUILDER
# =========================================================


def build_network():
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
    graph.add_edge(source, target, weight=weight, cost=cost)

  return graph


# =========================================================
# 3. PROPAGATION ENGINE
# =========================================================


def combine_impacts(existing, contribution):
  existing = np.clip(existing, 0, 100)
  contribution = np.clip(contribution, 0, 100)

  result = 100 * (1 - (1 - existing / 100) * (1 - contribution / 100))

  return float(np.clip(result, 0, 100))


def get_timeline_impacts(graph, source, magnitude, actions=None, steps=4):
  if actions is None:
    actions = []

  reductions = {}

  for action in actions:
    key = (action["source"], action["target"])
    reductions[key] = action["reduction"]

  current = {node: 0.0 for node in graph.nodes}

  current[source] = float(magnitude)

  timeline = {"T0": current.copy()}

  for step in range(1, steps + 1):
    next_impacts = current.copy()

    for u, v, data in graph.edges(data=True):
      base_weight = data["weight"]
      reduction = reductions.get((u, v), 0)
      effective_weight = base_weight * (1 - reduction / 100)
      contribution = current[u] * effective_weight
      next_impacts[v] = combine_impacts(next_impacts[v], contribution)

    current = next_impacts
    timeline[f"T{step}"] = current.copy()

  return timeline


def final_impacts(timeline):
  last_key = list(timeline.keys())[-1]
  return timeline[last_key]


def calculate_total_damage(impacts):
  return float(sum(impacts.values()))


# =========================================================
# 4. INTERVENTIONS & PORTFOLIOS
# =========================================================


def intervention_cost(base_cost, reduction):
  return base_cost * (reduction / 100.0)


def generate_portfolios(graph, intervention_levels, max_interventions):
  edges = list(graph.edges(data=True))
  portfolios = []

  for count in range(1, max_interventions + 1):
    for selected_edges in combinations(edges, count):
      for reductions in product(intervention_levels, repeat=count):
        actions = []
        total_cost = 0.0

        for edge, reduction in zip(selected_edges, reductions):
          source, target, data = edge
          cost = intervention_cost(data["cost"], reduction)
          total_cost += cost
          actions.append({
              "source": source,
              "target": target,
              "reduction": reduction,
              "cost": cost,
          })

        portfolios.append({"actions": actions, "cost": total_cost})

  return portfolios


# =========================================================
# 5. METRICS & PARETO
# =========================================================


def calculate_containment(baseline_damage, final_damage):
  if baseline_damage <= 0:
    return 0.0
  containment = ((baseline_damage - final_damage) / baseline_damage) * 100
  return max(0.0, min(100.0, containment))


def calculate_efficiency(containment, cost):
  if cost <= 0:
    return 0.0
  return containment / cost


def evaluate_portfolio(
    graph, source, magnitude, portfolio, baseline_damage
):
  timeline = get_timeline_impacts(
      graph=graph,
      source=source,
      magnitude=magnitude,
      actions=portfolio["actions"],
      steps=4,
  )

  impacts = final_impacts(timeline)
  damage = calculate_total_damage(impacts)
  containment = calculate_containment(baseline_damage, damage)
  avoided_damage = baseline_damage - damage
  efficiency = calculate_efficiency(containment, portfolio["cost"])

  return {
      "actions": portfolio["actions"],
      "cost": portfolio["cost"],
      "timeline": timeline,
      "final_impacts": impacts,
      "final_damage": damage,
      "damage_avoided": avoided_damage,
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
  return sorted(frontier, key=lambda x: x["cost"])


# =========================================================
# 6. OPTIMIZATION
# =========================================================


def select_budget_strategy(
    results, budget, objective="Maximum Containment", containment_priority=70
):
  feasible = [r for r in results if r["cost"] <= budget]
  if not feasible:
    return None, []

  if objective == "Maximum Containment":
    selected = max(feasible, key=lambda x: x["containment"])
  elif objective == "Maximum Efficiency":
    selected = max(feasible, key=lambda x: x["efficiency"])
  elif objective == "Minimum Cost":
    selected = min(feasible, key=lambda x: x["cost"])
  else:
    max_containment = max(r["containment"] for r in feasible)
    min_efficiency = min(r["efficiency"] for r in feasible)
    max_efficiency = max(r["efficiency"] for r in feasible)

    def score(result):
      containment_score = (
          result["containment"] / max_containment if max_containment > 0 else 0
      )
      if max_efficiency > min_efficiency:
        efficiency_score = (result["efficiency"] - min_efficiency) / (
            max_efficiency - min_efficiency
        )
      else:
        efficiency_score = 1
      p = containment_priority / 100
      return p * containment_score + (1 - p) * efficiency_score

    selected = max(feasible, key=score)

  return selected, feasible


def reverse_resilience(results, target_containment):
  feasible = [r for r in results if r["containment"] >= target_containment]
  if not feasible:
    return None
  return min(feasible, key=lambda x: x["cost"])


def build_reverse_curve(results, targets=None):
  if targets is None:
    targets = [25, 40, 50, 60, 70, 75, 80, 90]
  rows = []
  for target in targets:
    solution = reverse_resilience(results, target)
    if solution:
      rows.append(
          {"Target Containment (%)": target, "Minimum Cost ($k)": solution["cost"]}
      )
  return pd.DataFrame(rows)


def strategy_table(results, budget):
  feasible = [r for r in results if r["cost"] <= budget]
  if not feasible:
    return pd.DataFrame()

  max_containment = max(r["containment"] for r in feasible)
  max_efficiency = max(r["efficiency"] for r in feasible)
  min_cost = min(r["cost"] for r in feasible)

  rows = [
      {
          "Strategy": "Maximum Containment",
          "Cost ($k)": max(feasible, key=lambda x: x["containment"])["cost"],
          "Containment (%)": max_containment,
      },
      {
          "Strategy": "Maximum Efficiency",
          "Cost ($k)": max(feasible, key=lambda x: x["efficiency"])["cost"],
          "Containment (%)": max(feasible, key=lambda x: x["efficiency"])[
              "containment"
          ],
      },
      {
          "Strategy": "Minimum Cost",
          "Cost ($k)": min_cost,
          "Containment (%)": min(feasible, key=lambda x: x["cost"])[
              "containment"
          ],
      },
  ]
  return pd.DataFrame(rows)


# =========================================================
# 7. VISUALIZATION
# =========================================================


def network_impact_figure(graph, impacts, title):
  positions = nx.spring_layout(graph, seed=42)
  edge_x, edge_y = [], []
  for u, v in graph.edges():
    x0, y0 = positions[u]
    x1, y1 = positions[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

  edge_trace = go.Scatter(
      x=edge_x, y=edge_y, mode="lines", line=dict(width=1), hoverinfo="none"
  )

  node_x, node_y, labels, values = [], [], [], []
  for node in graph.nodes():
    x, y = positions[node]
    node_x.append(x)
    node_y.append(y)
    labels.append(node)
    values.append(round(impacts.get(node, 0), 2))

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
          colorbar=dict(title="Impact"),
      ),
      hovertemplate=(
          "<b>%{text}</b><br>Impact: %{marker.color:.2f}<extra></extra>"
      ),
  )

  fig = go.Figure(data=[edge_trace, node_trace])
  fig.update_layout(
      title=title,
      showlegend=False,
      xaxis=dict(visible=False),
      yaxis=dict(visible=False),
      margin=dict(l=10, r=10, t=50, b=10),
  )
  return fig


def pareto_figure(frontier):
  fig = go.Figure()
  fig.add_trace(
      go.Scatter(
          x=[r["cost"] for r in frontier],
          y=[r["containment"] for r in frontier],
          mode="lines+markers",
          text=[f"{r['containment']:.1f}%" for r in frontier],
          hovertemplate=(
              "Cost: %{x:.1f} $k<br>Containment: %{y:.1f}%<extra></extra>"
          ),
      )
  )
  fig.update_layout(
      title="Pareto Frontier",
      xaxis_title="Intervention Cost ($k)",
      yaxis_title="Containment (%)",
  )
  return fig


def reverse_cost_figure(df):
  fig = go.Figure()
  fig.add_trace(
      go.Scatter(
          x=df["Target Containment (%)"],
          y=df["Minimum Cost ($k)"],
          mode="lines+markers",
      )
  )
  fig.update_layout(
      title="Reverse Resilience Cost Curve",
      xaxis_title="Target Containment (%)",
      yaxis_title="Minimum Cost ($k)",
  )
  return fig


# =========================================================
# 8. SIDEBAR INTERFACE
# =========================================================

st.sidebar.header("⚙️ Scenario")

shock_source = st.sidebar.selectbox(
    "Shock Source", ["Energy", "Suppliers", "Raw_Materials"]
)
shock_magnitude = st.sidebar.slider(
    "Shock Magnitude", min_value=30, max_value=100, value=75
)
optimization_mode = st.sidebar.radio(
    "Optimization Mode", ["Budget Frontier", "Reverse Resilience"]
)

st.sidebar.divider()

if optimization_mode == "Budget Frontier":
  budget = st.sidebar.slider(
      "Budget Limit ($k)", min_value=20, max_value=300, value=100
  )
  objective = st.sidebar.selectbox(
      "Objective",
      ["Maximum Containment", "Maximum Efficiency", "Minimum Cost", "Balanced"],
  )
  containment_priority = st.sidebar.slider("Containment Priority", 0, 100, 70)
else:
  target_containment = st.sidebar.slider("Target Containment (%)", 10, 95, 75)

intervention_levels = st.sidebar.multiselect(
    "Intervention Levels (%)", [25, 50, 75, 100], default=[25, 50, 75, 100]
)
max_interventions = st.sidebar.slider("Maximum Interventions", 1, 3, 3)

run = st.sidebar.button("🚀 RUN SIMULATION", use_container_width=True)

# =========================================================
# 9. EXECUTION & LOGIC
# =========================================================

if not run:
  st.info("حدد السيناريو من القائمة الجانبية ثم اضغط RUN SIMULATION.")
  st.markdown("""
        ### فكرة المحرك
        هذا النظام لا يسأل فقط: **"ماذا يحدث إذا حدثت صدمة؟"**
        بل يسأل: **"ما أقل تدخل يمكنه احتواء الصدمة إلى المستوى المطلوب؟"**
        """)
  st.stop()

if not intervention_levels:
  st.error("يجب اختيار مستوى تدخل واحد على الأقل.")
  st.stop()

graph = build_network()

baseline_timeline = get_timeline_impacts(
    graph=graph,
    source=shock_source,
    magnitude=shock_magnitude,
    actions=[],
    steps=4,
)
baseline_impacts = final_impacts(baseline_timeline)
baseline_damage = calculate_total_damage(baseline_impacts)

downstream = {
    node: impact
    for node, impact in baseline_impacts.items()
    if node != shock_source
}
critical_node = max(downstream, key=downstream.get)

with st.spinner("Generating counterfactual portfolios..."):
  portfolios = generate_portfolios(
      graph=graph,
      intervention_levels=intervention_levels,
      max_interventions=max_interventions,
  )

with st.spinner("Evaluating portfolios..."):
  results = []
  for portfolio in portfolios:
    result = evaluate_portfolio(
        graph=graph,
        source=shock_source,
        magnitude=shock_magnitude,
        portfolio=portfolio,
        baseline_damage=baseline_damage,
    )
    results.append(result)

if not results:
  st.error("لم يتم إنشاء أي تدخلات.")
  st.stop()

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Shock", f"{shock_magnitude}%")
col2.metric("Baseline Damage", f"{baseline_damage:.1f}")
col3.metric("Critical Downstream", critical_node)
col4.metric("Portfolios", len(results))

st.divider()

# Optimization Selection
if optimization_mode == "Budget Frontier":
  selected, feasible = select_budget_strategy(
      results=results,
      budget=budget,
      objective=objective,
      containment_priority=containment_priority,
  )
  if selected is None:
    st.error("لا توجد استراتيجية ضمن الميزانية المحددة.")
    st.stop()
else:
  selected = reverse_resilience(results, target_containment)
  if selected is None:
    st.error(f"لا توجد محفظة تحقق احتواء {target_containment}%.")
    st.stop()

# Selected Result View
st.header("🎯 Selected Counterfactual")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Intervention Cost", f"${selected['cost']:.1f}k")
m2.metric("Containment", f"{selected['containment']:.1f}%")
m3.metric("Damage Avoided", f"{selected['damage_avoided']:.1f}")
m4.metric("Efficiency", f"{selected['efficiency']:.3f}")

# Actions
st.subheader("🛠 Selected Interventions")
actions_df = pd.DataFrame(selected["actions"])
if not actions_df.empty:
  actions_df["reduction"] = actions_df["reduction"].astype(str) + "%"
  actions_df["cost"] = actions_df["cost"].round(2)
  st.dataframe(actions_df, use_container_width=True, hide_index=True)

if optimization_mode == "Budget Frontier":
  st.subheader("📊 Strategy Comparison")
  st.dataframe(strategy_table(results, budget), use_container_width=True, hide_index=True)

st.subheader("📈 Pareto Frontier")
frontier = pareto_frontier(results)
if frontier:
  st.plotly_chart(pareto_figure(frontier), use_container_width=True)

if optimization_mode == "Reverse Resilience":
  st.subheader("🔄 Reverse Resilience")
  st.write(f"Target containment: **{target_containment}%**")
  reverse_df = build_reverse_curve(results)
  if not reverse_df.empty:
    st.plotly_chart(reverse_cost_figure(reverse_df), use_container_width=True)
    st.dataframe(reverse_df, use_container_width=True, hide_index=True)

st.subheader("🗺️ Network Impact Map")
c1, c2 = st.columns(2)
with c1:
  st.plotly_chart(
      network_impact_figure(graph, baseline_impacts, "Baseline — No Intervention"),
      use_container_width=True,
  )
with c2:
  st.plotly_chart(
      network_impact_figure(
          graph, selected["final_impacts"], "Counterfactual — Selected"
      ),
      use_container_width=True,
  )

st.subheader("📉 Before vs After")
comparison = pd.DataFrame(
    {"Baseline": baseline_impacts, "Counterfactual": selected["final_impacts"]}
)
st.bar_chart(comparison)

st.subheader("⏱️ Temporal Propagation Replay")
selected_time = st.select_slider(
    "Select Time", options=list(baseline_timeline.keys())
)
replay_df = pd.DataFrame({
    "Baseline": baseline_timeline[selected_time],
    "Counterfactual": selected["timeline"][selected_time],
})
st.dataframe(replay_df.round(2), use_container_width=True)

st.divider()
st.caption(
    "Shock Sim AI — Counterfactual Containment & Reverse Resilience Engine"
  )
  
