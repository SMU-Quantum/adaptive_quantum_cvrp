```bash
uv venv --python-python3.10
```





## 🧮 CVRP Mathematical Formulation for Augmented Lagrangian Method (ALM)

This section outlines the precise formulation of the Capacitated Vehicle Routing Problem (CVRP) used in our ALM-based optimization pipeline. This formulation directly guides the implementation of constraints, objectives, and penalty handling in the `augmented_lagrangian.py` module.

---

### 🔢 Decision Variables

Let:

- $ x_{ijk} \in \{0, 1\} $: Binary variable, 1 if vehicle $ k $ travels directly from customer $ i $ to customer $ j $, 0 otherwise.
- $ y_{ik} \in \{0, 1\} $: Binary variable, 1 if customer $ i $ is visited by vehicle $ k $, 0 otherwise.
- $ q_{ik} \in \mathbb{R}_{\geq 0} $: Flow variable representing the load carried by vehicle $ k $ after visiting node $ i $ (used in flow-based formulations).

---

### 🎯 Objective Function

Minimize the total travel cost across all vehicles:

$$
\min \sum_{k} \sum_{i} \sum_{j} c_{ij} \cdot x_{ijk}
$$

where $ c_{ij} $ is the cost (typically the Euclidean distance) between node $ i $ and node $ j $.

---

### ✅ Constraints

#### 1. Customer Visit Constraints

Each customer must be visited exactly once by exactly one vehicle:

$$
\sum_{k} \sum_{j \neq i} x_{ijk} = 1 \quad \forall i \in \text{Customers}
$$

#### 2. Vehicle Capacity Constraints

The total demand served by each vehicle must not exceed its capacity $ Q $:

$$
\sum_{i} d_i \cdot y_{ik} \leq Q \quad \forall k \in \text{Vehicles}
$$

where $ d_i $ is the demand of customer $ i $.

#### 3. Flow Conservation / Route Continuity

For all non-depot nodes, ensure vehicle route continuity:

$$
\sum_{j} x_{ijk} - \sum_{j} x_{jik} = 0 \quad \forall i \in \text{Customers}, \forall k
$$

#### 4. Depot Start and End Constraints

Each vehicle must start and end at the depot:

$$
\sum_{j \neq 0} x_{0jk} = 1 \quad \text{(start from depot)}
$$
$$
\sum_{i \neq 0} x_{i0k} = 1 \quad \text{(return to depot)}
$$

---

### 📝 Notes

- This formulation assumes a single depot indexed as node 0.
- Subtour elimination is handled using flow-based or additional combinatorial constraints in the implementation.
- The penalty formulation for the ALM framework is layered on top of these constraints.

---

### 📁 Location

This formulation is used in:
- `src/augmented_lagrangian.py`
- `src/qubo_encoder.py` (when translating constraints into QUBO terms)

---

For full implementation details and examples, see the [`examples/`](examples/) and [`docs/`](docs/) folders.
