# **A Tripartite Investigation into Advanced Solution Methodologies for the Capacitated Vehicle Routing Problem**

## **I. Strategic Overview: Advancing CVRP Solutions**

The Capacitated Vehicle Routing Problem (CVRP) stands as a cornerstone in the field of combinatorial optimization, with profound implications for logistics, transportation, and supply chain management. This section lays the groundwork by defining CVRP, outlining its inherent complexities, and establishing the vision for a tripartite investigation into novel solution methodologies.

### **A. The Capacitated Vehicle Routing Problem: Definition, Formulation, and Complexity**

The Capacitated Vehicle Routing Problem (CVRP) is concerned with determining a set of optimal routes for a fleet of vehicles, all possessing identical capacities, to serve a geographically dispersed set of customers from a central depot. The primary objective is to minimize the total cost, typically measured as the total distance traveled by all vehicles. Key constraints that define the CVRP include: each customer must be visited exactly once by a single vehicle; each vehicle route must originate and terminate at the depot; and the cumulative demand of customers served on any given route must not surpass the vehicle's capacity.1  
A standard mathematical formulation for the CVRP, essential for understanding its structure and constraints, can be expressed as an integer linear program (ILP). One common three-index vehicle flow formulation is as follows 2:  
Let V={0,1,…,n} be the set of vertices, where vertex 0 represents the depot and vertices 1,…,n represent the customers. Let K be the set of available homogeneous vehicles, each with capacity Q. Let cij​ be the cost (e.g., distance) of travel between vertex i and vertex j. Let di​ be the demand of customer i (d0​=0).  
The decision variables are:

* xijk​=1 if vehicle k travels directly from vertex i to vertex j, and 0 otherwise.  
* yik​=1 if customer i is visited by vehicle k, and 0 otherwise (though often yik​ can be derived from xijk​).

The objective function is to minimize the total travel cost:  
Minimize k∈K∑​i∈V∑​j∈V,i=j∑​cij​xijk​  
Subject to constraints:

1. Each customer is visited exactly once: k∈K∑​i∈V,i=j∑​xijk​=1,∀j∈{1,…,n} (Or, using yik​: ∑k∈K​yik​=1,∀i∈{1,…,n} 2)  
2. Each vehicle starts at the depot: j∈{1,…,n}∑​x0jk​=1,∀k∈K  
3. Each vehicle that leaves the depot must return to the depot: i∈{1,…,n}∑​xi0k​=1,∀k∈K  
4. Flow conservation at each customer node for each vehicle: $$ \\sum\_{i \\in V, i \\neq j} x\_{ijk} \- \\sum\_{l \\in V, l \\neq j} x\_{jlk} \= 0, \\quad \\forall j \\in {1, \\dots, n}, \\forall k \\in K $$  
5. Capacity constraint for each vehicle: $$ \\sum\_{i \\in {1, \\dots, n}} d\_i \\left( \\sum\_{j \\in V, j \\neq i} x\_{jik} \\right) \\leq Q, \\quad \\forall k \\in K $$ (Alternative capacity formulation using subtour elimination constraints, often Miller-Tucker-Zemlin, or by ensuring load variables uik​ satisfy uik​+dj​≤ujk​+Q(1−xijk​) 2)  
6. Binary variable constraints: xijk​∈{0,1},∀i,j∈V,k∈K

The CVRP is fundamentally NP-hard.1 This classification implies that finding exact optimal solutions becomes computationally intractable for all but small-scale instances. The combinatorial explosion in the number of possible routes 10 as the number of customers increases is the primary driver of this complexity. This inherent difficulty directly necessitates the exploration of sophisticated heuristic, metaheuristic, and learning-based approaches, as exact methods fail to provide solutions within practical timeframes for real-world problem sizes.7 The research presented herein is motivated by this fundamental challenge.  
While this paper focuses on the "pure" CVRP to allow for a clear comparison of solution methodologies, it is important to acknowledge its place within a broader family of Vehicle Routing Problems (VRPs). Common variants include the VRP with Time Windows (VRPTW), Split Delivery VRP (SDVRP), Multi-Depot VRP (MDVRP), and Green VRP (GVRP), each introducing additional complexities and constraints.6 Advances in solving the fundamental CVRP often provide foundational techniques and insights applicable to these more complex variants, underscoring the potential broad impact of this research.

### **B. Research Vision: A Tripartite Investigation into Novel CVRP Solvers**

This research aims to conduct a rigorous and comprehensive comparative study of three distinct and advanced paradigms for solving the CVRP:

1. **Augmented Lagrangian Methods (ALM):** A classical mathematical optimization technique adapted for the combinatorial nature of CVRP.  
2. **Reinforcement Learning (RL) with Classical Method Integration:** Employing state-of-the-art RL architectures, potentially hybridized with traditional heuristics or optimization components to enhance performance.  
3. **Reinforcement Learning (RL) with Quantum Method Integration:** A pioneering exploration leveraging quantum computing algorithms, such as the Quantum Approximate Optimization Algorithm (QAOA) or Variational Quantum Eigensolver (VQE), within an RL framework, likely involving Quadratic Unconstrained Binary Optimization (QUBO) formulations for CVRP subproblems.

The novelty of this investigation lies not only in the individual exploration of each approach but critically in their direct comparison on a common problem, using standardized benchmarks and metrics. The inclusion of the RL-Quantum paradigm, in particular, positions this work at the cutting edge of current research efforts.

### **C. Positioning the Research: Potential Contributions and Paper Framing Strategy**

This study is framed as an interdisciplinary investigation into the frontiers of CVRP solving, bridging the domains of classical operations research, artificial intelligence/machine learning, and quantum computing. The central narrative will revolve around the comparative strengths and weaknesses of these diverse approaches, seeking to identify pathways towards more robust, scalable, and potentially higher-quality solutions for this NP-hard problem.  
**Potential Contributions:**

* **Methodological Advancements:**  
  * For ALM: Development or adaptation of an ALM formulation specifically tailored to CVRP's combinatorial structure, potentially incorporating advanced decomposition techniques or novel parameter update strategies.  
  * For RL-Classical: Design and implementation of a state-of-the-art RL architecture (e.g., Graph Neural Network (GNN) or Transformer-based) for CVRP, featuring meticulous engineering of state representations, action spaces, and reward functions, along with effective hybridization with classical heuristics.  
  * For RL-Quantum: A pioneering exploration of RL for managing and optimizing quantum algorithms (QAOA/VQE) applied to CVRP subproblems formulated as QUBOs. This includes the strategic use of techniques like the Augmented Lagrangian-inspired Method (ALiM) for efficient constraint handling in the quantum domain.16  
* **Empirical Insights:** A comprehensive and fair empirical comparison of these three paradigms on standard CVRP benchmark instances. This will yield valuable insights into their relative performance concerning solution quality, computational efficiency (classical and quantum), scalability with problem size, and practical applicability.  
* **Interdisciplinary Bridge:** The research will demonstrate how techniques from traditionally disparate fields can be synergistically combined to tackle complex optimization problems. For instance, the ALM approach, particularly ALiM, is not merely a classical baseline but can serve as a critical enabler for the RL-Quantum methodology by facilitating the efficient encoding of CVRP constraints (like capacity) into QUBOs suitable for quantum solvers.10 This reduction in the number of required qubits for handling inequality constraints is vital for the feasibility of near-term quantum implementations. This interconnectedness highlights a non-obvious but crucial synergy that will be a recurring theme.

The successful integration of these diverse methodologies to solve a common, challenging problem like CVRP will showcase the power of interdisciplinary approaches and potentially inspire similar research endeavors in other combinatorial optimization domains.  
To provide an immediate high-level understanding, Table 1 offers a comparative overview of the three proposed paradigms.

**Table 1: Comparative Overview of the Three Proposed CVRP Solution Paradigms**

| Approach | Core Principle | Key Strengths (Theoretical/Practical) | Main Challenges for CVRP | Expected Scalability for CVRP (Hypothesized) | Key Novelty in this Paper |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Augmented Lagrangian (ALM) | Transforms constrained CVRP into a sequence of simpler, penalized unconstrained/bound-constrained subproblems. | Strong theoretical foundation for convex problems; can provide dual bounds; good at handling hard constraints; decomposition possible. | Application to discrete/non-convex CVRP; tuning penalties/multipliers; subproblem complexity; potentially slow convergence. | Moderate, depends on subproblem solver efficiency. | Novel decomposition or ALM adaptation for CVRP structure; advanced multiplier/penalty update schemes. |
| RL-Classical | Learns a policy to construct CVRP solutions sequentially, often using GNNs/Transformers; can integrate classical heuristics. | Data-driven policy learning; fast inference once trained; adaptable to dynamic elements; potential for generalization. | Sample inefficiency in training; reward shaping; handling hard constraints; generalization to unseen instance types/sizes; local optima. | Good for moderate sizes; generalization is key. | State-of-the-art RL architecture (e.g., GNN/Transformer) with careful CVRP-specific state/action/reward engineering and novel hybridization with classical methods. |
| RL-Quantum | Uses RL to manage/tune quantum algorithms (QAOA/VQE) that solve QUBO-formulated CVRP subproblems. | Potential for quantum speedup on subproblems; RL automates complex quantum algorithm design/tuning; explores novel solution space. | NISQ hardware limitations (qubits, noise, connectivity); QUBO formulation for CVRP constraints (ALiM helps); VQA trainability (barren plateaus); high cost. | Very small instances/subproblems currently; highly dependent on quantum hardware progress. | Pioneering use of RL for tuning/managing QAOA/VQE applied to CVRP subproblems (via QUBO with ALiM); comparative analysis against classical and RL-classical methods for a complex logistics problem. |

## **II. Deep Dive: Augmented Lagrangian Methods for CVRP**

Augmented Lagrangian Methods (ALM) offer a powerful framework for constrained optimization, and their adaptation to combinatorial problems like CVRP presents both opportunities and challenges. This section explores the theoretical underpinnings of ALM, proposes a tailored architectural design for CVRP, critically analyzes its potential, and outlines an implementation blueprint.

### **A. Scholarly Foundations: ALM in Constrained and Combinatorial Optimization**

The fundamental concept of ALM involves reformulating a constrained optimization problem into a series of unconstrained or less constrained subproblems. This is achieved by augmenting the objective function with penalty terms for constraint violations and terms involving estimates of Lagrange multipliers.17 Unlike pure penalty methods, ALM incorporates Lagrange multiplier updates, which allows it to find optimal solutions without requiring penalty parameters to tend towards infinity, thereby mitigating issues of ill-conditioning that can plague pure penalty approaches.17 The Lagrange multipliers themselves provide valuable sensitivity information regarding the constraints, guiding the search process effectively. The iterative updates of these multipliers, typically based on the magnitude of constraint violations, reflect the dual nature of the method.17  
Convergence properties of ALM are well-studied for continuous, particularly convex, optimization problems. Linear convergence rates can often be established under conditions such as quadratic growth of the objective function or when strict complementarity and bounded solution sets hold (analogous concepts may exist for combinatorial problems).20 For instance, studies on semidefinite programs (SDPs) have shown linear convergence of primal and dual iterates under strict complementarity.20  
The application of ALM to purely discrete or combinatorial optimization problems is less direct than for continuous problems and often involves relaxations or specialized decomposition schemes.21 For VRP variants, ALM has been explored by dualizing specific sets of constraints, such as task allocation or capacity constraints, and then decomposing the problem into a series of more manageable routing or assignment subproblems.24 For example, 24 describes dualizing task allocation constraints in a multi-compartment VRP, leading to identical routing subproblems. 25 applies augmented Lagrangian relaxation (ALR) to a VRP variant with a nonlinear objective, again utilizing decomposition. The nature of ALM, particularly variants like the Alternating Direction Method of Multipliers (ADMM) 17, aligns philosophically with column generation, a powerful exact method for VRP. Both techniques decompose the problem into a master problem and one or more subproblems. This conceptual link can inspire novel ALM decomposition strategies for CVRP, where ALM might relax linking constraints in a manner analogous to how column generation handles route selection and generation.

### **B. Architectural Design: An ALM Framework Tailored for CVRP**

A successful ALM framework for CVRP hinges on an effective decomposition strategy and robust mechanisms for updating multipliers and penalty parameters.  
1\. Decomposition Strategies and Subproblem Formulation:  
The choice of which CVRP constraints to relax (dualize) determines the nature and complexity of the resulting subproblems. Potential strategies include:

* **Relaxing Customer-to-Route Assignment Constraints:** If the constraints that assign each customer to exactly one vehicle and route are relaxed, the problem could decompose by vehicle. Each subproblem might then become a form of capacity-constrained shortest path problem (e.g., Elementary Shortest Path Problem with Resource Constraints \- ESPPRC) for a single vehicle, aiming to find the best route through a subset of customers given modified costs from the Lagrangian terms. This approach is inspired by works that decompose VRPs into routing subproblems.25  
* **Relaxing Routing/Sequencing Constraints:** Alternatively, if the complex routing constraints (ensuring valid tours) are relaxed, the subproblems might resemble assignment problems (assigning customers to positions in routes) or flow problems, with penalties for violating tour structures or capacities.  
* **ADMM-based Decomposition:** If the CVRP formulation can be split into two primary sets of variables with separable objectives linked by linear constraints (e.g., variables representing vehicle flows and variables representing customer assignments), ADMM could be applied.17 This would lead to alternatingly solving subproblems for each set of variables.

The mathematical formulation of these subproblems is critical. For instance, an ESPPRC subproblem, while still NP-hard, can be solved using dynamic programming or specialized label-setting algorithms for paths of moderate length. The trade-off here is crucial: simpler subproblems are computationally cheaper per iteration but may lead to a weaker dual bound and thus more ALM iterations. Conversely, more complex subproblems might provide stronger bounds but increase the per-iteration cost. This balance is a central research question in designing the ALM.  
2\. Advanced Multiplier and Penalty Parameter Update Schemes:  
Standard multiplier updates often follow the form $ \\lambda\_i^{k+1} \= \\lambda\_i^k \+ \\rho\_k g\_i(x^k) $, where gi​(xk) is the violation of constraint i at iteration k, and ρk​ is the penalty parameter.19 For CVRP, specific multipliers would be associated with capacity constraints, customer visit constraints (each customer visited once), and flow conservation constraints.  
Penalty parameter management is vital. If ρk​ is too small, convergence to feasibility can be slow; if too large, the subproblems become ill-conditioned and difficult to solve.28 Adaptive strategies are preferred over fixed schedules. These might involve increasing ρk​ when feasibility improvements stall, or decreasing it if subproblems become too hard, guided by the degree of constraint violation or the behavior of the Lagrange multipliers.29 Some research suggests lower limits on penalty parameters to maintain convexity of the augmented Lagrangian.29  
Solutions obtained from an ALM approach, even if not fully optimal, can serve as high-quality warm starts for other heuristic methods or even for the RL-based approaches discussed later. ALM tends to find feasible or near-feasible solutions relatively quickly, which can effectively guide the exploration phase of an RL agent or provide a strong initial population for genetic algorithms or other metaheuristics.

### **C. Critical Analysis: Efficacy, Limitations, and Scalability of ALM for CVRP**

**Pros:**

* **Improved Convergence over Pure Penalty Methods:** ALM typically exhibits better numerical stability and convergence properties because it does not require penalty parameters to approach infinity, thus avoiding severe ill-conditioning.17  
* **Effective Constraint Handling:** The method is inherently designed to manage complex constraints, which is a hallmark of CVRP.19 Its strength lies in rigorously ensuring constraint satisfaction (e.g., vehicle capacities, customer visit requirements). This contrasts with some RL approaches that might learn to violate "soft" constraints if not meticulously designed with appropriate penalties.  
* **Decomposition Potential:** ALM, especially through variants like ADMM, can decompose large, complex problems into smaller, potentially parallelizable subproblems.17  
* **Dual Bound Generation:** ALM can provide lower bounds (for minimization problems) on the optimal solution value via the dual problem, which is useful for assessing the quality of feasible solutions found.24

**Cons:**

* **Convergence Rate:** While better than pure penalty methods, convergence can still be linear and thus slower than, for example, Sequential Quadratic Programming (SQP) methods for continuous problems.19  
* **Parameter Tuning:** The performance of ALM can be sensitive to the choice of initial penalty parameters, their update strategy, and the initial Lagrange multipliers. This tuning can be problem-dependent and may require experimentation.19  
* **Subproblem Complexity:** The computational effort per iteration heavily depends on the difficulty of solving the subproblems. If subproblems themselves are NP-hard (like ESPPRC), overall solution times can be substantial.12  
* **Theoretical Guarantees for Discrete Problems:** While strong for convex continuous problems, convergence guarantees for non-convex or purely discrete problems like CVRP are generally weaker.19 The application to CVRP often involves solving discrete subproblems or using ALM within a larger framework like branch-and-bound where ALM solves relaxations. This interface between continuous ALM theory and the discrete reality of CVRP is a significant research challenge.

Scalability for CVRP:  
The scalability of ALM for CVRP will largely be determined by the efficiency of the chosen subproblem solver and the number of ALM iterations required for convergence. For very large instances, the repeated solution of even moderately complex subproblems can become a bottleneck. However, its ability to produce good feasible solutions and dual bounds relatively quickly might make it competitive for medium-sized instances or as a component in a hybrid approach.

### **D. Python Implementation Blueprint for ALM-CVRP**

A modular Python implementation is envisioned:

* **Core Modules:**  
  * cvrp\_data\_handler.py: Parses CVRP instances (e.g., from CVRPLIB format) and manages problem data (coordinates, demands, capacities, distance matrix).  
  * alm\_subproblem\_solver.py: Contains functions or classes to solve the specific subproblems defined by the chosen decomposition (e.g., an ESPPRC solver using dynamic programming, or an interface to a MIP solver like Gurobi/CPLEX via their Python APIs if subproblems are formulated as MIPs).  
  * alm\_optimizer.py: Implements the main ALM iterative loop, manages Lagrange multiplier updates, penalty parameter adjustments, checks for convergence criteria (e.g., primal/dual feasibility, objective stabilization), and orchestrates calls to the subproblem solver.  
* **Supporting Utilities:**  
  * solution\_checker.py: Verifies the feasibility of generated CVRP solutions (all constraints met) and calculates their objective function value.  
  * logger.py: For logging progress, constraint violations, objective values, and timing information during the ALM iterations.  
* **Key Libraries:**  
  * numpy: For numerical operations, especially matrix and vector manipulations.  
  * scipy.optimize: May be useful if subproblems involve continuous relaxations or for auxiliary optimization tasks.  
  * MIP Solver APIs (e.g., gurobipy, cplex): Essential if subproblems are complex enough to be formulated and solved as mixed-integer programs.  
  * Potentially a graph library like networkx if subproblems involve explicit graph algorithms.

This structure promotes modularity, allowing different subproblem solvers or update strategies to be tested and integrated with relative ease.

## **III. Deep Dive: Reinforcement Learning with Classical Integration for CVRP**

Reinforcement Learning (RL) has emerged as a promising data-driven paradigm for tackling complex combinatorial optimization (CO) problems, including the CVRP. This section delves into the scholarly foundations of RL for routing, proposes a novel RL framework design incorporating classical heuristics, critically analyzes its potential, and outlines a Python implementation strategy.

### **A. Scholarly Foundations: RL Paradigms for Vehicle Routing**

RL frames CO problems as sequential decision-making processes where an agent learns a policy to construct a solution step-by-step.8 For CVRP, this typically involves an agent (e.g., a neural network) deciding which customer to visit next at each step of route construction.  
Key RL Algorithms and Architectures for VRP:  
The choice of RL algorithm and neural architecture is pivotal. There's a discernible evolution in RL architectures for VRP, moving from simpler Recurrent Neural Network (RNN)-based models towards more sophisticated Graph Neural Networks (GNNs) and Transformer/Attention-based models.35 This progression reflects a deeper understanding of the necessity to effectively capture the inherent graph structure of VRP instances and the long-range dependencies between decision steps. VRP solutions are essentially permutations of nodes, characterized by intricate interactions; attention mechanisms excel in sequence-to-sequence tasks and learning contextual information in such problems, while GNNs are adept at learning from graph-structured data.

* **Policy Gradient (PG) Methods:**  
  * Examples: REINFORCE, Advantage Actor-Critic (A2C), Proximal Policy Optimization (PPO).8  
  * Principle: Directly learn a parameterized policy $ \\pi\_\\theta(a|s) $ that maps states to actions (or distributions over actions).  
  * Strengths: Effective for continuous or large discrete action spaces (common in VRP where the next node is chosen from many possibilities); can learn stochastic policies, which aids exploration.40  
  * Weaknesses: Often suffer from high variance in gradient estimates, which can slow down or destabilize training; may converge to local optima rather than global ones.35  
  * Actor-Critic variants (e.g., A2C, PPO) mitigate high variance by learning a value function (the critic) to baseline the policy gradient updates.35 PPO, for instance, has shown strong performance in various CO tasks.38  
* **Value-Based Methods:**  
  * Example: Deep Q-Networks (DQN).36  
  * Principle: Learn an action-value function $ Q(s,a) $ that estimates the expected return of taking action a in state s. The policy is then typically derived by choosing actions that maximize Q(s,a).  
  * Strengths: Can be more sample-efficient and stable than PG methods when they work well, particularly with techniques like experience replay and target networks.40  
  * Weaknesses: Standard DQN struggles with large or continuous action spaces, as it requires computing Q-values for all possible actions.40 This is a significant limitation for VRP with many customers. They can also suffer from overestimation of Q-values.  
* **Attention Models and Graph Neural Networks (GNNs):**  
  * These are not RL algorithms per se, but powerful neural network architectures used within RL frameworks to process VRP instances.  
  * Encoders (GNNs/Transformers) create rich embeddings of customer nodes and the depot, capturing their features and relational information (e.g., distances).35 Some advanced models like EEMHA explicitly incorporate edge information into the attention mechanism.38  
  * Decoders (often attention-based) use these embeddings and the current state of the constructed solution (e.g., current location, remaining capacity) to autoregressively select the next action (e.g., the next customer to visit).36

A notable trend in the application of RL to CO is the pursuit of end-to-end learning systems.8 Such systems aim to learn the entire solution-construction process directly from raw problem inputs to final solutions, minimizing reliance on manually engineered intermediate heuristics. This approach, while ambitious, places a substantial learning burden on the RL agent and its underlying neural architecture to discover effective solution strategies autonomously.

### **B. Architectural Design: A Novel RL Framework for CVRP**

This research will focus on a policy gradient approach, likely PPO, due to its robustness and common application in CO, combined with a state-of-the-art neural architecture.  
1\. Chosen RL Algorithm: Proximal Policy Optimization (PPO)  
PPO is selected for its balance of sample efficiency, stability, and ease of implementation compared to other policy gradient methods. It has demonstrated strong performance on various CO problems, including VRP.38 An actor-critic setup will be used, where the actor determines the policy and the critic evaluates states or state-action pairs to reduce variance.  
**2\. State-of-the-Art Neural Architectures (GNNs/Transformers with Attention):**

* **Encoder:** A Transformer-based encoder, inspired by architectures like the Attention Model (AM) by Kool et al. 36, will be employed. This encoder will take the set of customer locations, demands, and depot information as input. It will generate context-aware embeddings for each node. Consideration will be given to incorporating edge features (e.g., distances) directly into the attention mechanism, possibly through an Edge-Embedded Multi-Head Attention (EEMHA) layer 38, to provide a richer representation of the problem graph.  
* **Decoder:** An attention-based decoder will autoregressively construct vehicle routes. At each step, it will attend to the node embeddings produced by the encoder and the current dynamic state of the route (e.g., current vehicle location, remaining capacity, set of unvisited customers) to output a probability distribution over feasible next customers to visit.36

3\. CVRP-Specific State, Action, and Reward Engineering:  
The design of these components is critical for successful RL application.  
**Table 2: CVRP State, Action, and Reward Design for the RL-Classical Approach**

| Component | Detailed Description for CVRP | Key Considerations/Challenges | Relevant Information Sources |
| :---- | :---- | :---- | :---- |
| **State \- Static Features** | Node coordinates (depot, customers), customer demands. | Normalization of coordinates and demands. Representing graph structure implicitly or explicitly. | 56 |
| **State \- Dynamic Features** | Current vehicle location, current vehicle load/remaining capacity, mask of visited/unvisited customers for the current route, (optionally) current time elapsed/distance traveled on route, ID of the current vehicle being routed. | Efficiently updating these features at each step. Scaling for different numbers of customers/vehicles. Ensuring all necessary information for constraint checking is present. | 32 |
| **State \- Contextual Features** | Output embeddings from the GNN/Transformer encoder (graph embedding, node embeddings). For multi-vehicle problems, potentially information about the state of other vehicles/routes already constructed. | Ensuring encoder captures relevant global problem structure. Handling dependencies between routes if constructed sequentially. | 36 |
| **Action** | Selection of the next customer to visit by the current vehicle. If all customers are served or current vehicle is full, action could be to return to depot or select a new vehicle. | Managing a potentially large discrete action space (number of customers). Ensuring actions lead to feasible states. | 36 |
| **Masking** | Infeasible actions are masked out from the policy's output distribution. This includes: visiting an already visited customer, selecting a customer whose demand exceeds remaining vehicle capacity, selecting the depot if the route is empty (unless no customers left). | Correctly and efficiently implementing the masking logic is crucial for learning valid policies and avoiding wasted exploration. | 36 |
| **Reward** | Primary: Negative of the total distance traveled for all routes upon completion of a full CVRP solution. Secondary: Penalties for constraint violations (e.g., capacity overflow) if not strictly enforced by masking. Sparse rewards (only at episode end) make credit assignment difficult. | Designing effective reward functions that guide the agent towards globally optimal solutions. Balancing exploration and exploitation. The temporal credit assignment problem is significant with sparse rewards. | 32 |

One of the significant challenges in applying RL to CVRP is the large action space, especially as the number of customers grows. Attention mechanisms in the decoder help by learning to focus on a relevant subset of potential next actions. Another fundamental issue is the temporal credit assignment problem: the primary reward (total tour length) is typically sparse, received only after all routes are completed.40 This makes it difficult for the agent to determine which specific actions in a long sequence were good or bad. Actor-critic methods help by providing intermediate value estimates, and careful reward shaping could be explored, though this carries the risk of inadvertently guiding the agent towards suboptimal behaviors if not designed perfectly.  
4\. Synergizing RL with Classical Heuristics:  
To enhance the RL agent's learning process and final solution quality, integration with classical heuristics will be explored:

* **Guided Exploration/Initialization:** Use solutions from simple classical heuristics (e.g., Nearest Neighbor 62, Clarke-Wright Savings 63) to provide initial trajectories or to bias the early exploration of the RL agent, potentially accelerating learning.  
* **Local Search Refinement:** Employ classical local search operators (e.g., 2-opt, Or-opt, relocate, exchange) as a post-processing step to improve the routes generated by the RL agent. The RL agent can learn to produce good "global" route structures, which local search can then fine-tune. This is a common and effective hybridization strategy.  
* **RL for Heuristic Control:** Train the RL agent to dynamically select among a portfolio of classical heuristics or to tune their parameters during the solution process, adapting the heuristic strategy to the problem instance or current solution state. 95 mentions RL fine-tuning parameters of a Genetic Algorithm.  
* **RL in Column Generation:** A more advanced integration involves using RL to solve the pricing subproblem (generating new routes with negative reduced cost) within a column generation framework for CVRP.58 This leverages RL's ability to learn complex patterns for a traditionally hard subproblem.

### **C. Critical Analysis: Performance, Generalization, and Challenges of RL in CVRP**

**Pros:**

* **Adaptive Policy Learning:** RL agents can learn complex, non-linear policies directly from interaction with the environment (or simulations), potentially discovering strategies not captured by handcrafted heuristics.61  
* **Fast Inference:** Once trained, neural network-based policies can generate solutions very quickly, which is advantageous for dynamic or real-time applications.53 This makes RL essentially a learned construction heuristic.  
* **Adaptability to Dynamics:** If trained in dynamic environments or with dynamic features, RL agents can learn policies that adapt to changing conditions (e.g., new customer requests, travel time variations).32  
* **Flexibility for Hybridization:** RL frameworks can be flexibly combined with other ML techniques or classical optimization methods.64

**Cons and Challenges:**

* **Generalization:** A major hurdle is the generalization of trained RL models to problem instances of different sizes, scales, or underlying data distributions than those seen during training.8 The "No Free Lunch" theorem suggests that a policy learned for one instance distribution may not perform well on another without retraining or sophisticated transfer learning techniques. Data augmentation strategies, such as those inspired by POMO 12, can help improve generalization.  
* **Training Cost and Sample Inefficiency:** Training deep RL models can be computationally intensive and require a vast number of interactions (episodes) with the environment, making it data-hungry.61  
* **High Variance (especially for PG methods):** Policy gradient methods often exhibit high variance in their gradient estimates, leading to unstable training and slow convergence.35  
* **Reward Function Design:** The performance of an RL agent is highly sensitive to the reward function. Designing rewards that accurately reflect the desired objectives without leading to unintended "reward hacking" is a non-trivial task.61  
* **Solution Quality vs. SOTA:** While RL can produce good solutions quickly, achieving the same level of optimality as highly tuned, problem-specific classical solvers (like LKH3 or state-of-the-art commercial solvers) is often challenging, especially for static, offline CVRP instances.57  
* **Hard Constraint Handling:** Enforcing hard CVRP constraints (e.g., capacity, every customer visited exactly once) strictly within an RL framework can be difficult. It often requires careful action masking, significant penalties in the reward function, or specialized network architectures, and the agent might still learn to violate them if the penalties are not severe enough.65

**Specific CVRP Issues:** Beyond general RL challenges, specific issues for CVRP include effectively managing vehicle capacities during route construction, ensuring all customer demands are met, and coordinating multiple vehicle routes (either by training a single policy to output all routes sequentially, or by using multi-agent RL approaches, which introduce their own complexities).

### **D. Python Implementation Blueprint for RL-Classical CVRP**

A modular Python implementation will facilitate development and experimentation:

* **Core Modules:**  
  * cvrp\_environment.py: Implements the CVRP environment following the gymnasium (formerly OpenAI Gym) API. This module will handle state representation, action application, transition dynamics (e.g., updating vehicle load, visited customers), reward calculation, and termination conditions.  
  * neural\_networks.py: Defines the neural network architectures (e.g., GNN/Transformer encoder, attention-based decoder) using PyTorch 47 or TensorFlow.52 This module will be separate from the agent logic.  
  * rl\_agent.py: Contains the implementation of the chosen RL algorithm (e.g., PPO), including policy and value function updates, loss calculations, and interaction with the environment.  
  * experience\_buffer.py: If an off-policy algorithm like DQN were used, or for certain actor-critic implementations, a replay buffer would be necessary to store and sample past experiences. For PPO, this would store trajectories from policy rollouts.  
  * classical\_integration.py: Provides interfaces or wrappers for any classical heuristics (e.g., local search operators, initial solution generators) that are integrated with the RL framework.  
* **Supporting Utilities:**  
  * trainer.py: Manages the overall training loop, including episode generation, agent updates, logging, and model checkpointing.  
  * config\_handler.py: Loads and manages configuration parameters for experiments (e.g., learning rates, network dimensions, batch sizes).  
* **Key Libraries:**  
  * pytorch or tensorflow: For defining and training neural networks.  
  * Graph-specific libraries like torch\_geometric or dgl if implementing GNNs.  
  * numpy: For numerical computations.  
  * gymnasium: For structuring the CVRP environment.  
  * matplotlib: For visualizing routes and training progress.62

This structure will allow for systematic testing of different network components, RL algorithms, and hybridization strategies.

## **IV. Deep Dive: Reinforcement Learning with Quantum Integration for CVRP**

This section ventures into the highly exploratory domain of integrating Reinforcement Learning with quantum computing methods to solve the CVRP. The core idea is to leverage quantum algorithms for specific, computationally hard subproblems within a larger RL or classical optimization framework, and to use RL to manage or optimize these quantum components.

### **A. Scholarly Foundations: Quantum Optimization and RL-Quantum Hybrids**

1\. CVRP to QUBO Transformation Strategies:  
Quantum annealers and many gate-based variational quantum algorithms like QAOA and VQE are designed to solve problems formulated as Quadratic Unconstrained Binary Optimization (QUBO) problems.11 A QUBO problem seeks to minimize an objective function of the form $ E(x) \= \\sum\_{i,j} A\_{i,j} x\_i x\_j \+ \\sum\_i B\_i x\_i $, where xi​ are binary variables.  
Transforming a constrained problem like CVRP (or its subproblems) into a QUBO involves encoding decision variables as binary variables and representing constraints as penalty terms added to the objective function. The magnitude of these penalty coefficients is crucial: if too small, constraints may be violated; if too large, they can dominate the original objective, making it hard to find good solutions.69  
A significant challenge is handling inequality constraints (like vehicle capacity) in QUBOs. Standard methods often require introducing slack variables to convert inequalities to equalities, which increases the number of qubits needed.10 The **Augmented Lagrangian-inspired Method (ALiM)** offers a more qubit-efficient way to incorporate such constraints.10 ALiM modifies the objective function by adding both quadratic penalty terms (like standard penalty methods) and linear terms related to Lagrange multipliers, effectively avoiding the need for slack variables for inequalities and thus reducing qubit requirements.16 However, the coefficients of these ALiM terms (penalty parameters and multipliers) often require careful tuning.  
2\. Variational Quantum Algorithms (QAOA/VQE) for CVRP Subproblems:  
Variational Quantum Algorithms (VQAs) are hybrid quantum-classical algorithms well-suited for Noisy Intermediate-Scale Quantum (NISQ) devices. They use a parameterized quantum circuit (an "ansatz") to prepare a trial quantum state, measure an observable corresponding to the problem's objective function, and then use a classical optimizer to update the circuit parameters to minimize this observable.

* **Quantum Approximate Optimization Algorithm (QAOA):** QAOA is specifically designed for combinatorial optimization problems.10 It involves applying alternating layers of a "cost Hamiltonian" (encoding the problem objective) and a "mixer Hamiltonian" (inducing exploration of the solution space). The classical optimizer tunes the parameters (angles) associated with these layers. QAOA is a strong candidate for solving QUBO-formulated CVRP subproblems.  
* **Variational Quantum Eigensolver (VQE):** VQE aims to find the ground state (lowest energy) of a given Hamiltonian.81 If a CVRP subproblem's QUBO can be mapped to a Hamiltonian whose ground state corresponds to the optimal solution, VQE can be used.

For CVRP, VQAs would likely be applied to solve decomposed subproblems, such as finding an optimal route for a single vehicle (an ESPPRC-like problem if capacity is involved) or solving the pricing problem in a column generation framework.10 The success of this approach heavily relies on the effectiveness of the QUBO formulation, particularly how ALiM handles constraints like capacity, as this directly impacts the size and complexity of the quantum subproblem.  
3\. RL for Optimizing Quantum Circuit Parameters and Ansatz Design:  
A key challenge in using VQAs is the optimization of their classical parameters (e.g., QAOA angles, ansatz gate parameters) and the design of the ansatz circuit itself. The optimization landscape can be complex, featuring many local minima and "barren plateaus" (regions where gradients vanish, hindering optimization).87  
RL offers a promising avenue to address these challenges:

* **Parameter Optimization:** RL agents can be trained to find optimal or near-optimal parameters for QAOA/VQE circuits, potentially navigating complex landscapes more effectively than standard classical optimizers.75  
* **Quantum Architecture Search (QAS):** RL can be used to automatically design the structure of the quantum ansatz, selecting gates and their arrangement to tailor the circuit to the specific problem or hardware.81 This could lead to the discovery of novel, more efficient ansatzes.  
* **Resource Management:** RL could also optimize other aspects of VQA execution, such as shot allocation (number of measurements) to balance accuracy and computational cost.86

This RL-for-quantum-control aspect introduces a hierarchical structure: an outer loop might handle the classical decomposition of CVRP, while an inner RL loop optimizes the quantum solver for the subproblems. The state representation for the RL agent tuning quantum parameters would need to capture information about the quantum circuit (e.g., current parameters, depth) and the subproblem's characteristics, while actions would involve adjustments to these parameters or the circuit structure. The reward would be linked to the quality of the subproblem solution obtained from the quantum device and potentially the quantum resources consumed.

### **B. Architectural Design: A Hybrid RL-Quantum Framework for CVRP**

The proposed RL-Quantum framework will likely involve a classical decomposition of the CVRP, with quantum algorithms solving specific subproblems, and RL potentially playing a role in optimizing these quantum solvers.  
1\. Problem Decomposition for Quantum Processing:  
A practical approach for near-term quantum devices is to decompose the CVRP into smaller subproblems.

* **Column Generation (CG) Framework:** As explored in 10, CG decomposes the CVRP into a Restricted Master Problem (RMP) that selects a combination of existing routes, and one or more subproblems (pricing problems) that generate new, cost-reducing routes. The pricing problem, often an ESPPRC, could be formulated as a QUBO and tackled by a VQA.  
* **Cluster-First, Route-Second:** Classical clustering algorithms can partition customers into groups, and then a VQA can solve the TSP-like routing problem for each cluster.11

This research will lean towards the CG approach due to its strong theoretical basis and recent explorations with quantum methods. The subproblem will be generating a single vehicle's route considering capacity, formulated as a QUBO using ALiM.  
2\. RL Agent Design for Quantum Environment (States, Actions, Rewards specific to QAOA/VQE tuning):  
If RL is used to optimize the VQA (e.g., QAOA) solving the CVRP subproblem:  
**Table 3: CVRP State, Action, and Reward Design for the RL-Quantum Approach (Parameter/Ansatz Tuning)**

| Component | Detailed Description for Quantum Context | Key Considerations/Challenges | Relevant Information Sources |
| :---- | :---- | :---- | :---- |
| **State \- Quantum Circuit Properties** | Current QAOA/VQE parameters (e.g., angles $ \\gamma, \\beta $ for QAOA layers), ansatz structure (if adaptable, e.g., number of layers, gate types), circuit depth, number of qubits, gate counts, possibly estimated noise levels of the QPU. | Representing continuous parameters and discrete structures. Scalability with circuit size. Capturing relevant information for parameter updates. | 80 |
| **State \- Problem Instance Properties** | Characteristics of the current CVRP subproblem's QUBO (e.g., size, density of quadratic terms, magnitude of coefficients from ALiM). History of energy measurements or solution qualities from previous VQA runs for this subproblem. | Extracting salient features from the QUBO that inform parameter tuning. Handling variability if subproblems change dynamically. |  |
| **Action \- Parameter Adjustment** | Small incremental changes to QAOA/VQE parameters (e.g., angles $ \\Delta\\gamma, \\Delta\\beta $). Selecting parameters from a discrete set. | Defining an appropriate action space (continuous vs. discrete). Step sizes for adjustments. Avoiding overly aggressive changes that destabilize learning. | 75 |
| **Action \- Ansatz Modification** | Adding/removing layers or gates to the VQA ansatz. Changing connectivity or types of entangling gates (if QAS is performed). Adjusting the number of measurement shots. | Large combinatorial action space for ansatz design. Ensuring modified ansatz remains valid and implementable. Balancing circuit expressiveness with depth/noise constraints. | 81 |
| **Reward \- Solution Quality** | Quality of the CVRP sub-route found by the VQA (e.g., negative of its cost/length, or its reduced cost if in CG). Improvement over a baseline for the subproblem. Binary reward for finding a valid sub-route meeting constraints. | Aligning subproblem reward with global CVRP objective. Handling noisy quantum outputs when evaluating solution quality. | 81 |
| **Reward \- Resource Penalty** | Penalty for excessive quantum resources used (e.g., circuit depth, two-qubit gate count, total shots), especially relevant for NISQ. | Balancing solution quality with resource efficiency. Defining appropriate penalty weights. | 81 |

The successful application of ALiM is a prerequisite here. Without an effective way to map capacity-constrained sub-routes to manageable QUBOs, the quantum part of the framework becomes infeasible.16 If RL can effectively learn to tune or design quantum circuits for these CVRP subproblems, the learned strategies might offer insights applicable to other combinatorial optimization problems on quantum computers, potentially discovering novel ansatz structures or parameter-setting heuristics that are non-intuitive to human researchers.81

### **C. Critical Analysis: Potential, Current Bottlenecks, and Future Viability of RL-Quantum CVRP**

**Potential:**

* **Quantum Speedups:** Quantum algorithms like Grover's (related to QAOA's mixer) or those underlying VQAs hold the theoretical promise of speedups for certain classes of optimization problems or search tasks.91  
* **Automated Quantum Algorithm Design:** RL can automate the complex and often heuristic process of designing VQA ansatzes and tuning their parameters, potentially leading to more performant quantum solutions.88

**Current Bottlenecks:**

* **NISQ Hardware Limitations:** This is the most significant bottleneck. Current quantum processors have a limited number of qubits, high gate error rates, restricted qubit connectivity, and can only support shallow circuit depths.10 These limitations severely constrain the size and complexity of CVRP subproblems that can be tackled.  
* **QUBO Formulation Challenges:** Efficiently and accurately mapping complex, constrained CO problems like CVRP subproblems (especially with capacity and routing logic) to QUBOs is non-trivial. Penalty coefficients are hard to tune, and the QUBO size can grow rapidly.11  
* **VQA Trainability:** VQAs suffer from issues like barren plateaus (vanishing gradients in flat optimization landscapes), which hinder the classical optimization of parameters, especially for larger circuits.87 The cost of repeatedly executing quantum circuits for function evaluations and gradient estimations (many shots per evaluation) is also very high.86  
* **RL Sample Inefficiency:** Training RL agents, especially for complex tasks like quantum circuit optimization, can require a large number of interactions (quantum circuit executions), which is expensive and time-consuming.61  
* **Noise Impact:** Noise in NISQ devices can significantly corrupt the quantum computation, leading to inaccurate objective function evaluations and hindering the RL agent's ability to learn effective policies.87 RL could potentially learn error mitigation strategies by adapting parameters or circuits to known noise models 89, but this is an active research area.

Future Viability:  
The future viability of RL-Quantum for CVRP is strongly tied to advancements in quantum hardware (more and better qubits, lower error rates, improved connectivity) and quantum algorithm development (more noise-resilient VQAs, better QUBO embedding techniques). In the near term, the focus should remain on hybrid algorithms where quantum computation provides a specific, demonstrable advantage for a well-defined and sufficiently small sub-task within a larger classical framework. This research is inherently high-risk but also high-reward; even negative results regarding quantum advantage, if rigorously obtained and analyzed, are valuable contributions to the field by systematically mapping the challenges and limitations for complex applied problems like CVRP.

### **D. Python Implementation Blueprint for RL-Quantum CVRP**

A modular implementation is crucial for managing the complexity of this hybrid approach:

* **Core Modules:**  
  * cvrp\_subproblem\_qubo\_formulator.py: Takes a CVRP subproblem definition (e.g., a set of customers, vehicle capacity for a single route) and constructs the corresponding QUBO matrix. This module will incorporate the ALiM logic for handling capacity constraints.16  
  * variational\_quantum\_circuits.py: Defines parameterized quantum circuits for QAOA and/or VQE using a library like Qiskit 84 or Pennylane.82 This includes functions to build the ansatz, apply Hamiltonians, and specify parameterization.  
  * quantum\_execution\_environment.py: Serves as an interface to quantum simulators or real quantum hardware (via cloud platforms). It takes a quantum circuit and parameters, executes it, and returns measurement outcomes or expectation values. This environment will be what the RL agent for quantum control interacts with.  
  * quantum\_rl\_optimizer\_agent.py: Implements the RL agent (e.g., PPO, DQN) responsible for tuning the VQA parameters or searching for an optimal ansatz structure. Its actions modify the quantum circuit or its parameters, and it receives rewards based on the VQA's performance on the CVRP subproblem.  
  * classical\_decomposition\_handler.py: (If applicable, e.g., for Column Generation) Manages the overall classical problem decomposition, generates subproblems for the quantum solver, and integrates subproblem solutions back into the main problem.  
* **Key Libraries:**  
  * qiskit or pennylane: For quantum circuit construction, simulation, and execution.  
  * numpy: For QUBO matrix manipulations and general numerical tasks.  
  * pytorch or tensorflow: For implementing the neural network components of the RL agent.  
  * Interface libraries for specific quantum hardware providers if real device access is planned.

This structure allows for independent development and testing of the QUBO formulation, the quantum circuits, the RL agent for quantum control, and the classical decomposition framework.

## **V. Rigorous Experimental Validation Plan**

A robust experimental plan is essential to rigorously evaluate and compare the three proposed CVRP solution methodologies. This involves careful selection of benchmark datasets, definition of comprehensive performance metrics, choice of appropriate baseline algorithms, and a clear strategy for cross-approach benchmarking.

### **A. Benchmark Dataset Selection and Preparation**

The primary source for benchmark instances will be **CVRPLIB** 92, which is a standard repository used in VRP research. The selection will aim for diversity in instance characteristics:

* **Instance Sets:** A variety of well-known sets will be included, such as:  
  * Augerat sets (e.g., Set P, Set A, Set B) 92  
  * Christofides and Eilon sets (Set E) 92  
  * Fisher set (Set F) 92  
  * Uchoa et al. X-instances 92: These are particularly valuable as they were generated by systematically varying attributes like depot positioning, customer distribution, and demand patterns, providing a structured way to test robustness.  
* **Instance Sizes:** A range of problem sizes will be tested:  
  * Small-scale: \~20-50 customers. These will be tractable for all three proposed methods, including initial explorations of the RL-Quantum approach on subproblems.  
  * Medium-scale: \~50-100 customers. These will challenge the scalability of all methods and are typical for many research papers.  
  * Large-scale: 100+ customers. These will primarily test the ALM and RL-Classical approaches. The RL-Quantum approach will likely only address subproblems derived from these larger instances.  
  * Quantum experiments will necessarily be on very small instances or subproblems (e.g., up to 6-10 customers for a single route subproblem, as seen in 16).  
* **Data Format and Parsers:** Standard CVRPLIB formats will be used, and robust parsers will be implemented to read instance data, including node coordinates, demands, vehicle capacities, and edge weights (or methods to calculate them if Euclidean).92

Table 4 will detail the selected instances.  
**Table 4: Selected CVRPLIB Benchmark Instances and Characteristics (Illustrative Sample)**

| Instance Set | Specific Instance Name | Num. Customers | Vehicle Capacity | Depot Position | Customer Dist. | Known Optimal/Best Solution (BKS) | Source |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Augerat P | P-n23-k8 | 22 | 3000 | (varies) | Clustered | 529 | Augerat et al. |
| Christofides E | E-n51-k5 | 50 | 160 | (varies) | Random | 521 | Christofides and Eilon \[CE69\] |
| Uchoa X | X-n101-k25 | 100 | 206 | Central | Random-Clustered | (lookup BKS) | Uchoa et al. \[UPP+17\] |
| DIMACS (Loggi) | Loggi-n401-k23 | 400 | (varies) | Real-world | Real-world | (lookup BKS) | Loggi (via CVRPLIB) |
| *...(additional instances covering various sizes and characteristics)* |  |  |  |  |  |  |  |

This systematic selection ensures that the evaluation is comprehensive, allows for comparison with existing literature, and tests the algorithms across a spectrum of problem difficulties.

### **B. Comprehensive Performance Metrics and Evaluation Protocols**

To ensure a thorough and fair comparison, a multifaceted set of performance metrics will be employed:

* **Solution Quality:**  
  * **Total Cost/Distance:** The primary objective function value for CVRP. Lower values are better.  
  * **Number of Vehicles Used:** Often a secondary objective. Some formulations minimize vehicles first, then distance. If fixed, this is a constraint.  
  * **Gap to Best Known Solution (BKS):** Calculated as $ ((SolutionCost \- BKS) / BKS) \\times 100% $. This normalizes performance across instances with different optimal values.93  
  * **Optimality Gap (for ALM):** If ALM provides dual bounds, the gap between the best feasible solution and the dual bound can indicate proximity to optimality.  
* **Computational Resources:**  
  * **Execution Time:** CPU time (for ALM, RL training, RL inference) and GPU time (if used for RL training). For quantum methods, this includes QPU access time, classical optimization time for VQA parameters, and any pre/post-processing time.90  
  * **Training Time (for RL):** Total time required to train the RL models to a satisfactory performance level.  
  * **Quantum Resources (for RL-Quantum):** Number of qubits utilized, circuit depth, number of two-qubit gates, total number of quantum measurement shots.90  
* **Scalability:** How solution quality and computational time metrics degrade or improve as the problem size (number of customers, vehicles) increases.  
* **Robustness and Generalization (especially for RL models):**  
  * Performance on unseen test instances drawn from the same distribution as training data.  
  * Performance on test instances from different distributions or with different characteristics (e.g., different customer clustering, depot locations) to assess generalization capability.  
* **Convergence Behavior:**  
  * For ALM: Number of iterations to reach convergence, evolution of primal objective and constraint violations.  
  * For RL training: Learning curves (e.g., reward per episode, loss function values over training epochs/steps).95  
* **Feasibility Rate:** The percentage of benchmark instances for which an algorithm successfully finds a feasible solution satisfying all CVRP constraints.  
* **Primal Integral (Optional):** As used in the DIMACS VRP Challenge 92, this metric combines solution quality and the time taken to achieve it into a single score, useful for ranking solvers that might find solutions of varying quality at different speeds.

Evaluation protocols will specify fixed time limits for solvers where appropriate 92, averaging results over multiple runs (especially for stochastic algorithms like RL and some metaheuristics) to account for variance, and using consistent hardware and software environments for all classical computations.

### **C. Selection of Baseline Algorithms for Comparative Analysis**

A strong set of baselines is crucial for contextualizing the performance of the three proposed advanced methodologies. These baselines will span from classical heuristics to modern state-of-the-art solvers:  
**1\. Established Classical Heuristics (for fundamental comparison):**

* **Clarke-Wright Savings Algorithm:** A widely recognized constructive heuristic for CVRP, known for its intuitive approach to merging routes based on distance savings.3  
* **Nearest Neighbor Heuristic:** A simple, greedy constructive heuristic that iteratively adds the closest unvisited customer to the current route. It is fast but often yields suboptimal solutions.62  
* **Sweep Algorithm:** A cluster-first, route-second heuristic that groups customers based on polar angle around the depot and then solves routing problems within clusters.4

These heuristics will provide a baseline for "good, quick" solutions and highlight the improvement offered by more complex methods.  
**2\. Modern State-of-the-Art (SOTA) Solvers (for benchmarking against current best practical performance):**

* **Google OR-Tools:** A comprehensive and widely used open-source software suite for optimization, providing robust solvers for VRPs that incorporate advanced heuristics and metaheuristics (e.g., guided local search, tabu search).36  
* **PyVRP:** A recently developed high-performance Python package implementing a hybrid genetic search algorithm. It has demonstrated state-of-the-art results on VRPTW and CVRP benchmarks.94  
* **OptaPlanner:** An open-source AI constraint solver that uses metaheuristics like Tabu Search and Simulated Annealing to solve VRPs and other planning problems.59  
* **LKH3 (Lin-Kernighan-Helsgaun):** While primarily a TSP solver, LKH3 is highly effective and often adapted for CVRP, representing one of the best-performing classical heuristics for routing problems.36 If feasible to integrate or obtain results from, it would serve as a very strong benchmark.

Comparing against these different tiers of baselines—from simple constructive heuristics to sophisticated SOTA solvers—will provide a rich and nuanced understanding of where the proposed ALM, RL-Classical, and RL-Quantum approaches stand in the current landscape of CVRP solution techniques. For the RL-Quantum approach, it will also be important to define a "fair" classical baseline for the specific quantum-solved subproblem. For instance, if a VQA solves a small routing subproblem for a single vehicle, its performance (solution quality and time) should be compared against an exact classical solver (e.g., a MIP solver) for that specific subproblem size and structure, in addition to comparing the overall CVRP solution quality.

### **D. Cross-Approach Performance Benchmarking Strategy**

A unified strategy will be employed for benchmarking all methods:

* **Standardized Input/Output:** All algorithms will operate on the same parsed instance data and produce solutions in a standardized format that can be processed by a common evaluation module.  
* **Controlled Environment:** All classical computations will be run on the same hardware configuration to ensure fair timing comparisons. Quantum computations will clearly state the simulator or specific QPU used, along with relevant parameters (e.g., noise model for simulators).  
* **Time Limits:** Appropriate time limits will be set for algorithms that are iterative or search-based, particularly for larger instances, to ensure practical comparability.92  
* **Multiple Runs:** For stochastic algorithms (RL, some metaheuristics), results will be averaged over multiple runs with different random seeds to report mean performance and variability (e.g., standard deviation).  
* **Statistical Analysis:** Where appropriate (e.g., comparing mean solution costs), statistical significance tests (e.g., t-tests, ANOVA) will be used to validate observed differences.  
* **Ablation Studies:** For the novel ALM, RL-Classical, and RL-Quantum approaches, ablation studies will be conducted to understand the contribution of different components. For example:  
  * ALM: Impact of different subproblem formulations or penalty update strategies.  
  * RL-Classical: Effect of different GNN layers, attention mechanisms, or the contribution of hybridized classical heuristics.  
  * RL-Quantum: Performance with vs. without RL-based VQA tuning; impact of ALiM in QUBO formulation.  
* **Qualitative Analysis:** Beyond quantitative metrics, the routes generated by different methods may be visualized and qualitatively analyzed for structural differences or patterns.

This comprehensive benchmarking strategy will enable robust conclusions about the relative merits and demerits of each proposed approach for solving the CVRP.

## **VI. Unified Python Implementation Strategy**

A well-structured, modular Python codebase is paramount for the success of this multifaceted research project. It will facilitate development, testing, experimentation, and ensure reproducibility. The implementation will be organized to allow for shared components where possible, while also accommodating the unique requirements of each of the three core solution methodologies.

### **A. Modular Project Architecture and Directory Structure**

A standardized directory structure will be adopted:

cvrp\_tripartite\_solver/  
├── data/                     \# Benchmark instances (e.g., CVRPLIB.vrp files)  
│   └── cvrplib\_instances/  
├── src/                      \# Source code  
│   ├── common/               \# Modules shared across all approaches  
│   │   ├── cvrp\_instance.py  \# Class for CVRP instance representation, parsing  
│   │   ├── solution.py       \# Class for CVRP solution representation (routes, cost)  
│   │   ├── evaluator.py      \# Evaluates solution cost, checks feasibility  
│   │   └── visualizer.py     \# Plots routes (using matplotlib)  
│   ├── alm/                  \# Augmented Lagrangian specific modules  
│   │   ├── alm\_solver.py     \# Main ALM algorithm, multiplier/penalty updates  
│   │   └── subproblem\_solvers.py \# Solvers for decomposed subproblems (e.g., ESPPRC)  
│   ├── rl\_classical/         \# Reinforcement Learning with classical methods  
│   │   ├── environment.py    \# CVRP environment (Gymnasium-like API)  
│   │   ├── networks.py       \# GNN/Transformer models (PyTorch/TensorFlow)  
│   │   ├── agent.py          \# RL agent (PPO, A2C, etc.)  
│   │   ├── replay\_buffer.py  \# For off-policy algorithms (if used)  
│   │   └── classical\_heuristics\_interface.py \# Wrappers for local search, etc.  
│   └── rl\_quantum/           \# Reinforcement Learning with quantum methods  
│       ├── qubo\_formulator.py \# Maps CVRP subproblems to QUBO (with ALiM)  
│       ├── quantum\_circuits.py \# QAOA/VQE ansatz definitions (Qiskit/Pennylane)  
│       ├── quantum\_environment.py \# Interface for quantum execution, RL for VQA tuning  
│       └── quantum\_rl\_agent.py  \# RL agent for VQA parameter/ansatz optimization  
├── experiments/              \# Scripts for running experiments  
│   ├── configs/              \# Configuration files for experiments (YAML or JSON)  
│   ├── run\_alm\_experiment.py  
│   ├── run\_rl\_classical\_experiment.py  
│   ├── run\_rl\_quantum\_experiment.py  
│   └── run\_baselines\_experiment.py  
├── notebooks/                \# Jupyter notebooks for analysis, visualization, prototyping  
├── results/                  \# Storage for raw and processed experimental results  
│   ├── alm/  
│   ├── rl\_classical/  
│   ├── rl\_quantum/  
│   └── baselines/  
├── tests/                    \# Unit and integration tests  
│   ├── common/  
│   ├── alm/  
│   ├── rl\_classical/  
│   └── rl\_quantum/  
├── requirements.txt          \# Python dependencies  
└── README.md                 \# Project overview and setup instructions

This structure promotes separation of concerns and allows for focused development on each component.

### **B. Core Python Modules and Their Functionalities**

The functionalities of key modules outlined above are critical:

* **src/common/cvrp\_instance.py**: Will handle parsing of CVRPLIB files 92, storing customer coordinates, demands, vehicle capacity, depot location, and the distance matrix.  
* **src/common/solution.py**: Will define a standardized way to represent a CVRP solution, typically as a list of routes, where each route is a sequence of customer IDs. It will also store the total cost and number of vehicles used.  
* **src/common/evaluator.py**: This module is vital for fair comparison. It will take a Solution object and an Instance object, verify feasibility (all customers visited once, capacity constraints met for each route, routes start/end at depot), and calculate the objective function value (total distance). All three proposed methods and baselines must produce solutions compatible with this evaluator.  
* **src/alm/alm\_solver.py**: Implements the iterative ALM logic, including subproblem calls, Lagrange multiplier updates (e.g., $ \\lambda\_i^{k+1} \= \\lambda\_i^k \+ \\rho\_k g\_i(x^k) $ 19), and penalty parameter adjustments based on predefined strategies.29  
* **src/rl\_classical/environment.py**: Defines the CVRP as an RL environment. The state will include static node data and dynamic information like current vehicle load and visited customer mask.32 Actions will be selecting the next customer, with masking for infeasible choices.36 Rewards will typically be sparse (e.g., negative total distance at episode end).32  
* **src/rl\_classical/networks.py**: Contains PyTorch or TensorFlow implementations of GNN/Transformer encoders and attention-based decoders.36  
* **src/rl\_quantum/qubo\_formulator.py**: Implements the logic to convert a CVRP subproblem (e.g., a single vehicle route generation task) into a QUBO formulation. This will crucially include the ALiM technique to handle capacity constraints efficiently by adding linear and quadratic penalty terms without slack variables, thus reducing qubit requirements.16  
* **src/rl\_quantum/quantum\_circuits.py**: Defines QAOA or VQE ansatz circuits using Qiskit 84 or Pennylane.82 These circuits will be parameterized, with parameters to be optimized by a classical optimizer or an RL agent.  
* **src/rl\_quantum/quantum\_environment.py**: This will be a complex module acting as the bridge between a classical RL agent and the quantum execution backend (simulator or actual QPU). It will take actions from the RL agent (e.g., new VQA parameters), execute the quantum circuit, and return a state representation and reward (e.g., based on measured energy or subproblem solution quality).

Clear APIs between these modules will be enforced to ensure maintainability and allow for independent testing and development.

### **C. Essential Libraries and Environment Configuration (requirements.txt)**

The project will rely on a combination of standard Python libraries and specialized packages for optimization, machine learning, and quantum computing.  
**Table 5: Key Python Libraries and Their Roles**

| Library Category | Library Name | Primary Use Case in this Project | Relevant Information Sources |
| :---- | :---- | :---- | :---- |
| Core Numerics & Utilities | python | Base language (version \>= 3.8 recommended) | General |
|  | numpy | Fundamental package for numerical computation (arrays, matrices) | General |
|  | scipy | Scientific computing, optimization routines (potentially for ALM subproblems or classical optimizers for VQAs) | 20 |
|  | pandas | Data manipulation and analysis (e.g., for results) | 62 |
| Optimization (Classical) | gurobipy / cplex | Python APIs for Gurobi/CPLEX MIP solvers (for ALM subproblems if formulated as MIPs, or as high-quality baselines) |  |
|  | ortools | Google OR-Tools: for baseline CVRP solver, routing algorithms | 36 |
| Machine Learning (RL-Classical & RL for Quantum Control) | pytorch or tensorflow | Deep learning frameworks for defining and training neural networks (GNNs, Transformers, RL agents) | 47 |
|  | torch\_geometric / dgl | Libraries for implementing Graph Neural Networks (if PyTorch/TensorFlow respectively) |  |
|  | gymnasium | Toolkit for developing and comparing RL environments (for cvrp\_environment.py) |  |
| Quantum Computing (RL-Quantum) | qiskit | IBM's quantum computing SDK for circuit design, simulation, and execution on IBM hardware/simulators | 84 |
|  | pennylane | Quantum machine learning library supporting various backends, good for VQAs and integration with PyTorch/TensorFlow | 82 |
| VRP Utilities | vrplib | For parsing CVRPLIB benchmark instances |  |
| Plotting & Visualization | matplotlib | For plotting routes, training curves, and other results | 62 |
|  | seaborn | Statistical data visualization, enhancing matplotlib plots |  |
| Development & Testing | jupyter | For interactive development, prototyping, and results analysis |  |
|  | pytest | Framework for writing and running unit/integration tests |  |

A requirements.txt file will precisely list all dependencies and their versions to ensure a reproducible environment. The choice between PyTorch and TensorFlow for RL components will be made based on team familiarity and library support for specific architectures (e.g., GNNs, Transformers).  
The interfacing between the classical RL agent and the quantum environment in the rl\_quantum part will be a particularly challenging yet critical piece of engineering. This module must abstract away the complexities of quantum execution (which can be noisy and probabilistic) and provide a stable interface for the RL agent to learn how to control or optimize the quantum VQA parameters or ansatz structure.

## **VII. Phased Research and Development Timeline**

A phased approach is proposed to manage the complexity of this research and ensure incremental progress and validation.

* **Phase 1: Foundational Work & Literature Synthesis (Months 1-3)**  
  * **Tasks:** Conduct an exhaustive literature review covering CVRP, ALM for CO, RL for VRP (GNNs, Transformers, PG, DQN), QUBO formulations for CO, VQAs (QAOA, VQE), ALiM, and RL for quantum control. Finalize the precise mathematical formulation of CVRP to be used. Establish the core Python project structure, including common modules like cvrp\_instance.py, solution.py, and the crucial evaluator.py. Acquire, parse, and preprocess all selected CVRPLIB benchmark datasets.  
  * **Deliverables:** Comprehensive literature review document; finalized CVRP mathematical model; initial Python project structure with common modules implemented and unit-tested; curated benchmark dataset suite.  
* **Phase 2: Approach 1 \- Augmented Lagrangian Development & Validation (Months 3-6)**  
  * **Tasks:** Implement the chosen ALM framework for CVRP, including the specific decomposition strategy (e.g., customer assignment relaxation), the subproblem solver (e.g., ESPPRC solver), and the multiplier/penalty update schemes. Perform initial tuning of ALM parameters on a small subset of benchmark instances. Conduct preliminary benchmarking of the ALM approach against selected classical heuristics (e.g., Clarke-Wright, Nearest Neighbor) on small to medium-sized instances.  
  * **Deliverables:** Working ALM solver for CVRP; report on ALM parameter tuning experiments; preliminary performance results of ALM vs. basic heuristics.  
* **Phase 3: Approach 2 \- RL-Classical Development & Validation (Months 5-9)**  
  * **Tasks:** Develop the CVRP environment.py compatible with gymnasium. Implement the chosen GNN/Transformer neural network architecture (encoder-decoder) and the RL agent (e.g., PPO). Train the RL model on small and medium-sized CVRP instances, focusing on achieving stable learning and reasonable solution quality. Experiment with hybridization strategies, such as integrating a local search module to refine RL-generated solutions. Benchmark the RL-Classical approach against the ALM solver and classical baselines.  
  * **Deliverables:** Trained RL model for CVRP; CVRP RL environment; report on RL training performance and hyperparameter tuning; comparative benchmark results against ALM and classical heuristics.  
* **Phase 4: Approach 3 \- RL-Quantum Development & Validation (Months 8-14)**  
  * **Tasks:** Develop the qubo\_formulator.py module to convert CVRP subproblems (e.g., single vehicle route generation from CG) into QUBOs, incorporating the ALiM technique for capacity constraints.16 Implement QAOA/VQE circuits for these QUBOs using Qiskit or Pennylane. Develop the RL agent (quantum\_rl\_agent.py) and quantum\_environment.py for tuning VQA parameters or ansatz structures. Conduct initial tests on very small CVRP subproblems using quantum simulators, focusing on the feasibility of the RL-for-quantum-control loop. If simulator results are promising and resources allow, perform exploratory tests on real quantum hardware via cloud platforms for selected small-scale subproblems.  
  * **Deliverables:** QUBO formulation module for CVRP subproblems with ALiM; parameterized QAOA/VQE circuits; RL agent for VQA optimization; simulation results for RL-Quantum on small subproblems; (optional) preliminary results from real quantum hardware.  
* **Phase 5: Integrated Benchmarking, Analysis & Manuscript Preparation (Months 13-18)**  
  * **Tasks:** Conduct final, comprehensive benchmarking runs comparing all three developed approaches (ALM, RL-Classical, RL-Quantum for subproblems if applicable) against the full suite of SOTA classical solvers (OR-Tools, PyVRP, etc.) on all selected CVRPLIB instances. Perform thorough statistical analysis of all collected performance data. Generate plots, tables, and visualizations to clearly present the findings. Write the research manuscript, detailing the methodologies, experimental setup, results, and a comprehensive discussion of the findings, including limitations and implications. Iterate on the manuscript based on internal reviews and prepare for submission to a high-impact academic journal or conference.  
  * **Deliverables:** Complete set of benchmark results and analysis; final research manuscript.

This timeline is ambitious and assumes dedicated resources. Flexibility will be needed, especially for the RL-Quantum phase, which is highly dependent on the rapidly evolving state of quantum technology and algorithms.

## **VIII. Concluding Remarks and Avenues for Future Inquiry**

This research roadmap outlines a comprehensive investigation into three advanced and diverse solution methodologies for the Capacitated Vehicle Routing Problem: Augmented Lagrangian methods, Reinforcement Learning with classical integration, and Reinforcement Learning with quantum integration. The successful execution of this plan is anticipated to yield significant contributions to the field of combinatorial optimization, particularly by providing a rigorous comparative analysis of these cutting-edge techniques applied to a canonical NP-hard problem.  
The expected outcomes include:

1. Novel or refined algorithmic frameworks for each of the three approaches, tailored to the specifics of CVRP.  
2. A deep understanding of the relative strengths, weaknesses, scalability, and practical applicability of ALM, RL-Classical, and RL-Quantum paradigms for CVRP.  
3. Pioneering insights into the use of RL for managing and optimizing quantum algorithms (QAOA/VQE) in the context of a complex logistics problem, including the critical role of methods like ALiM in making quantum subproblems tractable.  
4. A high-quality, publishable research paper that bridges operations research, artificial intelligence, and quantum computing.

Despite the comprehensive nature of this plan, certain limitations are inherent. The study will focus primarily on the standard, static CVRP to ensure a clear comparative basis. The specific choices of RL algorithms (e.g., PPO) and VQAs (e.g., QAOA) represent a subset of possible options. Furthermore, the RL-Quantum component's success is heavily contingent on the progress of NISQ-era quantum hardware and simulation capabilities.  
This research opens several exciting avenues for future inquiry:

* **Extension to Richer VRP Variants:** Applying the developed frameworks and comparative methodology to more complex VRPs, such as those with time windows (VRPTW), multiple depots (MDVRP), stochastic demands, or dynamic requests (DVRP).  
* **Advanced Quantum Algorithms and Formulations:** Exploring alternative quantum algorithms beyond QAOA/VQE, or investigating more sophisticated QUBO embedding techniques and constraint handling methods for CVRP subproblems. This could also involve exploring quantum algorithms that do not rely on QUBO formulations.  
* **Multi-Agent Reinforcement Learning (MARL):** Investigating MARL systems where each vehicle or a central dispatcher acts as an independent or cooperative agent, potentially offering better scalability and coordination for large fleets.  
* **Transfer Learning and Generalization in RL for VRP:** Developing more robust transfer learning techniques to improve the generalization of RL models across different CVRP instance sizes, distributions, and constraints, reducing the need for extensive retraining.  
* **Scalability of Quantum Approaches:** As quantum hardware matures, revisiting the RL-Quantum approach for larger CVRP subproblems or even end-to-end quantum solutions, and benchmarking against continually improving classical SOTA.  
* **Real-World Deployment Challenges:** Addressing the practical challenges of deploying these advanced optimization techniques in real-world logistics operations, including data integration, real-time decision-making pressures, and user trust.

In conclusion, the proposed tripartite investigation promises to advance the understanding and solution capabilities for CVRP, offering valuable insights for researchers and practitioners across multiple scientific and engineering disciplines. The journey will undoubtedly be challenging, particularly at the intersection of RL and quantum computing, but the potential for groundbreaking discoveries and impactful solutions is substantial.

#### **Works cited**

1. (PDF) Two models of the capacitated vehicle routing problem, accessed May 14, 2025, [https://www.researchgate.net/publication/323173028\_Two\_models\_of\_the\_capacitated\_vehicle\_routing\_problem](https://www.researchgate.net/publication/323173028_Two_models_of_the_capacitated_vehicle_routing_problem)  
2. (PDF) CAPACITATED VEHICLE ROUTING PROBLEM \- ResearchGate, accessed May 14, 2025, [https://www.researchgate.net/publication/343777001\_CAPACITATED\_VEHICLE\_ROUTING\_PROBLEM](https://www.researchgate.net/publication/343777001_CAPACITATED_VEHICLE_ROUTING_PROBLEM)  
3. Implementation of Clark and Wright Savings Algorithm in Generating Initial Solution for Solid Waste Collection \- Science UTM, accessed May 14, 2025, [https://science.utm.my/procscimath/wp-content/uploads/sites/605/2022/10/101-109-Khoo-Xu-Yi\_Farhana-Johar.pdf](https://science.utm.my/procscimath/wp-content/uploads/sites/605/2022/10/101-109-Khoo-Xu-Yi_Farhana-Johar.pdf)  
4. Analysis of The Use of Sweep Algorithms to Solve Capacitated Vehicle Routing Problems, accessed May 14, 2025, [https://siakad.univamedan.ac.id/ojs/index.php/JMPM/article/download/416/339/1372](https://siakad.univamedan.ac.id/ojs/index.php/JMPM/article/download/416/339/1372)  
5. Capacitated Vehicle Routing Problem (CVRP): Complete Guide 2025, accessed May 14, 2025, [https://www.upperinc.com/glossary/route-optimization/capacitated-vehicle-routing-problem-cvrp/](https://www.upperinc.com/glossary/route-optimization/capacitated-vehicle-routing-problem-cvrp/)  
6. A Survey on the Vehicle Routing Problem and Its Variants, accessed May 14, 2025, [https://www.scirp.org/journal/paperinformation?paperid=19355](https://www.scirp.org/journal/paperinformation?paperid=19355)  
7. (PDF) A Survey on the Vehicle Routing Problem and Its Variants \- ResearchGate, accessed May 14, 2025, [https://www.researchgate.net/publication/267202931\_A\_Survey\_on\_the\_Vehicle\_Routing\_Problem\_and\_Its\_Variants](https://www.researchgate.net/publication/267202931_A_Survey_on_the_Vehicle_Routing_Problem_and_Its_Variants)  
8. Reinforcement Learning for Solving the Vehicle Routing Problem \- Lehigh University, accessed May 14, 2025, [https://engineering.lehigh.edu/sites/engineering.lehigh.edu/files/\_DEPARTMENTS/ise/pdf/tech-papers/19/19T\_002.pdf](https://engineering.lehigh.edu/sites/engineering.lehigh.edu/files/_DEPARTMENTS/ise/pdf/tech-papers/19/19T_002.pdf)  
9. Enhanced vehicle routing for medical waste management via hybrid deep reinforcement learning and optimization algorithms \- Frontiers, accessed May 14, 2025, [https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1496653/pdf](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1496653/pdf)  
10. Solving Capacitated Vehicle Routing Problem with Quantum Alternating Operator Ansatz and Column Generation \- arXiv, accessed May 14, 2025, [https://arxiv.org/pdf/2503.17051](https://arxiv.org/pdf/2503.17051)  
11. A Hybrid Solution Method for the Capacitated Vehicle Routing Problem Using a Quantum Annealer \- Frontiers, accessed May 14, 2025, [https://www.frontiersin.org/journals/ict/articles/10.3389/fict.2019.00013/full](https://www.frontiersin.org/journals/ict/articles/10.3389/fict.2019.00013/full)  
12. arXiv:2503.10421v1 \[cs.LG\] 13 Mar 2025, accessed May 14, 2025, [https://arxiv.org/pdf/2503.10421](https://arxiv.org/pdf/2503.10421)  
13. Review of research on vehicle routing problems \- SPIE Digital Library, accessed May 14, 2025, [https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13018/3024185/Review-of-research-on-vehicle-routing-problems/10.1117/12.3024185.full](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13018/3024185/Review-of-research-on-vehicle-routing-problems/10.1117/12.3024185.full)  
14. A Review of the Vehicle Routing Problem and the Current Routing Services in Smart Cities, accessed May 14, 2025, [https://www.mdpi.com/2813-2203/2/1/1](https://www.mdpi.com/2813-2203/2/1/1)  
15. arxiv.org, accessed May 14, 2025, [https://arxiv.org/pdf/2303.04147](https://arxiv.org/pdf/2303.04147)  
16. Hybrid Quantum-Classical Algorithm for Solving Capacitated ..., accessed May 14, 2025, [https://www.computer.org/csdl/proceedings-article/qce/2024/413702a488/23oqjIMX3iM](https://www.computer.org/csdl/proceedings-article/qce/2024/413702a488/23oqjIMX3iM)  
17. Augmented Lagrangian method \- Wikipedia, accessed May 14, 2025, [https://en.wikipedia.org/wiki/Augmented\_Lagrangian\_method](https://en.wikipedia.org/wiki/Augmented_Lagrangian_method)  
18. Augmented Lagrangian Methods \- NEOS Guide, accessed May 14, 2025, [https://neos-guide.org/guide/algorithms/augmented-lagrangian/](https://neos-guide.org/guide/algorithms/augmented-lagrangian/)  
19. Augmented Lagrangian methods | Mathematical Methods for Optimization Class Notes, accessed May 14, 2025, [https://library.fiveable.me/mathematical-methods-for-optimization/unit-14/augmented-lagrangian-methods/study-guide/sg1sGKsoZQBZWkPQ](https://library.fiveable.me/mathematical-methods-for-optimization/unit-14/augmented-lagrangian-methods/study-guide/sg1sGKsoZQBZWkPQ)  
20. proceedings.neurips.cc, accessed May 14, 2025, [https://proceedings.neurips.cc/paper\_files/paper/2024/file/480eb35745feb11c9120b666f640893e-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/480eb35745feb11c9120b666f640893e-Paper-Conference.pdf)  
21. A Bundle-based Augmented Lagrangian Framework: Algorithm, Convergence, and Primal-dual Principles This work is supported by NSF ECCS 2154650, NSF CMMI 2320697, and NSF CAREER 2340713\. Emails: fliao@ucsd.edu \- arXiv, accessed May 14, 2025, [https://arxiv.org/html/2502.08835v1](https://arxiv.org/html/2502.08835v1)  
22. An Augmented Lagrangian Decomposition Method for Chance-Constrained Optimization Problems, accessed May 14, 2025, [https://espace.curtin.edu.au/bitstream/handle/20.500.11937/91428/91252.pdf;jsessionid=336D0712CDE0A708FFEBCE14C4718D1A?sequence=3](https://espace.curtin.edu.au/bitstream/handle/20.500.11937/91428/91252.pdf;jsessionid=336D0712CDE0A708FFEBCE14C4718D1A?sequence=3)  
23. Question on solving an optimization problem using Variable splitting and ADMM, accessed May 14, 2025, [https://mathoverflow.net/questions/228444/question-on-solving-an-optimization-problem-using-variable-splitting-and-admm](https://mathoverflow.net/questions/228444/question-on-solving-an-optimization-problem-using-variable-splitting-and-admm)  
24. Solving the multi-compartment vehicle routing problem by an augmented Lagrangian relaxation method \- ResearchGate, accessed May 14, 2025, [https://www.researchgate.net/publication/373816443\_Solving\_the\_multi-compartment\_vehicle\_routing\_problem\_by\_an\_augmented\_Lagrangian\_relaxation\_method](https://www.researchgate.net/publication/373816443_Solving_the_multi-compartment_vehicle_routing_problem_by_an_augmented_Lagrangian_relaxation_method)  
25. An augmented Lagrangian relaxation method for the mean-standard deviation based vehicle routing problem \- ResearchGate, accessed May 14, 2025, [https://www.researchgate.net/publication/359837466\_An\_augmented\_Lagrangian\_relaxation\_method\_for\_the\_mean-standard\_deviation\_based\_vehicle\_routing\_problem](https://www.researchgate.net/publication/359837466_An_augmented_Lagrangian_relaxation_method_for_the_mean-standard_deviation_based_vehicle_routing_problem)  
26. Coordinating assignment and routing decisions in transit vehicle schedules: A variable-splitting Lagrangian decomposition approach for solution symmetry breaking | Request PDF \- ResearchGate, accessed May 14, 2025, [https://www.researchgate.net/publication/321744338\_Coordinating\_assignment\_and\_routing\_decisions\_in\_transit\_vehicle\_schedules\_A\_variable-splitting\_Lagrangian\_decomposition\_approach\_for\_solution\_symmetry\_breaking](https://www.researchgate.net/publication/321744338_Coordinating_assignment_and_routing_decisions_in_transit_vehicle_schedules_A_variable-splitting_Lagrangian_decomposition_approach_for_solution_symmetry_breaking)  
27. Augmented Lagrangian method \- Wikipedia, accessed May 14, 2025, [https://en.wikipedia.org/wiki/Augmented\_Lagrangian\_method\#Alternating\_direction\_method\_of\_multipliers](https://en.wikipedia.org/wiki/Augmented_Lagrangian_method#Alternating_direction_method_of_multipliers)  
28. How to understand what's going wrong in a code for solving a problem with augmented lagrangian?, accessed May 14, 2025, [https://scicomp.stackexchange.com/questions/23235/how-to-understand-whats-going-wrong-in-a-code-for-solving-a-problem-with-augmen](https://scicomp.stackexchange.com/questions/23235/how-to-understand-whats-going-wrong-in-a-code-for-solving-a-problem-with-augmen)  
29. A new penalty parameter update rule in the augmented lagrange multiplier method for dynamic response optimization \- ResearchGate, accessed May 14, 2025, [https://www.researchgate.net/publication/251214940\_A\_new\_penalty\_parameter\_update\_rule\_in\_the\_augmented\_lagrange\_multiplier\_method\_for\_dynamic\_response\_optimization](https://www.researchgate.net/publication/251214940_A_new_penalty_parameter_update_rule_in_the_augmented_lagrange_multiplier_method_for_dynamic_response_optimization)  
30. A New Penalty Parameter Update Rule in the Augmented Lagrange Multiplier Method for Dynamic Response Optimization \- Korea Science, accessed May 14, 2025, [https://www.koreascience.kr/article/JAKO200027542999901.page](https://www.koreascience.kr/article/JAKO200027542999901.page)  
31. Augmented Lagrangian methods with smooth penalty functions, accessed May 14, 2025, [https://www.math.univ-toulouse.fr/\~noll/PAPERS/alclassic\_rev5.pdf](https://www.math.univ-toulouse.fr/~noll/PAPERS/alclassic_rev5.pdf)  
32. (PDF) Vehicle Routing Problem Solving Using Reinforcement ..., accessed May 14, 2025, [https://www.researchgate.net/publication/378538215\_Vehicle\_Routing\_Problem\_Solving\_Using\_Reinforcement\_Learning](https://www.researchgate.net/publication/378538215_Vehicle_Routing_Problem_Solving_Using_Reinforcement_Learning)  
33. Solving the Capacitated Vehicle Routing Problem via Reinforcment Learning \- ScienceOpen, accessed May 14, 2025, [https://www.scienceopen.com/hosted-document?doi=10.14293/P2199-8442.1.SOP-.PVJ7PQ.v1](https://www.scienceopen.com/hosted-document?doi=10.14293/P2199-8442.1.SOP-.PVJ7PQ.v1)  
34. \[1802.04240\] Reinforcement Learning for Solving the Vehicle Routing Problem \- arXiv, accessed May 14, 2025, [https://arxiv.org/abs/1802.04240](https://arxiv.org/abs/1802.04240)  
35. Learning Vehicle Routing Problems using Policy Optimisation \- arXiv, accessed May 14, 2025, [https://arxiv.org/pdf/2012.13269](https://arxiv.org/pdf/2012.13269)  
36. RP-DQN: An application of Q-Learning to Vehicle Routing Problems \- ResearchGate, accessed May 14, 2025, [https://www.researchgate.net/publication/351104703\_RP-DQN\_An\_application\_of\_Q-Learning\_to\_Vehicle\_Routing\_Problems](https://www.researchgate.net/publication/351104703_RP-DQN_An_application_of_Q-Learning_to_Vehicle_Routing_Problems)  
37. Edge-Driven Multiple Trajectory Attention Model for Vehicle Routing Problems \- MDPI, accessed May 14, 2025, [https://www.mdpi.com/2076-3417/15/5/2679](https://www.mdpi.com/2076-3417/15/5/2679)  
38. Graph Transformer with Reinforcement Learning for Vehicle Routing ..., accessed May 14, 2025, [https://www.researchgate.net/publication/368496592\_Graph\_Transformer\_with\_Reinforcement\_Learning\_for\_Vehicle\_Routing\_Problem](https://www.researchgate.net/publication/368496592_Graph_Transformer_with_Reinforcement_Learning_for_Vehicle_Routing_Problem)  
39. Machine learning approach to solving the capacitated vehicle routing problem | Request PDF \- ResearchGate, accessed May 14, 2025, [https://www.researchgate.net/publication/378321792\_Machine\_learning\_approach\_to\_solving\_the\_capacitated\_vehicle\_routing\_problem](https://www.researchgate.net/publication/378321792_Machine_learning_approach_to_solving_the_capacitated_vehicle_routing_problem)  
40. Deep Q Network vs Policy Gradients \- An Experiment on VizDoom with Keras | Felix Yu, accessed May 14, 2025, [https://flyyufelix.github.io/2017/10/12/dqn-vs-pg.html](https://flyyufelix.github.io/2017/10/12/dqn-vs-pg.html)  
41. The advantages and disadvantages of policy-gradient methods \- Hugging Face Deep RL Course, accessed May 14, 2025, [https://huggingface.co/learn/deep-rl-course/unit4/advantages-disadvantages](https://huggingface.co/learn/deep-rl-course/unit4/advantages-disadvantages)  
42. Policy Gradient Methods in Reinforcement Learning | GeeksforGeeks, accessed May 14, 2025, [https://www.geeksforgeeks.org/policy-gradient-methods-in-reinforcement-learning/](https://www.geeksforgeeks.org/policy-gradient-methods-in-reinforcement-learning/)  
43. Advantage Actor Critic (A2C) \- Hugging Face, accessed May 14, 2025, [https://huggingface.co/blog/deep-rl-a2c](https://huggingface.co/blog/deep-rl-a2c)  
44. Advantage Actor-Critic (A2C) \- Schneppat AI, accessed May 14, 2025, [https://schneppat.com/advantage-actor-critic\_a2c.html](https://schneppat.com/advantage-actor-critic_a2c.html)  
45. The idea behind Actor-Critics and how A2C and A3C improve them \- AI Summer, accessed May 14, 2025, [https://theaisummer.com/Actor\_critics/](https://theaisummer.com/Actor_critics/)  
46. Comparing A2C and Q-learning algorithms : r/reinforcementlearning \- Reddit, accessed May 14, 2025, [https://www.reddit.com/r/reinforcementlearning/comments/154ofvl/comparing\_a2c\_and\_qlearning\_algorithms/](https://www.reddit.com/r/reinforcementlearning/comments/154ofvl/comparing_a2c_and_qlearning_algorithms/)  
47. Evaluating DQN, Vehicle Routing Problem (VRP) \- Stack Overflow, accessed May 14, 2025, [https://stackoverflow.com/questions/75164008/evaluating-dqn-vehicle-routing-problem-vrp](https://stackoverflow.com/questions/75164008/evaluating-dqn-vehicle-routing-problem-vrp)  
48. Enhanced vehicle routing for medical waste management via hybrid deep reinforcement learning and optimization algorithms \- Frontiers, accessed May 14, 2025, [https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1496653/full](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1496653/full)  
49. A Large Language Model-Enhanced Q-learning for Capacitated Vehicle Routing Problem with Time Windows \- arXiv, accessed May 14, 2025, [https://www.arxiv.org/pdf/2505.06178](https://www.arxiv.org/pdf/2505.06178)  
50. Deep Q-Learning (DQN): A Comprehensive Guide \- BytePlus, accessed May 14, 2025, [https://www.byteplus.com/en/topic/514170](https://www.byteplus.com/en/topic/514170)  
51. Deep Reinforcement Learning for Dynamic Capacitated Vehicle ..., accessed May 14, 2025, [https://openreview.net/forum?id=Gs8jWk0F01¬eId=4JLV3Clkml](https://openreview.net/forum?id=Gs8jWk0F01&noteId=4JLV3Clkml)  
52. Deep Reinforcement Learning for Multi-Truck Vehicle Routing Problems with Multi-Leg Demand Routes \- arXiv, accessed May 14, 2025, [https://arxiv.org/html/2401.08669v1](https://arxiv.org/html/2401.08669v1)  
53. A Scalable Graph Learning Approach to Capacitated Vehicle Routing Problem Using Capsule Networks and Attention Mechanism | IDETC-CIE | ASME Digital Collection, accessed May 14, 2025, [https://asmedigitalcollection.asme.org/IDETC-CIE/proceedings/IDETC-CIE2022/86236/V03BT03A045/1150455](https://asmedigitalcollection.asme.org/IDETC-CIE/proceedings/IDETC-CIE2022/86236/V03BT03A045/1150455)  
54. A Scalable Graph Learning Approach to Capacitated Vehicle Routing Problem Using Capsule Networks and Attention Mechanism (Conference Paper) \- NSF Public Access Repository, accessed May 14, 2025, [https://par.nsf.gov/biblio/10345362-scalable-graph-learning-approach-capacitated-vehicle-routing-problem-using-capsule-networks-attention-mechanism](https://par.nsf.gov/biblio/10345362-scalable-graph-learning-approach-capacitated-vehicle-routing-problem-using-capsule-networks-attention-mechanism)  
55. Application of Reinforcement Learning Methods Combining Graph Neural Networks and Self-Attention Mechanisms in Supply Chain Route Optimization \- MDPI, accessed May 14, 2025, [https://www.mdpi.com/1424-8220/25/3/955](https://www.mdpi.com/1424-8220/25/3/955)  
56. A Deep Reinforcement Learning-Based Decision-Making Approach ..., accessed May 14, 2025, [https://www.mdpi.com/2076-3417/15/9/4951](https://www.mdpi.com/2076-3417/15/9/4951)  
57. Neural Combinatorial Optimization Algorithms for Solving Vehicle Routing Problems: A Comprehensive Survey with Perspectives \- arXiv, accessed May 14, 2025, [https://arxiv.org/html/2406.00415v3](https://arxiv.org/html/2406.00415v3)  
58. Reinforcement Learning for Pricing Problem Optimization in VRP. \- arXiv, accessed May 14, 2025, [https://arxiv.org/html/2504.02383v1](https://arxiv.org/html/2504.02383v1)  
59. Vehicle Routing Problem \- OptaPlanner, accessed May 14, 2025, [https://www.optaplanner.org/learn/useCases/vehicleRoutingProblem.html](https://www.optaplanner.org/learn/useCases/vehicleRoutingProblem.html)  
60. (PDF) SED2AM: Solving Multi-Trip Time-Dependent Vehicle Routing Problem using Deep Reinforcement Learning \- ResearchGate, accessed May 14, 2025, [https://www.researchgate.net/publication/389648674\_SED2AM\_Solving\_Multi-Trip\_Time-Dependent\_Vehicle\_Routing\_Problem\_using\_Deep\_Reinforcement\_Learning](https://www.researchgate.net/publication/389648674_SED2AM_Solving_Multi-Trip_Time-Dependent_Vehicle_Routing_Problem_using_Deep_Reinforcement_Learning)  
61. 10 Pros and Cons of Reinforcement Learning \[2025\] \- DigitalDefynd, accessed May 14, 2025, [https://digitaldefynd.com/IQ/reinforcement-learning-pros-cons/](https://digitaldefynd.com/IQ/reinforcement-learning-pros-cons/)  
62. ngchunlong279645/Capacitated-Vehicle-Routing-Problem-CVRP \- GitHub, accessed May 14, 2025, [https://github.com/ngchunlong279645/Capacitated-Vehicle-Routing-Problem-CVRP-](https://github.com/ngchunlong279645/Capacitated-Vehicle-Routing-Problem-CVRP-)  
63. A Heuristic Approach Based on Clarke-Wright Algorithm for Open ..., accessed May 14, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC3870871/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3870871/)  
64. What are the Reinforcement Learning Advantages and Disadvantages, accessed May 14, 2025, [https://www.birchwoodu.org/reinforcement-learning-advantages-and-disadvantages/](https://www.birchwoodu.org/reinforcement-learning-advantages-and-disadvantages/)  
65. Reinforcement Learning for Dynamic Vehicle Routing Problem \- International Journal of Communication Networks and Information Security (IJCNIS), accessed May 14, 2025, [https://www.ijcnis.org/index.php/ijcnis/article/view/7013/1521](https://www.ijcnis.org/index.php/ijcnis/article/view/7013/1521)  
66. Flexible Capacitated Vehicle Routing Problem Solution Method Based on Memory Pointer Network \- MDPI, accessed May 14, 2025, [https://www.mdpi.com/2227-7390/13/7/1061](https://www.mdpi.com/2227-7390/13/7/1061)  
67. Review for NeurIPS paper: Reinforcement Learning with ..., accessed May 14, 2025, [https://papers.nips.cc/paper\_files/paper/2020/file/06a9d51e04213572ef0720dd27a84792-MetaReview.html](https://papers.nips.cc/paper_files/paper/2020/file/06a9d51e04213572ef0720dd27a84792-MetaReview.html)  
68. Capacitated Vehicle Routing Problem (CVRP) \- The Quantum Computing Cloud, accessed May 14, 2025, [https://amplify.fixstars.com/en/demo/cvrp](https://amplify.fixstars.com/en/demo/cvrp)  
69. Enhancing a QUBO solver via data driven multi-start and its application to vehicle routing problem, accessed May 14, 2025, [https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=8627\&context=sis\_research](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=8627&context=sis_research)  
70. Runtime breakdown of the D-Wave QPU when solving the QUBO problem... \- ResearchGate, accessed May 14, 2025, [https://www.researchgate.net/figure/Runtime-breakdown-of-the-D-Wave-QPU-when-solving-the-QUBO-problem-according-to-the-QAL-BP\_fig6\_378657723](https://www.researchgate.net/figure/Runtime-breakdown-of-the-D-Wave-QPU-when-solving-the-QUBO-problem-according-to-the-QAL-BP_fig6_378657723)  
71. Wei-Hao Huang's research works | National Cheng Kung University and other places, accessed May 14, 2025, [https://www.researchgate.net/scientific-contributions/Wei-Hao-Huang-2196192765](https://www.researchgate.net/scientific-contributions/Wei-Hao-Huang-2196192765)  
72. Solving Capacitated Vehicle Routing Problem with Quantum Alternating Operator Ansatz and Column Generation \- arXiv, accessed May 14, 2025, [https://www.arxiv.org/pdf/2503.17051](https://www.arxiv.org/pdf/2503.17051)  
73. \[2503.17051\] Solving Capacitated Vehicle Routing Problem with Quantum Alternating Operator Ansatz and Column Generation \- arXiv, accessed May 14, 2025, [https://arxiv.org/abs/2503.17051](https://arxiv.org/abs/2503.17051)  
74. Hybrid Quantum-Classical Algorithm for Solving Capacitated Vehicle Routing Problems, accessed May 14, 2025, [https://www.researchgate.net/publication/387920465\_Hybrid\_Quantum-Classical\_Algorithm\_for\_Solving\_Capacitated\_Vehicle\_Routing\_Problems](https://www.researchgate.net/publication/387920465_Hybrid_Quantum-Classical_Algorithm_for_Solving_Capacitated_Vehicle_Routing_Problems)  
75. Reinforcement Learning Assisted Recursive QAOA \- arXiv, accessed May 14, 2025, [https://arxiv.org/html/2207.06294v2](https://arxiv.org/html/2207.06294v2)  
76. Reinforcement learning assisted recursive QAOA \- PMC \- PubMed Central, accessed May 14, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10794381/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10794381/)  
77. Quantum-Assisted Vehicle Routing: Realizing QAOA-based Approach on Gate-Based Quantum Computer \- arXiv, accessed May 14, 2025, [https://arxiv.org/html/2505.01614v1](https://arxiv.org/html/2505.01614v1)  
78. Quantum-Assisted Vehicle Routing: Realizing QAOA-based Approach on Gate-Based Quantum Computer \- arXiv, accessed May 14, 2025, [https://www.arxiv.org/pdf/2505.01614](https://www.arxiv.org/pdf/2505.01614)  
79. A Quantum Q-Learning Model for the Capacitated Vehicle Routing Problem \- Inspire HEP, accessed May 14, 2025, [https://inspirehep.net/literature/2867584](https://inspirehep.net/literature/2867584)  
80. arxiv.org, accessed May 14, 2025, [https://arxiv.org/abs/2207.06294](https://arxiv.org/abs/2207.06294)  
81. Reinforcement Learning for Variational Quantum Circuits Design \- arXiv, accessed May 14, 2025, [https://arxiv.org/html/2409.05475v1](https://arxiv.org/html/2409.05475v1)  
82. How are the parameters in a variational circuit optimized?, accessed May 14, 2025, [https://quantumcomputing.stackexchange.com/questions/30249/how-are-the-parameters-in-a-variational-circuit-optimized](https://quantumcomputing.stackexchange.com/questions/30249/how-are-the-parameters-in-a-variational-circuit-optimized)  
83. \[2409.05475\] Reinforcement Learning for Variational Quantum Circuits Design \- arXiv, accessed May 14, 2025, [https://arxiv.org/abs/2409.05475](https://arxiv.org/abs/2409.05475)  
84. Variational quantum eigensolver, accessed May 14, 2025, [https://learning.quantum.ibm.com/tutorial/variational-quantum-eigensolver](https://learning.quantum.ibm.com/tutorial/variational-quantum-eigensolver)  
85. How to Create Reward Functions for Reinforcement Fine-Tuning \- Predibase, accessed May 14, 2025, [https://predibase.com/blog/reward-functions-reinforcement-fine-tuning](https://predibase.com/blog/reward-functions-reinforcement-fine-tuning)  
86. Optimizing Shot Assignment in Variational Quantum Eigensolver Measurement | Request PDF \- ResearchGate, accessed May 14, 2025, [https://www.researchgate.net/publication/378973405\_Optimizing\_Shot\_Assignment\_in\_Variational\_Quantum\_Eigensolver\_Measurement](https://www.researchgate.net/publication/378973405_Optimizing_Shot_Assignment_in_Variational_Quantum_Eigensolver_Measurement)  
87. Short quantum circuits in reinforcement learning policies for the vehicle routing problem, accessed May 14, 2025, [https://www.researchgate.net/publication/361082026\_Short\_quantum\_circuits\_in\_reinforcement\_learning\_policies\_for\_the\_vehicle\_routing\_problem](https://www.researchgate.net/publication/361082026_Short_quantum_circuits_in_reinforcement_learning_policies_for_the_vehicle_routing_problem)  
88. Scientists use reinforcement learning to train quantum algorithm, accessed May 14, 2025, [https://www.anl.gov/article/scientists-use-reinforcement-learning-to-train-quantum-algorithm](https://www.anl.gov/article/scientists-use-reinforcement-learning-to-train-quantum-algorithm)  
89. Curriculum reinforcement learning for quantum architecture search under hardware errors, accessed May 14, 2025, [https://openreview.net/forum?id=rINBD8jPoP¬eId=CrB0JrEQJY](https://openreview.net/forum?id=rINBD8jPoP&noteId=CrB0JrEQJY)  
90. A Comprehensive Review of Quantum Circuit Optimization: Current ..., accessed May 14, 2025, [https://www.mdpi.com/2624-960X/7/1/2](https://www.mdpi.com/2624-960X/7/1/2)  
91. What are the potential advantages of using quantum reinforcement learning with TensorFlow Quantum compared to traditional reinforcement learning methods? \- EITCA Academy, accessed May 14, 2025, [https://eitca.org/artificial-intelligence/eitc-ai-tfqml-tensorflow-quantum-machine-learning/quantum-reinforcement-learning/replicating-reinforcement-learning-with-quantum-variational-circuits-with-tfq/examination-review-replicating-reinforcement-learning-with-quantum-variational-circuits-with-tfq/what-are-the-potential-advantages-of-using-quantum-reinforcement-learning-with-tensorflow-quantum-compared-to-traditional-reinforcement-learning-methods/](https://eitca.org/artificial-intelligence/eitc-ai-tfqml-tensorflow-quantum-machine-learning/quantum-reinforcement-learning/replicating-reinforcement-learning-with-quantum-variational-circuits-with-tfq/examination-review-replicating-reinforcement-learning-with-quantum-variational-circuits-with-tfq/what-are-the-potential-advantages-of-using-quantum-reinforcement-learning-with-tensorflow-quantum-compared-to-traditional-reinforcement-learning-methods/)  
92. DIMACS :: Capacitated VRP \- DIMACS (Rutgers) \- Rutgers University, accessed May 14, 2025, [http://dimacs.rutgers.edu/programs/challenge/vrp/cvrp/](http://dimacs.rutgers.edu/programs/challenge/vrp/cvrp/)  
93. Benchmarks · VROOM-Project/vroom Wiki · GitHub, accessed May 14, 2025, [https://github.com/VROOM-Project/vroom/wiki/Benchmarks](https://github.com/VROOM-Project/vroom/wiki/Benchmarks)  
94. arxiv.org, accessed May 14, 2025, [https://arxiv.org/abs/2403.13795](https://arxiv.org/abs/2403.13795)  
95. improving the performance of genetic algorithms using reinforcement learning. \- DigitalCommons@URI, accessed May 14, 2025, [https://digitalcommons.uri.edu/theses/2357/](https://digitalcommons.uri.edu/theses/2357/)  
96. Capacitated Vehicle Routing Problems: Nearest Neighbour vs. Tabu Search \- IJCTE, accessed May 14, 2025, [https://www.ijcte.com/vol11/1246-E043.pdf](https://www.ijcte.com/vol11/1246-E043.pdf)  
97. accessed January 1, 1970, [https://www.ijcte.org/vol11/1246-E043.pdf](https://www.ijcte.org/vol11/1246-E043.pdf)  
98. \[1901.02771\] Sweep Algorithms for the Capacitated Vehicle Routing Problem with Structured Time Window \- arXiv, accessed May 14, 2025, [https://arxiv.org/abs/1901.02771](https://arxiv.org/abs/1901.02771)  
99. Sweep Algorithms for the Capacitated Vehicle Routing Problem with Structured Time Windows \- arXiv, accessed May 14, 2025, [https://arxiv.org/pdf/1901.02771](https://arxiv.org/pdf/1901.02771)  
100. OptaPlanner \- The fast, Open Source and easy-to-use solver, accessed May 14, 2025, [https://www.optaplanner.org/](https://www.optaplanner.org/)