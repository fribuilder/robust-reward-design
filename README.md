## Robust Reward Design

This repository contains code for the paper [*Robust Reward Design for Markov Decision Processes*](https://arxiv.org/abs/2406.05086)

### Reproducing the baselines

You may run the self contained `main.ipynb` notebook. It is divided into three sections, 6 x 6 grid world case, 10 x 10 grid world case and the attack graph case. Each of them corresponds to a different MDP environment.

### Code overview

- Function `LPSOS1.LP` solves a mixed integer linear program (MILP) to output an optimal solution (MILP solution) of the reward design problem, not necessarily an interior-point solution, however. The MILO solution contains an optimal occupancy measure and a reward function with optimal reward allocation. The inputs are 1) MDP environment, a class, 2) the number of states that allow allocation, 3) the initial state distribution of the MDP

- Function `chebyshev_center_v2.chebyshev` outputs an optimal interior allocation that may not have the largest margin. This part is not mentioned in the paper but could be used as an initial point for the later MILP to find the optimal interior-point allocation with the largest margin. The inputs are 1) the optimal occupancy measure of the MILP solution, 2) the number of states that allow allocation, 3) the MDP environment, a class 4) the initial state distribution of the MDP.

- Function 'chebyshev_center_new_2' solves another MILP that gives a reward function with optimal interior-point allocation with the largest margin (interior-point solution). The inputs are 1) the MDP environment, a class, 2) the number of states that allow allocation, 3) the initial state distribution of the MDP, and 4) the required margin; note that this is a parameter that the bisection method is implemented. Also, as mentioned, we may use the output of 'chebyshev_center_v2.chebyshev' as the lower bound.  5) optimal payoff of the leader obtained in MILP solution.

- Function 'chebyshev_center_new_2.extract_decoy_value' is used to obtain the reward allocation from the reward function with reward allocation.

- Function 'unique_BR' gives the OptVal(x) and PessVal(x) under tolerance 'tau', where x is the reward allocation. 

- Function 'gradient_new.ent_gradient' outputs 1) the feasibility of y, which tests whether the input occupancy measure is an optimal occupancy measure and the value is the difference of the payoff of the leader and the optimal payoff.  2) Attacker and defender's payoff under y_test 3) Attacker and defender's payoff when the agent is boundedly rational under y_test. The 'tau' in the input now controls the rationality of the agent.

- Function 'cyipopt_optimize.test_cyipopt_optimize' outputs the optimal payoff of the leader/defender under the given reward allocation (the last argument). 

This repository is forked from [alexalvis/SynthesisAttack_V2](https://github.com/alexalvis/SynthesisAttack_V2). The following is an example showing how to formulate the MILP problem, equation (22) in the paper, by the Python-MIP pacakge.  

## Example of Solving the MILP by Python Package

All the code excerpts in this example come from `reward_allocation_maximize.py` in the public code repository (link in the manuscript).

---

### 1. Initialize the model and define the optimization variables

```python
from mip import *  # Import the mip module

# Use Gurobi as the solver
model = Model(solver_name=GRB)
model.infeas_tol = 1e-4
model.integer_tol = 1e-4

# Decision variables
c = model.add_var()                                   # scalar: the margin
x = [model.add_var() for i in range(st_len)]          # vector: reward allocation
y = [model.add_var() for i in range(st_len * act_len)]  # vector: occupancy measure (m)
...
```

Here:

- **`x`** represents the reward allocation.  
- **`c`** represents the margin.  
- **`y`** represents the occupancy measure ($m$ in the mathematical formulation).

---

### 2. Add the equality (flow) constraints

Enforce the flow constraint $A m = 
\rho$, equation (19):

```python
for i in range(st_len):
    model += (
        xsum(
            (E[i][j] * y[j] - gamma * F[i][j] * y[j])
            for j in range(st_len * act_len)
        )
        - init[i] == 0
    )
```

- Here, `E - gamma * F` corresponds to the matrix \(A\).  
- `init` holds the initial distribution.

---

### 3. Add the inequality (non‑negativity) constraints

Ensure that each occupancy measure is non‑negative:

```python
for i in range(st_len * act_len):
    model += y[i] >= 0
```

---

### 4. Add the complementarity (SOS) constraints

Model the complementarity from equation (21) as a Type 1 Special Ordered Set (SOS1):

```python
for j in range(d_len):
    for i in range(st_len * act_len):
        # SOS1 on (y[i], λ⁺[…]) and (y[i], λ⁻[…])
        model.add_sos([(y[i], 1), (lmd_pos[j*st_len*act_len + i], 2)], 1)
        model.add_sos([(y[i], 1), (lmd_neg[j*st_len*act_len + i], 2)], 1)
```

- The final `1` in `add_sos(..., 1)` denotes an SOS1 constraint.

---

### 5. Set the objective and solve

Maximize the margin **`c`**, call the solver, and extract the results:

```python
# Define maximization objective
model.objective = maximize(c)

# Solve the MILP
status = model.optimize()

# On optimality, print and retrieve solution
if status == OptimizationStatus.OPTIMAL:
    print("The model objective is:", model.objective_value)
    x_res = [x[i].x for i in range(st_len)]
```

- `model.objective_value` gives the optimal margin.

---
