## Robust Reward Design

This repository contains code for the paper [*Robust Reward Design for Markov Decision Processes*](https://arxiv.org/html/2406.05086v1)

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

This repository is forked from [alexalvis/SynthesisAttack_V2](https://github.com/alexalvis/SynthesisAttack_V2). 
