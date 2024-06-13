import cvxpy as cp
import numpy as np
import gradient_v2 as gv
import pickle 
from LPSOS1 import generate_matrix

def is_unique(x: np.ndarray, eps: float, mdp, y_res, init):
    D, E, F = generate_matrix(mdp)
    st_len = len(mdp.statespace)
    act_len = len(mdp.A)

    init = init
    #init[12] = 1 #6 * 6 case
    #init[30] = 1 #10 * 10 case
    #init[0] = 1

    decoy_index = []  
    for decoy in mdp.F:
        decoy_index.append(mdp.statespace.index(decoy))

    # R1_np is the reward vector (of length |S|*|A|) for the defender. It is 1 for all the decoy states. 
    R1_np = np.zeros(st_len * act_len)  
    for i in decoy_index:
        R1_np[i*act_len:(i+1)*act_len] = 1

    m_star = np.array(y_res)
    x = x 
    eps = eps
    A1 = E - 0.95 * F


    m = cp.Variable(st_len*act_len)
    objective = cp.Minimize(R1_np @ m)
    constraints = [x @ (m_star - m) <= eps,
                   m >= 0,
                   A1 @ m == init]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver = cp.GUROBI)
    
    return R1_np @ m.value, R1_np @ m_star

