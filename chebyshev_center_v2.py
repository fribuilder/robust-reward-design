import pickle
import cvxpy as cp
import numpy as np
import LPSOS1
import GridWorldV2
import unique_BR as UB
from gradient_v2 import extract_decoy_value
import MDP

def from_x_to_R2(x, mdp):
    '''turn the x into r_2'''
    D, E, F = LPSOS1.generate_matrix(mdp)
    assert x.size == len(mdp.F)
    st_len = len(mdp.statespace)
    act_len = len(mdp.A)

    A5 = np.zeros((st_len*act_len, len(mdp.F)))
    for i,decoy in enumerate(mdp.F):
        index = mdp.statespace.index(decoy)
        A5[index*act_len:(index+1)*act_len,i] = 1

    R_2 = D + A5 @ x
    return R_2

def chebyshev(m_star,C,mdp,init):
    C = C

    '''Use this function to find the chebyshev center of the Linear Bilevel Problem,
    this function needs the solution (optimal occupancy measure) derived by the mip first'''

    init = init
    
    D, E, F = LPSOS1.generate_matrix(mdp)
    A1, A2, A3, A4, R1_np = generate_matrix_for_constraints(mdp, init)
    # D is the vector of length st_len * act_len, and E and F are 2-d arrays of 
    # (st_len, st_len * act_len)
    st_len = len(mdp.statespace)
    act_len = len(mdp.A)


    #init[12] = 1 #6 * 6 case
    #init[30] = 1
    #init[0] = 1

    gamma = cp.Variable() # radius of the Chebyshev center
    center = cp.Variable(len(mdp.F))# center of the Chebyshev center

    objective = cp.Maximize(gamma)

    E = gamma * np.eye(len(mdp.F))
    mu = cp.Variable(((A1.shape[0]), len(mdp.F)*2))

    constraints = [center >= 0, #gamma for finding the CC inside the X
                    cp.sum(center) <= C] #C - gamma for finding the CC inside X
    
    for k in range(len(mdp.F)):
        constraints += [
            from_x_to_R2(center + E[:, k],mdp) @ m_star >= mu[:, k] @ init,
            from_x_to_R2(center - E[:, k],mdp) @ m_star >= mu[:, k+len(mdp.F)] @ init,
            mu[:, k] @ A1 >= from_x_to_R2(center + E[:, k],mdp),
            mu[:, k+len(mdp.F)] @ A1 >= from_x_to_R2(center - E[:, k],mdp)
        ]
        
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GUROBI)
    # where is the constraint that flow in = flow out
    #ecos, MOSEK
    
    return center.value, gamma.value, m_star


def generate_matrix_for_constraints(mdp, init):
    D, E, F = LPSOS1.generate_matrix(mdp)
    # D is the vector of length st_len * act_len, and E and F are 2-d arrays of 
    # (st_len, st_len * act_len)
    st_len = len(mdp.statespace)
    act_len = len(mdp.A)
    tau = 0.95
    init = init
    #init[0] = 1 #6 * 6 case

    # A1 @ m = init certifies feasible occupancy measure
    A1 = E - tau * F

    decoy_index = []  
    for decoy in mdp.F:
        decoy_index.append(mdp.statespace.index(decoy))

    # R1_np is the reward vector (of length |S|*|A|) for the defender. It is 1 for all the decoy states. 
    R1_np = np.zeros(st_len * act_len)  
    for i in decoy_index:
        R1_np[i*act_len:(i+1)*act_len] = 1

    # A2 pin out states that do not allow decoy
    A2 = np.eye(st_len * act_len) - np.diag(R1_np)

    # A3 sum the deccoy in y
    A3 = np.zeros(st_len*act_len)
    for i in decoy_index:
        A3[i*act_len] = 1


    # A4 makes sure that the decoy in every acctions are the same for the decoy state
    A4 = np.zeros((len(decoy_index)*(act_len-1), st_len*act_len))
    for i,j in enumerate(decoy_index):
        A4[i*(act_len-1):(i+1)*(act_len-1),j*act_len] = 1
    for i in range(len(decoy_index)):
        for j in range(act_len-1):
            A4[i*(act_len-1)+j,decoy_index[i]*act_len+j+1] = -1
    
    return A1,A2,A3,A4,R1_np

def reconstruct_policy_numerical(occupancy_measure,act_len):
    policy = np.zeros(len(occupancy_measure))
    occupancy_measure[np.where(occupancy_measure<1e-6)] = 0
    for i in range(len(occupancy_measure)):
        if sum(occupancy_measure[i//4*act_len:(i//4+1)*act_len]) == 0:
            policy[i] = 0
        else:
            policy[i] = occupancy_measure[i]/sum(occupancy_measure[i//4*act_len:(i//4+1)*act_len])
    return policy 

# if __name__ == '__main__':
    
#     # with open('y_res', 'rb') as fp:
#     #     y_res = pickle.load(fp)

#     with open('x_res_10_2', 'rb') as fp:
#         x_res_2 = pickle.load(fp)
#     x_res_2 = np.array(x_res_2)

#     with open('x_res_10', 'rb') as fp:
#         x_res = pickle.load(fp)
#     x_res = np.array(x_res)

#     with open('y_res_10_2', 'rb') as fp: # this use the optimal OC from Chebyshev_center_new
#         y_res_2 = pickle.load(fp)
#     y_res_2 = np.array(y_res_2)

#     with open('y_res_10', 'rb') as fp:
#         y_res = pickle.load(fp)
#     y_res = np.array(y_res)


#     test = True # for convenience of debugging, output matrices

#     mdp, _, _ = GridWorldV2.createGridWorldBarrier_new3()#new3 to new2 to change case
#     #policy, V_att, V_def, st_visit, mdp = MDP.test_att()
#     if test == True:
#         D, E, F = LPSOS1.generate_matrix(mdp)
#         A1, A2, A3, A4, R1_np = generate_matrix_for_constraints(mdp)

#     center, gamma, m_star = chebyshev(y_res, 4, mdp)
#     # m_new, policy, payoff = gv.eval_decoy_allocation(mdp, 0, center)
#     minimal, optimal = UB.is_unique(from_x_to_R2(center, mdp), 1e-7, mdp, m_star)

#     # with open('m_new', 'wb') as fp:
#     #     pickle.dump(m_new, fp)
#     print('optimal allocation from final problem:', extract_decoy_value(x_res_2, mdp))
#     print('optimal allocation from Chebyshev center:', center, gamma, minimal, optimal)

#     decoy_value = extract_decoy_value(x_res, mdp)
#     minimal, optimal = UB.is_unique(from_x_to_R2(decoy_value, mdp), 1e-7, mdp, m_star)
#     print('optimal allocation from mip and its PessVal and OptVal', decoy_value, minimal, optimal)



