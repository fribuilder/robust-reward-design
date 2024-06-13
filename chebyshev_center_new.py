# -*- coding: utf-8 -*-
import pickle
from mip import *
import numpy as np
import MDP
import MDP_V2
import GridWorldV2
from gradient_v2 import extract_decoy_value
import unique_BR as UB

def LP(mdp, k, init, c, opt):
    model = Model(solver_name=GRB)
    model.emphasis = 0
    gamma = 0.95
    st_len = len(mdp.statespace)
    act_len = len(mdp.A)
    d_len = len(mdp.F)
    decoy_index = []
    for decoy in mdp.F:
        decoy_index.append(mdp.statespace.index(decoy))
    init = init
    #init[0] = 1 # mdp case
    #init[12] = 1 #6 * 6 case
    #init[30] = 1 #10 * 10 case
    #c = model.add_var() 
    c = c# this is get from the long time run of this problme when setting c to be variable
    #c = model.add_var()
    x = [model.add_var() for i in range(st_len)] # allocation
    y = [model.add_var() for i in range(st_len * act_len)] # occupancy measure
    lmd_pos = [model.add_var() for i in range(st_len * act_len)] # dual variable
    lmd_neg = [model.add_var() for i in range(st_len * act_len)]
    mu_pos = [model.add_var() for i in range(st_len)] # initial distribution
    mu_neg = [model.add_var() for i in range(st_len)]

    R2 = np.zeros(st_len) # defender's state action reward (why the reward is only relevant to the fake reward?)
    for i in decoy_index:
        R2[i] = 1

    #perturbation according to standard vector perturbation
    perturbation = np.zeros(st_len)
    perturbation[decoy_index[1]] = 1
    
    D, E, F = generate_matrix(mdp)
    #model.objective = maximize(c)
    #model.objective = maximize(xsum(R2[i] * xsum(y[i*act_len + j] for j in range(act_len)) for i in decoy_index))
    
    #m needs to be an optimal occupancy measure
    model += xsum(R2[i] * xsum(y[i*act_len + j] for j in range(act_len)) for i in decoy_index) >= opt
    
    #occupation meastures > 0
    for i in range(st_len * act_len):
        model += y[i] >= 0    
    
    #Total resource budget <= k
    model += xsum(x[i] for i in decoy_index) <= k 

    #lmd >=0 
    for i in range(st_len * act_len):
        model += lmd_pos[i] >= 0
        model += lmd_neg[i] >= 0
        
    #Only state in decoys can be non-zero
    for i in range(st_len):
        if i not in decoy_index:
            model += x[i] == 0
    
    #in flow = out flow
    for i in range(st_len):
        model += xsum((E[i][j] * y[j] - gamma * F[i][j] * y[j]) for j in range(st_len * act_len)) - init[i] == 0

    
    #SOS1 specification
#    model.add_sos([(y[i], lmd[i]) for i in range(st_len * act_len)], 1)
    for i in range(st_len * act_len):
        model.add_sos([(y[i],1),(lmd_pos[i],2)], 1)
        model.add_sos([(y[i],1),(lmd_neg[i],2)], 1)
        

    #KKT gradient
    G = (E - gamma * F).transpose()
    for i in range(st_len * act_len):
        model += -x[i//act_len] - D[i] - c*perturbation[i//act_len] - lmd_pos[i] + xsum(G[i][j] * mu_pos[j] for j in range(st_len)) == 0
        model += -x[i//act_len] - D[i] + c*perturbation[i//act_len] - lmd_neg[i] + xsum(G[i][j] * mu_neg[j] for j in range(st_len)) == 0


    print("Start optimization")
    #model.max_gap = 0.05
    status = model.optimize()   # Set the maximal calculation time
    print("Finish optimization")
    print(status)
    if status == OptimizationStatus.OPTIMAL:
        print("The model objective is:", model.objective_value)
        x_res = [x[i].x for i in range(st_len)]
        with open("x_res_A_2", "wb") as fp:   #Pickling
           pickle.dump(x_res, fp)
        y_res = [y[i].x for i in range(st_len * act_len)]
        with open("y_res_A_2", "wb") as fp:   #Pickling
           pickle.dump(y_res, fp)
        #print("x_res:", x_res)
        #print("y_res:", y_res)
        print('margin:', c)
    elif status == OptimizationStatus.FEASIBLE:
        print('sol.cost {} found, best possible: {}'.format(model.objective_value, model.objective_bound))
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print('no feasible solution found, lower bound is: {}'.format(model.objective_bound))
    else:
        print("The model objective is:", model.objective_value)
    
def generate_matrix(mdp):
    st_len = len(mdp.statespace)
    act_len = len(mdp.A)
    
    #sum over all action wrt one state, out visit
    E = np.zeros((st_len, st_len * act_len))
    for i in range(st_len):
        for j in range(act_len):
            E[i][i * act_len + j] = 1
    
    #in visit, corresponds to the upper one
    F = np.zeros((st_len, st_len * act_len))
    for st in mdp.stotrans.keys():
        for act in mdp.stotrans[st].keys():
            for st_, pro in mdp.stotrans[st][act].items():
                if st_ in mdp.statespace:
                    F[mdp.statespace.index(st_)][mdp.statespace.index(st) * act_len + mdp.A.index(act)] = pro
    
    '''goal index'''
    D = np.zeros(st_len * act_len)
    for i in mdp.G:
        goal_index = mdp.statespace.index(i)
        for j in range(act_len):
            D[goal_index * act_len + j] = 1
            
    return D, E, F

def from_x_to_R2(x, mdp):
    '''turn the x into r_2'''
    D, _, _ = generate_matrix(mdp)
    assert len(x) == len(mdp.F)
    st_len = len(mdp.statespace)
    act_len = len(mdp.A)

    A5 = np.zeros((st_len*act_len, len(mdp.F)))
    for i,decoy in enumerate(mdp.F):
        index = mdp.statespace.index(decoy)
        A5[index*act_len:(index+1)*act_len,i] = 1

    R_2 = D + A5 @ x
    return R_2

# def test():
#     #policy, V_att, V_def, st_visit, mdp = MDP.test_att()
#     # mdp, policy, V_att, V_def, st_visit, st_act_visit = MDP_V2.test_att()
#     mdp, V_def, policy = GridWorldV2.createGridWorldBarrier_new3()#new3 to new2 to change case
#     D, E, F = generate_matrix(mdp)
#     return D, E, F, mdp
    
# if __name__ == "__main__":
#     D, E, F, mdp = test()
#     k = 4
#     LP(mdp, k)

#     with open('x_res_A_2', 'rb') as fp:
#         x_res_2 = pickle.load(fp)
#     x_res_2 = np.array(x_res_2)

#     with open('x_res_A', 'rb') as fp:
#         x_res = pickle.load(fp)
#     x_res = np.array(x_res)

#     with open('y_res_A_2', 'rb') as fp:
#         y_res_2 = pickle.load(fp)
#     y_res_2 = np.array(y_res_2)

#     with open('y_res_A', 'rb') as fp:
#         y_res = pickle.load(fp)
#     y_res = np.array(y_res)

#     decoy_value = extract_decoy_value(x_res, mdp)
#     decoy_value_2 =  extract_decoy_value(x_res_2, mdp)
#     print('value of decoys:', decoy_value_2)

#     minimal, optimal = UB.is_unique(from_x_to_R2(decoy_value, mdp), 1e-3, mdp, y_res)
#     print('PessVal and Optval of MIP:', minimal, optimal)
#     minimal, optimal = UB.is_unique(from_x_to_R2(decoy_value_2, mdp), 1e-6, mdp, y_res_2)
#     print('PessVal and OptVal of Robust Solution:', minimal, optimal)
