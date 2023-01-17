# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:51:21 2023

@author: hma2
"""

from mip import *
import numpy as np
import MDP
import MDP_V2
import GridWorldV2
import time

def LP(mdp, k):
    model = Model(solver_name=GRB)
    gamma = 0.95
    st_len = len(mdp.statespace)
    act_len = len(mdp.A)
    decoy_index = []
    for decoy in mdp.F:
        decoy_index.append(mdp.statespace.index(decoy))
    init = np.zeros(st_len)
#    act_modify = [99, 103, 107, 112, 113, 114, 115]
    act_modify = []
    # init[0] = 1 # mdp case
    # init[12] = 1 #6 * 6 case
    # init[51] = 1 #10 * 10 case
    init[30] = 1
    x = [model.add_var() for i in range(st_len)]
    y = [model.add_var() for i in range(st_len * act_len)]
    z = [model.add_var() for i in range(st_len * act_len)]
    lmd = [model.add_var() for i in range(st_len * act_len)] 
    mu = [model.add_var() for i in range(st_len)]
    R2 = np.zeros(st_len)
    for i in decoy_index:
        R2[i] = 1
    
    D, E, F = generate_matrix(mdp)
    model.objective = maximize(xsum(R2[i] * xsum(y[i*act_len + j] for j in range(act_len)) for i in decoy_index))
    
    #occupation meastures > 0
    for i in range(st_len * act_len):
        model += y[i] >= 0    
    
    #Total decoy resource budget <= k
    model += xsum(x[i] for i in decoy_index) <= k
    
    #SOS1 specification
#    model.add_sos([(y[i], lmd[i]) for i in range(st_len * act_len)], 1)
    for i in range(st_len * act_len):
        model.add_sos([(y[i], 1),(lmd[i], 1)], 1)
    
    #lmd >=0 
    for i in range(st_len * act_len):
        model += lmd[i] >= 0
        
    #Only state in decoys can be non-zero
    for i in range(st_len):
        if i not in decoy_index:
            model += x[i] == 0

    for i in range(st_len * act_len):
        if i in act_modify:
            model += z[i] >= 0
        else:
            model += z[i] == 0
    # for i in range(st_len * act_len):
    #     st_index = i // act_len
    #     if st_index in decoy_index:
    #         print(st_index)
    #         # model += x[st_index * act_len] - x[i] == 0 #if at decoy, x should be the same for each one
    #     elif i in act_modify:
    #         model += x[i] <= 0  #if at action modify place, x should smaller or equal to 0
    #     else:
    #         model += x[i] == 0  #all other places should be zero
    # for i in range(st_len):
    #     if i in decoy_index:
    #         for j in range(1,act_len):
    #             model += x[i * act_len] == x[i * act_len + j]
    #     else:
    #         for j in range(act_len):
    #             if i * act_len + j in act_modify:
    #                 model += x[i] <= 0
    #             else:
    #                 print(i, "is 0")
    #                 model += x[i] == 0
    
    #in flow = out flow
    for i in range(st_len):
        model += xsum((E[i][j] * y[j] - gamma * F[i][j] * y[j]) for j in range(st_len * act_len)) - init[i] == 0

    #KKT gradient
    G = (E - gamma * F).transpose()
    for i in range(st_len * act_len):
        model += -x[i//act_len] - z[i] - D[i] - lmd[i] + xsum(G[i][j] * mu[j] for j in range(st_len)) == 0
        
    print("Start optimization")
    #model.max_gap = 0.05
    status = model.optimize()   # Set the maximal calculation time
    print("Finish optimization")
    print(status)
    if status == OptimizationStatus.OPTIMAL:
        print("The model objective is:", model.objective_value)
        x_res = [x[i].x for i in range(st_len)]
        y_res = [y[i].x for i in range(st_len * act_len)]
        z_res = [z[i].x for i in range(st_len * act_len)]
        # print("x_res:", x_res)
        # for i in range(st_len):
            # for j in range(act_len):
                # print(i, "y_res:", y_res[i*act_len+j])
#        print("z_res:", z_res)

    elif status == OptimizationStatus.FEASIBLE:
        print('sol.cost {} found, best possible: {}'.format(model.objective_value, model.objective_bound))
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print('no feasible solution found, lower bound is: {}'.format(model.objective_bound))
    else:
        print("The model objective is:", model.objective_value)

    for i in range(st_len * act_len):
        if y_res[i] > 0:
            print(i //act_len, i%act_len, y_res[i])
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
    
    D = np.zeros(st_len * act_len)
    for i in mdp.G:
        goal_index = mdp.statespace.index(i)
        for j in range(act_len):
            D[goal_index * act_len + j] = 1
            
    return D, E, F

def test():
    #policy, V_att, V_def, st_visit, mdp = MDP.test_att()
    # mdp, policy, V_att, V_def, st_visit, st_act_visit = MDP_V2.test_att()
    mdp, V_def, policy = GridWorldV2.createGridWorldBarrier_new3()
    D, E, F = generate_matrix(mdp)
    return D, E, F, mdp
    
if __name__ == "__main__":
    D, E, F, mdp = test()
    k = 4
    start_time = time.time()
    LP(mdp, k)
    end_time = time.time()
    # print(reward)
    print("Running time:", end_time - start_time)