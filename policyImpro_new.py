# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 17:20:49 2022

@author: 53055
"""

from mip import *
import numpy as np
import pickle

def PolicyImprove(mdp, p_ref, init_Z):
    model = Model()
    gamma = 1
    s_len = len(mdp.statespace)
    a_len = len(mdp.A)
    P = transferMatP(mdp.stotrans, mdp.statespace, mdp.A)
    G = []
    for st in mdp.G:
        G.append(mdp.statespace.index(st))
    F = []
    for st in mdp.F:
        F.append(mdp.statespace.index(st))
    Z = [[model.add_var() for j in range(a_len)] for i in range(s_len)]

    model.objective = minimize(xsum(Z[i][j] for i in G for j in range(a_len)))
    #Constrint 1: All state visiting frequency should larger than 0
    for i in range(s_len):
        for j in range(a_len):
            model += Z[i][j] >= 0
    
    #Constraint 2: The decoy visiting frequency should larger than the preferred policy
    model += (xsum(Z[i][j] for i in F for j in range(a_len)) >= p_ref)

    #Constraint 3: The inflow should equal to the outflow
    for i in range(len(mdp.statespace)):
        model += xsum(Z[i][j] for j in range(a_len)) == init_Z[i] + gamma * xsum(P[m][n][i] * Z[m][n] for m in range(s_len) for n in range(a_len))
        
    print("start optimization")
    status = model.optimize()
    print(status)
    print("Finish optimization")
    st_act_visit = np.zeros((s_len, a_len))
    for i in range(s_len):
        for j in range(a_len):
            st_act_visit[i][j] = Z[i][j]
    return st_act_visit
    
def transferMatP(P, statespace, actionspace):
    s_len = len(statespace)
    a_len = len(actionspace)
    trans = np.zeros((s_len, a_len, s_len))
    for i in range(s_len):
        state = statespace[i]
        for j in range(a_len):
            act = actionspace[j]
            for st_, pro in P[state][act].items():
                if st_ != "Sink":
                    st_index = statespace.index(st_)
                    trans[i][j][st_index] = pro
    return trans

if __name__ == "__main__":
    mdp_file = "gridworld2.pkl"
    with open(mdp_file, "rb") as f1:
        mdp = pickle.load(f1)
    reward_file = "rewardgrid2_5.pkl"
    with open(reward_file, "rb") as f1:
        reward = pickle.load(f1)
    init_Z = np.zeros(len(mdp.statespace))
    init_Z[12] = 1    
    policy_att, V_att = mdp.getpolicy(reward)
    st_visit = mdp.stVisitFre(policy_att)
    st_act_visit = mdp.stactVisitFre(policy_att)
    p_ref = 0
    for st in mdp.F:
        for act in mdp.A:
            p_ref += st_act_visit[st][act]
    new_st_act_visit = PolicyImprove(mdp, p_ref, init_Z)
    print(new_st_act_visit)
    