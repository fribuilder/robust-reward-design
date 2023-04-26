# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 00:58:34 2022

@author: 53055
"""

import numpy as np
from MDP_V2 import MDP
import copy
import pickle

def policyImpro(mdp, V, reward):
    tau = 0.01    #close to deterministic
    # tau = 1 #close to stochastic
    gamma = 0.95
    Q = {}
    Q_V = {}
    policy_improve = {}
    V1 = V.copy()
    for st in mdp.statespace:
        Q[st] = 0
        Q_V[st] = {}
        policy_improve[st] = {}
        for act in mdp.A:
            core = (reward[st][act] + gamma * mdp.getcore(V1, st, act))/tau
            Q_V[st][act] = np.exp(core)
        Q_s = sum(Q_V[st].values())
        for act in mdp.A:
            next_st_V = reward[st][act] + gamma * mdp.getcore(V1, st, act)
            policy_improve[st][act] = Q_V[st][act] / Q_s
            Q[st] += policy_improve[st][act] * next_st_V
    for st in mdp.statespace:
        if st in mdp.F:
            policy_improve[st] = {}
            policy[st][self.A[0]] = 0.9999997
            policy[st][self.A[1]] = 0.0000001
            policy[st][self.A[2]] = 0.0000001
            policy[st][self.A[3]] = 0.0000001
    return policy_improve, Q
        
    
def policyEval(mdp, reward, policy):
    threshold = 0.00001
    gamma = 0.95
    V = mdp.get_initial_value()
    V1 = V.copy()
    itcount = 1
    while (
        itcount == 1
        or np.inner(np.array(V) - np.array(V1), np.array(V) - np.array(V1))
        > threshold
    ):
        V1 = V.copy()
        for st in mdp.statespace:
            temp = 0
            for act in mdp.A:
                if act in policy[st].keys():
                    temp += policy[st][act] * (reward[st][act] + gamma * mdp.getcore(V1, st, act))
            V[mdp.statespace.index(st)] = temp
        itcount += 1
    return V

def policyEval_Ent(mdp, reward, policy):
    threshold = 0.00001
    gamma = 0.95
    tau = 0.01
    V = mdp.get_initial_value()
    V1 = V.copy()
    itcount = 1
    while (
        itcount == 1
        or np.inner(np.array(V) - np.array(V1), np.array(V) - np.array(V1))
        > threshold
    ):
        V1 = V.copy()
        for st in mdp.statespace:
            temp = 0
            for act in mdp.A:
                if act in policy[st].keys():
                    # temp += policy[st][act] * np.exp((reward[st][act] + gamma * mdp.getcore(V1, st, act))/tau)
                    temp += policy[st][act] * (reward[st][act] - tau * np.log(policy[st][act]) + gamma * mdp.getcore(V1, st, act))
            V[mdp.statespace.index(st)] =temp
        itcount += 1
    return V

def reward_def(mdp):
    reward = {}
    for st in mdp.statespace:
        reward[st] = {}
        if st in mdp.G:
            for act in mdp.A:
                reward[st][act] = -1
        else:
            for act in mdp.A:
                reward[st][act] = 0
    return reward


def norm_1(V1, V2, mdp):
    norm1 = 0
    theta_len = len(V1)
    for i in range(theta_len):
        diff = abs(V1[i]-V2[mdp.statespace[i]])
        norm1 += abs(diff)
#        print(i, abs(diff))
    return norm1

def cancelGoalReward(reward, mdp):
    for st in mdp.G:
        for act in mdp.A:
            reward[st][act] = 0
    return reward

def main():
    mdp_file = "gridworldAgent.pkl"
    with open(mdp_file, "rb") as f1:
        mdp = pickle.load(f1)
    reward_file = "rewardagent1_1.pkl"
    with open(reward_file, "rb") as f1:
        reward = pickle.load(f1)
#    print(reward)
#    reward_d = reward_def(mdp)
#    print(reward_d)
    policy_att, V_att = mdp.getpolicy(reward)
#    print(V_att[16])
    reward_d = mdp.getreward_def(1)
    V_def = policyEval(mdp, reward_d, policy_att)
#    print(V_def[16])
    policy_improve, V_improve = policyImpro(mdp, V_def, reward_d)
    st_visit_att = mdp.stVisitFre(policy_att)
    st_act_visit_att = mdp.stactVisitFre(policy_att)
    st_visit_imp = mdp.stVisitFre(policy_improve)
    st_act_visit_imp = mdp.stactVisitFre(policy_improve)
#    state = (2, 0)  #This is for gridworld with agent case
#    init_dist = mdp.init_dist(state)
#    st_visit = mdp.stVisitFre(policy_improve, init_dist)
#    st_act_visit = mdp.stactVisitFre(policy_improve, init_dist)
#    print("att policy:", policy_att)
#    print("improve policy:", policy_improve)
    return mdp, V_def, V_improve, st_visit, st_act_visit
    
if __name__ == "__main__":
    mdp, V_def, V_improve, st_visit_improve, st_act_visit_improve = main()
    st_act_visit_file = "st_act_visit_grid_agent1_1.pkl"
    picklefile = open(st_act_visit_file, "wb")
    pickle.dump(st_act_visit_improve, picklefile)
    picklefile.close()
    V_def_file = "V_def_1.pkl"
    picklefile = open(V_def_file, "wb")
    pickle.dump(V_def, picklefile)
    picklefile.close()
    V_improve_file = "V_improve_1.pkl"
    picklefile = open(V_improve_file, "wb")
    pickle.dump(V_improve, picklefile)
    picklefile.close()
    diff = norm_1(V_def, V_improve, mdp)