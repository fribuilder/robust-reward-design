# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 22:43:21 2022

@author: 53055
"""

import MDP_V2 as MDP
import numpy as np

def calculateKL(M1, M2):
    KL = 0
    width = M1.shape[0]
    length = M1.shape[1]
    for i in range(width):
        for j in range(length):
            if M1[i][j] != 0:
                KL += M1[i][j] * np.log(M1[i][j]/M2[i][j])
    return KL

def modifyR1(mdp):
    reward = mdp.getworstcase_att(1)
    policy_att, V_att = mdp.getpolicy(reward)
    st_act_visit = mdp.stactVisitFre(policy_att)
    print(st_act_visit)
    M1 = createMatrix(st_act_visit, mdp)
    return M1

def modifyR2(mdp):
    r1 = -0.0051
    r2 = 1.1955
    r3 = 1.2005
    reward = mdp.getreward_att(1)
    reward = manualReward1(reward, r1, r2, r3)
    policy_att, V_att = mdp.getpolicy(reward)
    st_act_visit = mdp.stactVisitFre(policy_att)
    M2 = createMatrix(st_act_visit, mdp)
    return M2

def modifyR3(mdp):
    r1 = -0.02934
    r2 = 1.12742255
    r3 = 1.25293242
    r4 = 0.01037
    r5 = 0.01037
    r6 = 0.06066
    reward = mdp.getreward_att(1)
    reward = manualReward2(reward, r1, r2, r3, r4, r5, r6)
    policy_att, V_att = mdp.getpolicy(reward)
    st_act_visit = mdp.stactVisitFre(policy_att)
    print(st_act_visit)
    M3 = createMatrix(st_act_visit, mdp)
    return M3

def manualReward1(reward, r1, r2, r3):
    reward["q0"]["a"] = r1
    reward["q13"]["a"] = r2
    reward["q13"]["b"] = r2
    reward["q13"]["c"] = r2
    reward["q13"]["d"] = r2
    reward["q14"]["a"] = r3
    reward["q14"]["b"] = r3
    reward["q14"]["c"] = r3
    reward["q14"]["d"] = r3
    return reward

def manualReward2(reward, r1, r2, r3, r4, r5, r6):
    reward["q8"]["a"] = -1
    reward["q8"]["b"] = -0.98483216
    reward["q8"]["c"] = 0.04220065
    reward["q8"]["d"] =  -0.98483216
    reward["q10"]["a"] = 0.0804249
    reward["q10"]["b"] = 0.00888291
    reward["q10"]["c"] = 0.01990804
    reward["q10"]["d"] = 0.01990804
    reward["q13"]["a"] = r2
    reward["q13"]["b"] = r2
    reward["q13"]["c"] = r2
    reward["q13"]["d"] = r2
    reward["q14"]["a"] = r3
    reward["q14"]["b"] = r3
    reward["q14"]["c"] = r3
    reward["q14"]["d"] = r3
    return reward

def createMDP():
    IDSlist = ["q9"]
    G1 = ["q12"]
    F1 = ["q13", "q14"]
    U = ["q0", "q1", "q2", "q3", "q4", "q12", "q13", "q14"]
    mdp = MDP.MDP()
    mdp.getgoals(G1)
    mdp.getfakegoals(F1)
    mdp.addU(U)
    mdp.stotrans = mdp.getstochastictrans()
    mdp.addIDS(IDSlist)
#    reward = mdp.getworstcase_att(1)
#    policy_att, V_att = mdp.getpolicy(reward)
#    V_def = mdp.policyevaluation(policy_att)
#    st_visit = mdp.stVisitFre(policy_att)
#    st_act_visit = mdp.stactVisitFre(policy_att)
#    M1 = createMatrix(st_act_visit, mdp)
    M1 = modifyR1(mdp)
    M2 = modifyR2(mdp)
    M3 = modifyR3(mdp)
    return M1, M2, M3
    
def createMatrix(st_act_visit, mdp):
    M = np.zeros((len(mdp.statespace), len(mdp.statespace)))
    for i in range(len(mdp.statespace)):
        for j in range(len(mdp.statespace)):
            enum = 0
            denom = 0
            for a in mdp.A:
                if mdp.statespace[j] in mdp.stotrans[mdp.statespace[i]][a].keys(): 
                    enum += st_act_visit[mdp.statespace[i]][a] * mdp.stotrans[mdp.statespace[i]][a][mdp.statespace[j]]
                denom += st_act_visit[mdp.statespace[i]][a]
            M[i][j] = enum/denom
    return M
            

if __name__ == "__main__":
    M1, M2, M3 = createMDP()
    KL1 = calculateKL(M1, M2)
    KL2 = calculateKL(M1, M3)