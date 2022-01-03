# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 13:57:54 2021

@author: 53055
"""

import numpy as np
import copy
from collections import defaultdict
from itertools import combinations


class MDP:
    def __init__(self):
        self.statespace = self.getstate()
        self.A = self.getaction()
        self.trans = self.gettransition()
        self.F = self.getfakegoals()
        self.U = (
            []
        )  # opposite of the draft, U is the set where sensors are not allowed.
        self.G = self.getgoals()
        self.IDS = []
        self.stotrans = self.getstochastictrans()

    def getstate(self):
        # Manually define statespace
        statelist = [
            "q0",
            "q1",
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",
            "q7",
            "q8",
            "q9",
            "q10",
            "q11",
            "q12",
            "q13",
            "q14",
        ]
        return statelist

    def getaction(self):
        A = ["a", "b", "c", "d"]
        return A

    def gettransition(self):
        # Manually define transition
        trans = {}
        for st in self.statespace:
            trans[st] = {}
        trans["q0"]["a"] = "q1"
        trans["q0"]["b"] = "q2"
        trans["q0"]["c"] = "q3"
        trans["q0"]["d"] = "q4"
        trans["q1"]["a"] = "q5"
        trans["q1"]["b"] = "q8"
        trans["q1"]["c"] = "q6"
        trans["q2"]["a"] = "q6"
        trans["q2"]["b"] = "q7"
        trans["q3"]["b"] = "q5"
        trans["q3"]["c"] = "q7"
        trans["q4"]["c"] = "q7"
        trans["q4"]["d"] = "q5"
        trans["q5"]["b"] = "q8"
        trans["q5"]["a"] = "q10"
        trans["q5"]["d"] = "q11"
        trans["q6"]["b"] = "q9"
        trans["q6"]["d"] = "q11"
        trans["q7"]["a"] = "q9"
        trans["q7"]["b"] = "q8"
        trans["q8"]["a"] = "q9"
        trans["q8"]["c"] = "q11"
        trans["q9"]["b"] = "q12"
        trans["q9"]["c"] = "q14"
        trans["q10"]["a"] = "q13"
        trans["q10"]["b"] = "q14"
        trans["q11"]["c"] = "q12"
        trans["q11"]["d"] = "q13"
        trans["q12"]["a"] = "q13"
        trans["q12"]["d"] = "q14"
        trans["q13"]["b"] = "q12"
        trans["q13"]["c"] = "q14"
        trans["q14"]["a"] = "q13"
        trans["q14"]["c"] = "q12"
        return trans

    def getstochastictrans(self):
        stotrans = {}
        for st in self.statespace:
            if (st not in self.F) and (st not in self.G):
                stotrans[st] = {}
                for act in self.A:
                    stotrans[st][act] = {}
                    if act in self.trans[st].keys():
                        stotrans[st][act][st] = 0
                        stotrans[st][act][self.trans[st][act]] = 0.7
                        for otheract in self.A:
                            if otheract != act:
                                if otheract not in self.trans[st].keys():
                                    stotrans[st][act][st] += 0.1
                                else:
                                    stotrans[st][act][self.trans[st][otheract]] = 0.1
                    else:
                        stotrans[st][act][st] = 0.7
                        for otheract in self.A:
                            if otheract != act:
                                if otheract not in self.trans[st].keys():
                                    stotrans[st][act][st] += 0.1
                                else:
                                    stotrans[st][act][self.trans[st][otheract]] = 0.1
            else:
                stotrans[st] = {}
                for act in self.A:
                    stotrans[st][act] = {}
                    stotrans[st][act][st] = 1.0
        if checkstotrans(stotrans):
            return stotrans
        else:
            print(stotrans)

    def getfakegoals(self, F=["q12", "q14"]):
        self.F = F
        return F

    def getgoals(self, G=["q13"]):
        self.G = G
        self.U = G  # we do not allow sensor placed in G.
        return G

    def addIDS(self, IDSlist):
        for ids in IDSlist:
            self.IDS.append(ids)
        for ids in IDSlist:
            self.trans[ids] = {}
            self.stotrans[ids] = {}
            for act in self.A:
                self.trans[ids][act] = ids
                self.stotrans[ids][act] = {}
                self.stotrans[ids][act][ids] = 1.0

    def getcore(self, V, st, act):
        core = 0
        for st_, pro in self.stotrans[st][act].items():
            core += pro * V[self.statespace.index(st_)]
        return core

    def getpolicy_det(self, gamma=0.95):
        threshold = 0.00001
        V = self.init_value_att()
        V1 = V.copy()
        policy = {}
        Q = {}
        for st in self.statespace:
            policy[st] = {}
            Q[st] = {}
        itcount = 1
        while (
            itcount == 1
            or np.inner(np.array(V) - np.array(V1), np.array(V) - np.array(V1))
            > threshold
        ):
            V1 = V.copy()
            for st in self.statespace:
                if st not in self.F and st not in self.G:
                    maxvalue = -1
                    for act in self.A:
                        tempvalue = gamma * self.getcore(V1, st, act)
                        if tempvalue > maxvalue:
                            maxvalue = tempvalue
                            policy[st] = {}
                            policy[st][act] = 1.0
                    V[self.statespace.index(st)] = maxvalue
                else:
                    policy[st] = {}
                    policy[st]["a"] = 1.0
            #            print("iteration count:", itcount)
            itcount += 1
        return policy, V

    def getpolicy_det_min(self, gamma=0.95):
        threshold = 0.00001
        V = self.init_value_min()
        V1 = V.copy()
        policy = {}
        Q = {}
        for st in self.statespace:
            policy[st] = {}
            Q[st] = {}
        itcount = 1
        while (
            itcount == 1
            or np.inner(np.array(V) - np.array(V1), np.array(V) - np.array(V1))
            > threshold
        ):
            V1 = V.copy()
            for st in self.statespace:
                if st not in self.F and st not in self.G and st not in self.IDS:
                    minvalue = 1
                    for act in self.A:
                        tempvalue = gamma * self.getcore(V1, st, act)
                        if tempvalue < minvalue:
                            minvalue = tempvalue
                            policy[st] = {}
                            policy[st][act] = 1.0
                    V[self.statespace.index(st)] = minvalue
                else:
                    policy[st] = {}
                    policy[st]["a"] = 1.0
            #            print("iteration count:", itcount)
            itcount += 1
        for st in self.statespace:
            for act in self.A:
                if act not in policy[st].keys():
                    policy[st][act] = 0
        return policy, V

    def init_value_min(self):
        V = []
        for st in self.statespace:
            if st not in self.G and st not in self.IDS:
                V.append(0)
            else:
                V.append(1)

        return V

    def init_value_att_v2(self):
        V = []
        for st in self.statespace:
            if st in self.F:
                V.append(1)
            else:
                V.append(0)
        return V

    def init_value_att_enu(self):
        #        V = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.82, 1.16, 1.01, 1.14]  #Include true goal
        V = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1.2107,
            0,
            1.2094,
        ]  # Exclude true goal
        #        V = [0, 0, 0, 0, 0, 0, 0, 0, 0.981, 0, 0, 1, 1.07, 1.064, 1.069]
        #        V = [0, 0, 0, 0, 0, 0, 0, 0, 0.607, 0, 0, 1, 0.776, 0.678, 0.784]
        return V

    def getpolicy(self,V_init, gamma=0.95):
        threshold = 0.00001
        tau = 0.01
#        V = self.init_value_att()   #Attacker's true value
#        V = self.init_value_att_v2()  #Maximize the probability of reaching the decoys and minimize the probability of reaching IDS and true goal
#        V = self.init_value_att_enu()  # Test the given value returned by maxEnt
        V = V_init
        V1 = V.copy()
        policy = {}
        Q = {}
        for st in self.statespace:
            policy[st] = {}
            Q[st] = {}
        itcount = 1
        while (
            itcount == 1
            or np.inner(np.array(V) - np.array(V1), np.array(V) - np.array(V1))
            > threshold
        ):
            V1 = V.copy()
            for st in self.statespace:
                if st not in self.F and st not in self.G and st not in self.IDS:
                    for act in self.A:
                        #                        if act in self.trans[st].keys():
                        #                            next_st = self.trans[st][act]
                        #                        else:
                        #                            next_st = st
                        #                        core = gamma * V1[self.statespace.index(next_st)]/tau
                        core = gamma * self.getcore(V1, st, act) / tau
                        Q[st][act] = np.exp(core)
                    Q_s = sum(Q[st].values())
                    #                    tempsum = 0
                    for act in self.A:
                        policy[st][act] = Q[st][act] / Q_s
                    #                        tempsum += policy[st][act] * self.getcore(V1, st, act)
                    #                    V[self.statespace.index(st)] = tempsum
                    V[self.statespace.index(st)] = tau * np.log(Q_s)
                else:
                    policy[st] = {}
                    policy[st]["a"] = 1.0
                    policy[st]["b"] = 0.0
                    policy[st]["c"] = 0.0
                    policy[st]["d"] = 0.0
            #                    for act in self.A:
            #                        core = gamma * self.getcore(V1, st, act)/tau
            #                        Q[st][act] = np.exp(core)
            #                    Q_s = sum(Q[st].values())
            #                    V[self.statespace.index(st)] = tau * np.log(Q_s)
            #            print("iteration count:", itcount)
            itcount += 1
        return policy, V

    def init_value_att(self):
        V = []
        for st in self.statespace:
            if st not in self.G:
                V.append(0)
            else:
                V.append(1)
        return V

    def init_value_def(self):
        V = []
        for st in self.statespace:
            if st in self.IDS:
                V.append(0)
            elif st in self.F:
                V.append(0)
            elif st in self.G:
                V.append(-1)
            else:
                V.append(0)
        return V

    def policyevaluation(self, policy, gamma=0.95):
        threshold = 0.00001
        V = self.init_value_def()
        V1 = V.copy()
        itcount = 1
        while (
            itcount == 1
            or np.inner(np.array(V) - np.array(V1), np.array(V) - np.array(V1))
            > threshold
        ):
            V1 = V.copy()
            for st in self.statespace:
                if st not in self.IDS and st not in self.F and st not in self.G:
                    temp = 0
                    for act in self.A:
                        if act in policy[st].keys():
                            temp += gamma * policy[st][act] * self.getcore(V1, st, act)
                    V[self.statespace.index(st)] = temp
                else:
                    pass
            #            print("iteration count:", itcount)
            itcount += 1
        return V

    def init_value_wstatt(self):
        V = []
        for st in self.statespace:
            if st not in self.F:
                V.append(0)
            else:
                V.append(1)
        return V

    def getwstattpolicy(self, gamma=0.95):
        threshold = 0.00001
        tau = 0.005
        V = self.init_value_wstatt()
        V1 = V.copy()
        policy = {}
        Q = {}
        for st in self.statespace:
            policy[st] = {}
            Q[st] = {}
        itcount = 1
        while (
            itcount == 1
            or np.inner(np.array(V) - np.array(V1), np.array(V) - np.array(V1))
            > threshold
        ):
            V1 = V.copy()
            for st in self.statespace:
                if st not in self.F and st not in self.G:
                    for act in self.A:
                        #                        if act in self.trans[st].keys():
                        #                            next_st = self.trans[st][act]
                        #                        else:
                        #                            next_st = st
                        #                        core = gamma * V1[self.statespace.index(next_st)]/tau
                        core = gamma * self.getcore(V1, st, act) / tau
                        Q[st][act] = np.exp(core)
                    Q_s = sum(Q[st].values())
                    for act in self.A:
                        policy[st][act] = Q[st][act] / Q_s
                    V[self.statespace.index(st)] = tau * np.log(Q_s)
                else:
                    policy[st] = {}
                    policy[st]["a"] = 1.0
            #            print("iteration count:", itcount)
            itcount += 1
        return policy, V

    def stVisitFre(self, policy):
        threshold = 0.0001
        Z0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Z_new = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Z_old = Z_new.copy()
        itcount = 1
        sinkst = self.F + self.G + self.IDS
        while (
            itcount == 1
            or np.inner(
                np.array(Z_new) - np.array(Z_old), np.array(Z_new) - np.array(Z_old)
            )
            > threshold
        ):
            print("Itcount:", itcount)
            Z_old = Z_new.copy()
            Z_new = Z0.copy()
            for st in self.statespace:
                index_st = self.statespace.index(st)
                if st not in self.F and st not in self.G and st not in self.IDS:
                    for act in self.A:
                        for st_ in self.statespace:
                            if st in self.stotrans[st_][act].keys():
                                Z_new[index_st] += (
                                    Z_old[self.statespace.index(st_)]
                                    * policy[st_][act]
                                    * self.stotrans[st_][act][st]
                                )
                else:
                    for act in self.A:
                        for st_ in self.statespace:
                            if st_ not in sinkst:
                                if st in self.stotrans[st_][act].keys():
                                    Z_new[index_st] += (
                                        Z_old[self.statespace.index(st_)]
                                        * policy[st_][act]
                                        * self.stotrans[st_][act][st]
                                    )
#            print(Z_new)
#            input("111")
            itcount += 1
        return Z_new


def checkstotrans(trans):
    for st in trans.keys():
        for act in trans[st].keys():
            if abs(sum(trans[st][act].values()) - 1) > 0.01:
                print(
                    "st is:",
                    st,
                    " act is:",
                    act,
                    " sum is:",
                    sum(trans[st][act].values()),
                )
                return False
    return True


def test_submod_IDS():
    IDSselections = ["q5", "q6", "q7", "q8", "q9", "q10", "q11"]
    IDSlist_1 = list(combinations(IDSselections, 1))
    IDSlist_2 = list(combinations(IDSselections, 2))
    IDSlist_3 = list(combinations(IDSselections, 3))
    Vdict_1 = {}
    Vdict_2 = {}
    Vdict_3 = {}
    V_att_1 = {}
    V_att_2 = {}
    V_att_3 = {}
    for idslist in IDSlist_1:
        mdp = MDP()
        mdp.addIDS(idslist)
        policy_att, V_att = mdp.getpolicy_det()
        V_att_1[idslist] = V_att
        V_defender = mdp.policyevaluation(policy_att)
        Vdict_1[idslist] = V_defender

    for idslist in IDSlist_2:
        mdp = MDP()
        mdp.addIDS(idslist)
        policy_att, V_att = mdp.getpolicy_det()
        V_att_2[idslist] = V_att
        V_defender = mdp.policyevaluation(policy_att)
        Vdict_2[idslist] = V_defender

    for idslist in IDSlist_3:
        mdp = MDP()
        mdp.addIDS(idslist)
        policy_att, V_att = mdp.getpolicy_det()
        V_att_3[idslist] = V_att
        V_defender = mdp.policyevaluation(policy_att)
        Vdict_3[idslist] = V_defender
    if (Vdict_2[("q5", "q7")][0] - Vdict_1[("q5",)][0]) > (
        Vdict_3[("q5", "q6", "q7")][0] - Vdict_2[("q5", "q6")][0]
    ):
        print("True")
    else:
        print("False")


def test_submod_dcy(IDSlist=["q5", "q10"]):

    F1 = [
        ["q12"],
        ["q13"],
        ["q14"],
        ["q12", "q13"],
        ["q12", "q14"],
        ["q13", "q14"],
        ["q12", "q13", "q14"],
    ]
    G1 = ["q11"]
    F2 = [
        ["q11"],
        ["q13"],
        ["q14"],
        ["q11", "q13"],
        ["q11", "q14"],
        ["q13", "q14"],
        ["q11", "q13", "q14"],
    ]
    G2 = ["q12"]
    F3 = [
        ["q11"],
        ["q12"],
        ["q14"],
        ["q11", "q12"],
        ["q11", "q14"],
        ["q12", "q14"],
        ["q11", "q12", "q14"],
    ]
    G3 = ["q13"]
    F4 = [
        ["q11"],
        ["q12"],
        ["q13"],
        ["q11", "q12"],
        ["q11", "q13"],
        ["q12", "q13"],
        ["q11", "q12", "q13"],
    ]
    G4 = ["q14"]
    G = [G1, G2, G3, G4]
    F = [F1, F2, F3, F4]
    valuedict = {}
    policy = {}
    for i in range(4):
        for j in range(len(F[i])):
            index = str(i) + str(j)
            mdp = MDP()
            mdp.getgoals(G[i])
            mdp.getfakegoals(F[i][j])
            mdp.stotrans = mdp.getstochastictrans()
            mdp.addIDS(IDSlist)
            #            print(mdp.G)
            #            print(mdp.F)
            #            policy_att, V_att = mdp.getpolicy_det()
            policy_att, V_att = mdp.getpolicy()
            V_defender = mdp.policyevaluation(policy_att)
            valuedict[index] = V_defender
            policy[index] = policy_att
    return valuedict, policy


def test_set_4(valuedict):
    flag = 0
    for i in range(4):
        str_i = str(i)
        #        str_i = "3"
        if (valuedict[str_i + "6"][0] - valuedict[str_i + "5"][0]) > (
            valuedict[str_i + "3"][0] - valuedict[str_i + "1"][0]
        ):
            print(
                str_i + "6",
                str_i + "5" + "set 1",
                valuedict[str_i + "6"][0]
                - valuedict[str_i + "5"][0]
                - (valuedict[str_i + "3"][0] - valuedict[str_i + "1"][0]),
            )
            flag = 1
        if (valuedict[str_i + "6"][0] - valuedict[str_i + "5"][0]) > (
            valuedict[str_i + "4"][0] - valuedict[str_i + "2"][0]
        ):
            print(
                str_i + "6",
                str_i + "5" + "set 2",
                valuedict[str_i + "6"][0]
                - valuedict[str_i + "5"][0]
                - (valuedict[str_i + "4"][0] - valuedict[str_i + "2"][0]),
            )
            flag = 1
        if (valuedict[str_i + "6"][0] - valuedict[str_i + "4"][0]) > (
            valuedict[str_i + "3"][0] - valuedict[str_i + "0"][0]
        ):
            print(
                str_i + "6",
                str_i + "4" + "set 3",
                valuedict[str_i + "6"][0]
                - valuedict[str_i + "5"][0]
                - (valuedict[str_i + "3"][0] - valuedict[str_i + "0"][0]),
            )
            flag = 1
        if (valuedict[str_i + "6"][0] - valuedict[str_i + "4"][0]) > (
            valuedict[str_i + "5"][0] - valuedict[str_i + "2"][0]
        ):
            print(
                str_i + "6",
                str_i + "4" + "set 4",
                valuedict[str_i + "6"][0]
                - valuedict[str_i + "5"][0]
                - (valuedict[str_i + "5"][0] - valuedict[str_i + "2"][0]),
            )
            flag = 1
        if (valuedict[str_i + "6"][0] - valuedict[str_i + "3"][0]) > (
            valuedict[str_i + "4"][0] - valuedict[str_i + "0"][0]
        ):
            print(
                str_i + "6",
                str_i + "3" + "set 5",
                valuedict[str_i + "6"][0]
                - valuedict[str_i + "5"][0]
                - (valuedict[str_i + "4"][0] - valuedict[str_i + "0"][0]),
            )
            flag = 1
        if (valuedict[str_i + "6"][0] - valuedict[str_i + "3"][0]) > (
            valuedict[str_i + "5"][0] - valuedict[str_i + "1"][0]
        ):
            print(
                str_i + "6",
                str_i + "3" + "set 6",
                valuedict[str_i + "6"][0]
                - valuedict[str_i + "5"][0]
                - (valuedict[str_i + "5"][0] - valuedict[str_i + "1"][0]),
            )
            flag = 1
        if flag:
            return False
        else:
            return True
    #        if valuedict[str_i + "6"][0] - valuedict[str_i + "4"][0] < 0:
    #            print(str_i + "6", str_i + "4", valuedict[str_i + "6"][0] - valuedict[str_i + "4"][0])
    #            return False
    #        if valuedict[str_i + "6"][0] - valuedict[str_i + "3"][0] < 0:
    #            print(str_i + "6", str_i + "3", valuedict[str_i + "6"][0] - valuedict[str_i + "3"][0])
    #            return False
    #        if valuedict[str_i + "6"][0] - valuedict[str_i + "2"][0] < 0:
    #            print(str_i + "6", str_i + "2", valuedict[str_i + "6"][0] - valuedict[str_i + "2"][0])
    #            return False
    #        if valuedict[str_i + "6"][0] - valuedict[str_i + "1"][0] < 0:
    #            print(str_i + "6", str_i + "1", valuedict[str_i + "6"][0] - valuedict[str_i + "1"][0])
    #            return False
    #        if valuedict[str_i + "6"][0] - valuedict[str_i + "0"][0] < 0:
    #            print(str_i + "6", str_i + "0", valuedict[str_i + "6"][0] - valuedict[str_i + "0"][0])
    #            return False
    #        if valuedict[str_i + "5"][0] - valuedict[str_i + "1"][0] < 0:
    #            print(str_i + "5", str_i + "1", valuedict[str_i + "5"][0] - valuedict[str_i + "1"][0])
    #            return False
    #        if valuedict[str_i + "5"][0] - valuedict[str_i + "2"][0] < 0:
    #            print(str_i + "5", str_i + "2", valuedict[str_i + "5"][0] - valuedict[str_i + "2"][0])
    #            return False
    #        if valuedict[str_i + "4"][0] - valuedict[str_i + "0"][0] < 0:
    #            print(str_i + "4", str_i + "0", valuedict[str_i + "4"][0] - valuedict[str_i + "0"][0])
    #            return False
    #        if valuedict[str_i + "4"][0] - valuedict[str_i + "2"][0] < 0:
    #            print(str_i + "4", str_i + "2", valuedict[str_i + "4"][0] - valuedict[str_i + "2"][0])
    #            return False
    #        if valuedict[str_i + "3"][0] - valuedict[str_i + "0"][0] < 0:
    #            print(str_i + "3", str_i + "0", valuedict[str_i + "3"][0] - valuedict[str_i + "0"][0])
    #            return False
    #        if valuedict[str_i + "3"][0] - valuedict[str_i + "1"][0] < 0:
    #            print(str_i + "3", str_i + "1", valuedict[str_i + "3"][0] - valuedict[str_i + "1"][0])
    #            return False
    return True


def test_worst_att():
    IDSlist = ["q5", "q10"]
    G1 = ["q11"]
    mdp = MDP()
    mdp.getgoals(G1)
    mdp.getfakegoals([])
    mdp.stotrans = mdp.getstochastictrans()
    mdp.addIDS(IDSlist)
    policy_att, V_att = mdp.getpolicy_det_min()
    return policy_att, V_att


def test_att():
    """
    In this test function, the agent is maximizing the probability of reaching the decoys
    while avoiding the IDS placements and the true goal
    """
    IDSlist = ["q8"]
    G1 = ["q11"]
    F1 = ["q12", "q14"]
#    F1 = ["q13", "q14"]
    mdp = MDP()
    mdp.getgoals(G1)
    mdp.getfakegoals(F1)
    mdp.stotrans = mdp.getstochastictrans()
    mdp.addIDS(IDSlist)
#    V_init = mdp.init_value_att()   #Attacker's true value
#    V_init = mdp.init_value_att_v2()  #Maximize the probability of reaching the decoys and minimize the probability of reaching IDS and true goal
    V_init = mdp.init_value_att_enu()  # Test the given value returned by maxEnt
    policy_att, V_att = mdp.getpolicy(V_init)
    V_def = mdp.policyevaluation(policy_att)
    st_visit = mdp.stVisitFre(policy_att)
#    st_visit = None
    return policy_att, V_att, V_def, st_visit, mdp


if __name__ == "__main__":
    policy, V_att, V_def, st_visit, mdp = test_att()
