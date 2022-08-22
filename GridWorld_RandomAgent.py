# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 02:31:08 2022

@author: 53055
"""

# -*- coding: utf-8 -*-

import numpy as np
import copy

class GridWorld:
    def __init__(self, width, height, stoPar):
        self.width = width
        self.height = height
        self.stoPar = stoPar
#        self.A = {"S":[0, 1], "N":[0, -1], "W":[-1, 0], "E":[1, 0]}
        self.A = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        self.complementA = self.getComplementA()
        self.statespace = self.getstate()
        self.gettrans()
        flag = self.checktrans()
        self.F = []
        self.G = []
        self.IDS = []
        self.U = []
    def getstate(self):
        statespace = []
        for i in range(self.width):
            for j in range(self.height):
                for p in range(self.width):
                    for q in range(self.height):
                        statespace.append(((i, j), (p, q)))
        return statespace
    
    def checkinside(self, st):
        if (st[0]< self.width and st[0]>=0) and (st[1]< self.width and st[1]>=0):
            return True
        return False
    
    def getComplementA(self):
        complementA = {}
        complementA[(0, 1)] = [(1, 0), (-1, 0)]
        complementA[(0, -1)] = [(1, 0), (-1, 0)]
        complementA[(1, 0)] = [(0, 1), (0, -1)]
        complementA[(-1, 0)] = [(0, 1), (0, -1)]
        return complementA
    
    def neighbourSt(self, st):
        st_0 = (st[0] - 1, st[1])
        st_1 = (st[0] + 1, st[1])
        st_2 = (st[0], st[1] - 1)
        st_3 = (st[0], st[1] + 1)
        neighbour = [st_0, st_1, st_2, st_3]
        dist_pro = {}
        dist_pro[st] = 0
        for st_ in neighbour:
            if self.checkinside(st_):
                dist_pro[st_] = 0.25
            else:
                dist_pro[st] += 0.25
        return dist_pro
    
    def trans_att(self, st):
        stoPar = self.stoPar
        trans = {}
        for act in self.A:
            trans[act] = {}
            trans[act][st] = 0
            tempst = tuple(np.array(st) + np.array(act))
            if self.checkinside(tempst):
                trans[act][tempst] = 1 - 2*stoPar
            else:
                trans[act][st] += 1- 2*stoPar
            for act_ in self.complementA[act]:
                tempst_ = tuple(np.array(st) + np.array(act_))
                if self.checkinside(tempst_):
                    trans[act][tempst_] = stoPar
                else:
                    trans[act][st] += stoPar
        return trans
               
    def gettrans(self):
        #Calculate transition
        trans = {}
        for st in self.statespace:
            trans[st] = {}
            att_trans = self.trans_att(st[0])
            agent_trans = self.neighbourSt(st[1])
            for act in att_trans.keys():
                trans[st][act] = {}
                for att_st, att_pro in att_trans[act].items():
                    for agent_st, agent_pro in agent_trans.items():
                        st_ = (att_st, agent_st)
                        trans[st][act][st_] = att_pro * agent_pro
        self.stotrans = trans

    
    def checktrans(self):
        for st in self.statespace:
            for act in self.A:
                if abs(sum(self.stotrans[st][act].values())-1) > 0.01:
                    print("st is:", st, " act is:", act, " sum is:", sum(self.stotrans[st][act].values()))
                    return False
        print("Transition is correct")
        return True
    
    def addFake(self, fakelist):
        #Add fake Goals
        for st in fakelist:
            for i in range(self.width):
                for j in range(self.height):
                    st_new = (st, (i, j))
                    self.F.append(st_new)
                    
                    for act in self.A:
                        self.stotrans[st_new][act] = {}
                        self.stotrans[st_new][act]["Sink"] = 1.0

    
    def addGoal(self, goallist):
        #Add true Goals
        for st in goallist:
            for i in range(self.width):
                for j in range(self.height):
                    st_new = (st, (i, j))
                    self.G.append(st_new)        
                    for act in self.A:
                        self.stotrans[st_new][act] = {}
                        self.stotrans[st_new][act]["Sink"] = 1.0

            
    def addIDS(self, IDSlist):
        #Add IDS states
        for st in IDSlist:
            for i in range(self.width):
                for j in range(self.height):
                    st_new = (st, (i, j))
                    self.IDS.append(st_new)
                    for act in self.A:
                        self.stotrans[st_new][act] = {}
                        self.stotrans[st_new][act]["Sink"] = 1.0
        for i in range(self.width):
            for j in range(self.height):
                st_caught = ((i, j), (i, j))
                if st_caught not in self.F and st_caught not in self.G and st_caught not in self.IDS:
                    self.IDS.append(st_caught)
                    for act in self.A:
                        self.stotrans[st_caught][act] = {}
                        self.stotrans[st_caught][act]["Sink"] = 1.0
                    
    def addBarrier(self, Barrierlist):
        #Add barriers in the world
        #If we want to add barriers, Add barriers first, then calculate trans, add True goal, add Fake goal, add IDS
        #No barrier in map have agent moving, maybe modify this later.
        for st in Barrierlist:
            self.statespace.remove(st)
            
    def addU(self, Ulist):
        for st in self.statespace:
            if st not in Ulist:
                self.U.append(st)
        
            
    def getcore(self, V, st, act):
        core = 0
        for st_, pro in self.stotrans[st][act].items():
            if st_ != "Sink":
                core += pro * V[self.statespace.index(st_)]
        return core
    
    def initial_reward(self, r = 1):
        reward = {}
        for st in self.statespace:
            reward[st] = {}
            if st in self.F:
                for act in self.A:
                    reward[st][act] = r
            else:
                for act in self.A:
                    reward[st][act] = 0
        return reward
        
    def getpolicy(self, reward, gamma = 0.95):
        threshold = 0.001
        tau = 0.01
        V = self.get_initial_value()
        V1 = V.copy()
        policy = {}
        Q = {}
        for st in self.statespace:
            policy[st] = {}
            Q[st] = {}
        itcount = 1
        diff = np.inf
        while itcount == 1 or np.inner(np.array(V)-np.array(V1), np.array(V)-np.array(V1)) > threshold:
            V1 = V.copy()
            for st in self.statespace:
                for act in self.A:
                    core = (reward[st][act] + gamma * self.getcore(V1, st, act)) / tau
                    Q[st][act] = np.exp(core)
                Q_s = sum(Q[st].values())
                for act in self.A:
                    policy[st][act] = Q[st][act]/Q_s
                V[self.statespace.index(st)] = tau * np.log(Q_s)
                    
            itcount += 1
            diff = np.inner(np.array(V)-np.array(V1), np.array(V)-np.array(V1))
            print("itcount:", itcount, "difference is:", diff)
        return policy, V
                    
        
    def get_initial_value(self):
        V = []
        for st in self.statespace:
            V.append(0)
        return V


    
    def policy_evaluation(self, policy, reward):
        threshold = 0.001
        gamma = 0.95
        V = self.get_initial_value()
        V1 = V.copy()
        itcount = 1
        diff = np.inf
        while (itcount == 1 or diff > threshold):
            V1 = V.copy()
            for st in self.statespace:
                temp = 0
                for act in self.A:
                    if act in policy[st].keys():
                        temp += policy[st][act] *(reward[st][act] + gamma * self.getcore(V1, st, act))
                V[self.statespace.index(st)] = temp
            itcount += 1
            diff = np.inner(np.array(V) - np.array(V1), np.array(V) - np.array(V1))
            print("iteration count:", itcount, "difference is:", diff)
        return V
    
    def stVisitFre(self, policy, init_dist):
        threshold = 0.0001
        Z0 = init_dist
        Z_new = Z0.copy()
        Z_old = Z_new.copy()
        itcount = 1
#        sinkst = self.F + self.G + self.IDS
#        print(sinkst)
        while itcount == 1 or np.inner(np.array(Z_new)-np.array(Z_old), np.array(Z_new)-np.array(Z_old)) > threshold:
#            print(itcount)
            Z_old = Z_new.copy()
            Z_new = Z0.copy()
            for st in self.statespace:
                index_st = self.statespace.index(st)
                for act in self.A:
                    for st_ in self.statespace:
                        if st in self.stotrans[st_][act].keys():
                            Z_new[index_st] += Z_old[self.statespace.index(st_)] * policy[st_][act] * self.stotrans[st_][act][st]
            
            itcount += 1
#            print(Z)
#            diff = np.subtract(Z_new, Z_old)
#            diff_list = list(diff)
#            print(diff_list)
        return Z_new
    
    def stactVisitFre(self, policy, init_dist):
        Z = self.stVisitFre(policy, init_dist)
        st_act_visit = {}
        for i in range(len(self.statespace)):
            st_act_visit[self.statespace[i]] ={}
            for act in self.A:
                st_act_visit[self.statespace[i]][act] = Z[i] * policy[self.statespace[i]][act]
        return st_act_visit

    def getreward_att(self, r):
        reward = {}
        for st in self.statespace:
            reward[st] = {}
            if st not in self.G and st not in self.F:
                for act in self.A:
                    reward[st][act] = 0
            else:
                for act in self.A:
                    reward[st][act] = r
        return reward
            
    def getworstcase_att(self, r):
        reward = {}
        for st in self.statespace:
            reward[st] = {}
            if st not in self.F:
                for act in self.A:
                    reward[st][act] = 0
            else:
                for act in self.A:
                    reward[st][act] = r
        return reward
    
    def init_dist(self, st):
        init_distribution = np.zeros(len(self.statespace))
        init_list = []
        for i in range(len(self.statespace)):
            if self.statespace[i][0] == st:
                init_list.append(i)
        dist_pro = 1/len(init_list)
        for index in init_list:
            init_distribution[index] = dist_pro
        return init_distribution
            


def createGridWorldAgent():
    gridworld = GridWorld(6, 6, 0.1)
    goallist = [(3, 4)]
    barrierlist = []
    gridworld.addBarrier(barrierlist)
    fakelist = [(1, 4), (4, 5)]
    IDSlist = [(0, 4), (1, 2), (2, 3), (3, 3), (5, 4)]
#    IDSlist = [(6, 5), (4, 5)]
#    fakelist = [(4, 6), (7, 4)]
    Ulist = []  #This U is the states that can place sensors
    for i in range(6):
        for j in range(2, 4):
            Ulist.append((i, j))
    gridworld.addU(Ulist)
    gridworld.gettrans()
    gridworld.addFake(fakelist)
    gridworld.addGoal(goallist)
    gridworld.addIDS(IDSlist)
#    V_0 = gridworld.init_preferred_attack_value()
    reward = gridworld.initial_reward()
#    print(reward)
    policy, V = gridworld.getpolicy(reward)

    V_def = gridworld.policy_evaluation(policy, reward)
    return gridworld, V_def, policy    
            
    
if __name__ == "__main__":
#    gridworld, V, policy = createGridWorld()
    gridworld, V_def, policy = createGridWorldAgent()
    state = (2, 0)
    init_dist = gridworld.init_dist(state)
#    Z = gridworld.stVisitFre(policy, init_dist)
    Z_act = gridworld.stactVisitFre(policy, init_dist)
#    print(V_def[14], Z[20], Z[48])
#    print(Z[35], Z[54])