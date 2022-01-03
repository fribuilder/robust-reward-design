# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 14:37:08 2021

@author: 53055
"""

import numpy as np
import copy

class GridWorld:
    def __init__(self, width, height, stoPar):
        self.width = width
        self.height = height
        self.stoPar = stoPar
#        self.A = {"S":[0, 1], "N":[0, -1], "W":[-1, 0], "E":[1, 0]}
        self.A = [(0, 1), (0, -1), (-1, 0), (1, 0)]
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
                statespace.append((i, j))
        return statespace
    
    def checkinside(self, st):
        if st in self.statespace:
            return True
        return False
    
    def gettrans(self):
        #Calculate transition
        stoPar = self.stoPar
        trans = {}
        for st in self.statespace:
            trans[st] = {}
            for act in self.A:
                trans[st][act] = {}
                trans[st][act][st] = 0
                tempst = tuple(np.array(st) + np.array(act))
                if self.checkinside(tempst):
                    trans[st][act][tempst] = 1 - 3*stoPar
                else:
                    trans[st][act][st] += 1- 3*stoPar
                for act_ in self.A:
                    if act_ != act:
                        tempst_ = tuple(np.array(st) + np.array(act_))
                        if self.checkinside(tempst_):
                            trans[st][act][tempst_] = stoPar
                        else:
                            trans[st][act][st] += stoPar
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
            self.F.append(st)
            for act in self.A:
                self.stotrans[st][act] = {}
                self.stotrans[st][act][st] = 1.0

    
    def addGoal(self, goallist):
        #Add true Goals
        for st in goallist:
            self.G.append(st)
            for act in self.A:
                self.stotrans[st][act] = {}
                self.stotrans[st][act][st] = 1.0

            
    def addIDS(self, IDSlist):
        #Add IDS states
        for st in IDSlist:
            self.IDS.append(st)
            for act in self.A:
                self.stotrans[st][act] = {}
                self.stotrans[st][act][st] = 1.0

        
    def addBarrier(self, Barrierlist):
        #Add barriers in the world
        #If we want to add barriers, Add barriers first, then calculate trans, add True goal, add Fake goal, add IDS
        for st in Barrierlist:
            self.statespace.remove(st)
            
    def addU(self, Ulist):
        for st in self.statespace:
            if st not in Ulist:
                self.U.append(st)
        
            
    def getcore(self, V, st, act):
        core = 0
        for st_, pro in self.stotrans[st][act].items():
            core += pro * V[self.statespace.index(st_)]
        return core
            
    def getpolicy(self, V, gamma = 0.95):
        threshold = 0.00001
        tau = 1
        V1 = V.copy()
        policy = {}
        Q = {}
        for st in self.statespace:
            policy[st] = {}
            Q[st] = {}
        itcount = 1
        while itcount == 1 or np.inner(np.array(V)-np.array(V1), np.array(V)-np.array(V1)) > threshold:
            V1 = V.copy()
            for st in self.statespace:
                if st not in self.F and st not in self.G and st not in self.IDS:
                    for act in self.A:
                        core = gamma * self.getcore(V1, st, act) / tau
                        Q[st][act] = np.exp(core)
                    Q_s = sum(Q[st].values())
                    for act in self.A:
                        policy[st][act] = Q[st][act]/Q_s
                    V[self.statespace.index(st)] = tau * np.log(Q_s)
                else:
                    policy[st] = {}
                    for act in self.A:
                        policy[st][act] = 0.25
                    
                itcount += 1
        return V, policy
                    
        
    def get_initial_value(self):
        V = []
        for st in self.statespace:
            if st in self.F:
                V.append(100)
            if st in self.G:
                V.append(100)
            if st not in self.F and st not in self.G:
                V.append(0)
        return V
    
    def init_value_def(self):
        V = []
        for st in self.statespace:
            if st in self.IDS:
                V.append(0)
            elif st in self.F:
                V.append(0)
            elif st in self.G:
                V.append(-100)
            else:
                V.append(0)
        return V
    
    def init_preferred_attack_value(self):
        V = []
        for st in self.statespace:
            if st in self.F:
                V.append(100)
            else:
                V.append(0)
        return V
    
    def policy_evaluation(self, policy):
        threshold = 0.00001
        gamma = 0.95
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
    
    def stvisitFreq(self, policy):
        threshold = 0.0001
        Z0 = np.zeros(len(self.statespace))
        Z0[14] = 1
        Z_new = Z0.copy()
        Z_old = Z_new.copy()
        itcount = 1
        sinkst = self.F + self.G + self.IDS
        print(sinkst)
        while itcount == 1 or np.inner(np.array(Z_new)-np.array(Z_old), np.array(Z_new)-np.array(Z_old)) > threshold:
            print(itcount)
            Z_old = Z_new.copy()
            Z_new = Z0.copy()
            for st in self.statespace:
                index_st = self.statespace.index(st)
                if st not in sinkst:
                    for act in self.A:
                        for st_ in self.statespace:
                            if st in self.stotrans[st_][act].keys():
                                Z_new[index_st] += Z_old[self.statespace.index(st_)] * policy[st_][act] * self.stotrans[st_][act][st]
                else:
                    for act in self.A:
                        for st_ in self.statespace:
                            if st_ not in sinkst:
                                if st in self.stotrans[st_][act].keys():
                                    Z_new[index_st] += Z_old[self.statespace.index(st_)] * policy[st_][act] * self.stotrans[st_][act][st]
            
            itcount += 1
        return Z_new

def createGridWorld():
    gridworld = GridWorld(6, 6, 0.05)
    goallist = [(4, 5)]
    fakelist = [(1, 4), (5, 4)]
    IDSlist = [(1, 1), (4, 3)]
    gridworld.addFake(fakelist)
    gridworld.addGoal(goallist)
    gridworld.addIDS(IDSlist)
    V_0= gridworld.get_initial_value()
#    V_0[10] = 103.87
#    V_0[34] = 103.992
    V_0[10] = 94.2
    V_0[34] = 61.1
    V, policy = gridworld.getpolicy(V_0)
    return gridworld, V, policy

def createGridWorldBarrier():
    gridworld = GridWorld(8, 8, 0.05)
    goallist = [(2, 7), (6, 6)]
    barrierlist = [(1, 5), (1, 6), (2, 6), (5, 1), (6, 1), (6, 2)]
    gridworld.addBarrier(barrierlist)
    fakelist = []
    IDSlist = [(0, 5), (3, 5), (5, 4)]
    fakelist = [(4, 6), (7, 4)]
#    IDSlist = [(3, 4), (5, 3)]
    Ulist = []  #This U is the states that can place sensors
    for i in range(8):
        for j in range(2, 6):
            Ulist.append((i, j))
    gridworld.addU(Ulist)
    gridworld.gettrans()
    gridworld.addFake(fakelist)
    gridworld.addGoal(goallist)
    gridworld.addIDS(IDSlist)
#    V_0 = gridworld.get_initial_value()
    V_0 = gridworld.init_preferred_attack_value()
#    V_0[35] = 97.2567
#    V_0[54] = 97.5303
    V, policy = gridworld.getpolicy(V_0)
    V_def = gridworld.policy_evaluation(policy)
    return gridworld, V_def, policy
    
if __name__ == "__main__":
#    gridworld, V, policy = createGridWorld()
    gridworld, V_def, policy = createGridWorldBarrier()
    Z = gridworld.stvisitFreq(policy)