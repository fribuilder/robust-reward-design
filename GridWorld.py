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
#        self.A = {"S":[0, 1], "N":[0, -1], "W":[-1, 0], "E":[1, 0]}
        self.A = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        self.statespace = self.getstate()
        self.trans = self.gettrans(stoPar)
        flag = self.checktrans()
        self.Fake = []
        self.Goal = []
        self.IDS = []
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
    
    def gettrans(self, stoPar):
        #Calculate transition
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
        return trans
    
    def checktrans(self):
        for st in self.statespace:
            for act in self.A:
                if abs(sum(self.trans[st][act].values())-1) > 0.01:
                    print("st is:", st, " act is:", act, " sum is:", sum(self.trans[st][act].values()))
                    return False
        print("Transition is correct")
        return True
    
    def addFake(self, fakelist):
        #Add fake Goals
        for st in fakelist:
            self.Fake.append(st)
            for act in self.A:
                self.trans[st][act] = {}
                self.trans[st][act][st] = 1.0

    
    def addGoal(self, goallist):
        #Add true Goals
        for st in goallist:
            self.Goal.append(st)
            for act in self.A:
                self.trans[st][act] = {}
                self.trans[st][act][st] = 1.0

            
    def addIDS(self, IDSlist):
        #Add IDS states
        for st in IDSlist:
            self.IDS.append(st)
            for act in self.A:
                self.trans[st][act] = {}
                self.trans[st][act][st] = 1.0

        
    def addBarrier(self, Barrierlist):
        #Add barriers in the world
        #If we want to add barriers, Add barriers first, then calculate trans, add True goal, add Fake goal, add IDS
        for st in Barrierlist:
            self.statespace.reomove(st)
            
    def getcore(self, V, st, act):
        core = 0
        for st_, pro in self.trans[st][act].items():
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
                if st not in self.Fake and st not in self.Goal and st not in self.IDS:
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
            if st in self.Fake:
                V.append(100)
            if st in self.Goal:
                V.append(100)
            if st not in self.Fake and st not in self.Goal:
                V.append(0)
        return V
    
    def stvisitFreq(self, policy):
        threshold = 0.0001
        Z0 = np.zeros(len(self.statespace))
        Z0[18] = 1
        Z_new = Z0.copy()
        Z_old = Z_new.copy()
        itcount = 1
        sinkst = self.Fake + self.Goal + self.IDS
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
                            if st in self.trans[st_][act].keys():
                                Z_new[index_st] += Z_old[self.statespace.index(st_)] * policy[st_][act] * self.trans[st_][act][st]
                else:
                    for act in self.A:
                        for st_ in self.statespace:
                            if st_ not in sinkst:
                                if st in self.trans[st_][act].keys():
                                    Z_new[index_st] += Z_old[self.statespace.index(st_)] * policy[st_][act] * self.trans[st_][act][st]
            
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
    
if __name__ == "__main__":
    gridworld, V, policy = createGridWorld()
    Z = gridworld.stvisitFreq(policy)