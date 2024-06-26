# -*- coding: utf-8 -*-

import numpy as np
import copy
import policyImprovement

class GridWorld:
    def __init__(self, width, height, stoPar):
        self.width = width
        self.height = height
        self.stoPar = stoPar
#        self.A = {"S":[0, 1], "N":[0, -1], "W":[-1, 0], "E":[1, 0]}
        self.A = [(0, 1), (0, -1), (-1, 0), (1, 0)] #E, W, S, N
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
                statespace.append((i, j))
        return statespace
    
    def checkinside(self, st):
        if st in self.statespace:
            return True
        return False
    
    def getComplementA(self):
        complementA = {}
        complementA[(0, 1)] = [(1, 0), (-1, 0)]
        complementA[(0, -1)] = [(1, 0), (-1, 0)]
        complementA[(1, 0)] = [(0, 1), (0, -1)]
        complementA[(-1, 0)] = [(0, 1), (0, -1)]
        return complementA
        
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
                    trans[st][act][tempst] = 1 - 2*stoPar
                else:
                    trans[st][act][st] += 1- 2*stoPar
                for act_ in self.complementA[act]:
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
#        print("Transition is correct")
        return True
    
    def addFake(self, fakelist):
        #Add fake Goals
        for st in fakelist:
            self.F.append(st)
            for act in self.A:
                self.stotrans[st][act] = {}
                self.stotrans[st][act]["Sink"] = 1.0

    
    def addGoal(self, goallist):
        #Add true Goals
        for st in goallist:
            self.G.append(st)
            for act in self.A:
                self.stotrans[st][act] = {}
                self.stotrans[st][act]["Sink"] = 1.0

            
    def addIDS(self, IDSlist):
        #Add IDS states
        for st in IDSlist:
            self.IDS.append(st)
            for act in self.A:
                self.stotrans[st][act] = {}
                self.stotrans[st][act]["Sink"] = 1.0

        
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

    def initial_reward_withoutDecoy(self, r = 1):
        reward = {}
        for st in self.statespace:
            reward[st] = {}
            if st in self.G:
                for act in self.A:
                    reward[st][act] = r
            else:
                for act in self.A:
                    reward[st][act] = 0
        return reward
    
    def initial_reward_manual(self, rlist):
        reward = {}
        for st in self.statespace:
            reward[st] = {}
            if st in self.G:
                for act in self.A:
                    reward[st][act] = 1
            elif st in self.F:
                for act in self.A:
                    reward[st][act] = rlist[self.F.index(st)]
            else:
                for act in self.A:
                    reward[st][act] = 0
        return reward
        
    def getpolicy(self, reward, gamma = 0.95):
        threshold = 0.00001
        tau = 0.01
        V = self.get_initial_value()
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
                for act in self.A:
                    core = (reward[st][act] + gamma * self.getcore(V1, st, act)) / tau
                    Q[st][act] = np.exp(core)
                Q_s = sum(Q[st].values())
                
                for act in self.A:
                    policy[st][act] = Q[st][act]/Q_s
                V[self.statespace.index(st)] = tau * np.log(Q_s)
                    
                itcount += 1
#                print(itcount)
            for st in self.statespace:
                if st in self.F:
                    policy[st] = {}
                    policy[st][self.A[0]] = 0.9999997
                    policy[st][self.A[1]] = 0.0000001
                    policy[st][self.A[2]] = 0.0000001
                    policy[st][self.A[3]] = 0.0000001
        return policy, V
    
    def getpolicy_det(self, reward, gamma = 0.95):
        threshold = 0.00001
        V = self.get_initial_value()
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
                max_value = -1
                max_act = None
                for act in self.A:
                    temp_value = reward[st][act] + gamma * self.getcore(V1, st, act)
                    if temp_value > max_value:
                        max_act = act
                        max_value = temp_value
                policy[st] = {}
                for act in self.A:
                    policy[st][act] = 0
                policy[st][max_act] = 1.0
                V[self.statespace.index(st)] = max_value
            itcount += 1
        return policy, V
        
    def get_initial_value(self):
        V = []
        for st in self.statespace:
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
                V.append(-1)
            else:
                V.append(0)
        return V
    
    def init_preferred_attack_value(self):
        V = []
        for st in self.statespace:
            if st in self.F:
                V.append(1)
            else:
                V.append(0)
        return V
    
    def policy_evaluation(self, policy, reward):
        threshold = 0.00001
        gamma = 0.95
        V = self.get_initial_value()
        V1 = V.copy()
        itcount = 1
        while (
            itcount == 1
            or np.inner(np.array(V) - np.array(V1), np.array(V) - np.array(V1))
            > threshold
        ):
            V1 = V.copy()
            for st in self.statespace:
                temp = 0
                for act in self.A:
                    if act in policy[st].keys():
                        temp += policy[st][act] * (reward[st][act] + gamma * self.getcore(V1, st, act))
                V[self.statespace.index(st)] = temp
            #            print("iteration count:", itcount)
            itcount += 1
        return V
    
    def stVisitFre(self, policy):
        threshold = 1e-8
        gamma = 0.95
        '''Z0 is also initial state'''
        Z0 = np.zeros(len(self.statespace))
#        Z0[9] = 1
        Z0[12] = 1  #6*6 case   #12 corresponds to the scenario in ppt
#        Z0[51] = 1  #10*10 case
#        Z0[30] = 1 #10*10 case
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
                            Z_new[index_st] += gamma * Z_old[self.statespace.index(st_)] * policy[st_][act] * self.stotrans[st_][act][st]
            
            itcount += 1
#            print(Z)
#            diff = np.subtract(Z_new, Z_old)
#            diff_list = list(diff)
#            print(diff_list)
        return Z_new
    
    def stactVisitFre(self, policy):
        Z = self.stVisitFre(policy)
        st_act_visit = {}
        for i in range(len(self.statespace)):
            st_act_visit[self.statespace[i]] ={}
            for act in self.A:
                st_act_visit[self.statespace[i]][act] = Z[i] * policy[self.statespace[i]][act]
        return st_act_visit

    def getreward_att(self, r = 1):
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
    
    def getreward_def(self, r = 1):
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
    
    def randomPolicy(self):
        lenA = len(self.A)
        policy = {}
        for st in self.statespace:
            policy[st] = {}
            for act in self.A:
                policy[st][act] = 1/lenA
        return policy
    
    def randomPolicy_1(self):
        policy = {}
        for st in self.statespace:
            policy[st] = {}
            policy[st][(0, 1)] = 0.1
            policy[st][(0, -1)] = 0.4
            policy[st][(-1, 0)] = 0.2
            policy[st][(1, 0)] = 0.3
        return policy
    
    def randomPolicy_2(self):
        policy = {}
        for st in self.statespace:
            policy[st] = {}
            policy[st][(0, 1)] = 0.3
            policy[st][(0, -1)] = 0.3
            policy[st][(-1, 0)] = 0.3
            policy[st][(1, 0)] = 0.1
        return policy
  

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
    policy, V = gridworld.getpolicy(V_0)  #change V to reward
    return gridworld, V, policy

def createGridWorldBarrier():
    gridworld = GridWorld(8, 8, 0.1)
    goallist = [(2, 7), (6, 6)]
    barrierlist = [(1, 5), (1, 6), (2, 6), (5, 1), (6, 1), (6, 2)]
    gridworld.addBarrier(barrierlist)
    fakelist = []
    IDSlist = [(0, 5), (3, 5), (5, 5), (7, 5)]
#    IDSlist = [(6, 5), (4, 5)]
#    fakelist = [(4, 6), (7, 4)]
    fakelist = [(7, 4)]
    Ulist = []  #This U is the states that can place sensors
    for i in range(8):
        for j in range(2, 6):
            Ulist.append((i, j))
    gridworld.addU(Ulist)
    gridworld.gettrans()
    gridworld.addFake(fakelist)
    gridworld.addGoal(goallist)
    gridworld.addIDS(IDSlist)
#    V_0 = gridworld.init_preferred_attack_value()
    V_0 = gridworld.get_initial_value()
    V_0[35] = 100.8890
    V_0[54] = 91.5199
    policy, V = gridworld.getpolicy(V_0)   #Change V to reward
    V_def = gridworld.policy_evaluation(policy)
    return gridworld, V_def, policy

# def createGridWorldBarrier_new():
#     #Not use this any more
#     gridworld = GridWorld(6, 6, 0.1)
#     goallist = [(5, 4)]
#     barrierlist = [(0, 1), (0, 2), (0, 3), (3, 1), (3, 2), (2, 2), (4, 2)]
#     gridworld.addBarrier(barrierlist)
#     fakelist = [(0, 5), (3, 5)]
#     IDSlist = [(3, 3)]
# #    IDSlist = [(6, 5), (4, 5)]
# #    fakelist = [(4, 6), (7, 4)]
#     Ulist = []  #This U is the states that can place sensors
#     for i in range(6):
#         for j in range(2, 4):
#             Ulist.append((i, j))
#     gridworld.addU(Ulist)
#     gridworld.gettrans()
#     gridworld.addFake(fakelist)
#     gridworld.addGoal(goallist)
#     gridworld.addIDS(IDSlist)
# #    V_0 = gridworld.init_preferred_attack_value()
#     reward = gridworld.initial_reward()
# #    print(reward)
#     policy, V = gridworld.getpolicy(reward)
# #    print(V)
#     V_def = gridworld.policy_evaluation(policy)
#     return gridworld, V_def, policy

def createGridWorldBarrier_new2():
    gridworld = GridWorld(6, 6, 0.1)
    goallist = [(3, 4), (5, 0)] 
#    barrierlist = [(0, 1), (0, 2), (0, 3), (3, 1), (3, 2), (2, 2), (4, 2)]
    barrierlist = []
    gridworld.addBarrier(barrierlist)
    fakelist = [(1, 4), (4, 5)] #case 1
    # fakelist = [(0, 2), (5, 3)] #case 2
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
    # reward = gridworld.getreward_def(1)   #Cant use this as the initial reward
    # reward = gridworld.initial_reward()
    reward = gridworld.initial_reward_withoutDecoy()
    reward = gridworld.initial_reward_manual([2.016, 1.826])
    # print(reward) change the print to annotation
#    print(reward)
#    policy, V = gridworld.getpolicy(reward)
    policy, V = gridworld.getpolicy_det(reward)
#    policy = gridworld.randomPolicy()
    reward_d = gridworld.getreward_def(1)
    # print(reward_d) change the print to annotation
    # print(V)
    V_def = gridworld.policy_evaluation(policy, reward_d)
    return gridworld, V_def, policy    

def createGridWorldBarrier_new3():
    gridworld = GridWorld(10, 10, 0.1)
    goallist = [(0, 7), (5, 7), (9, 4)]
    # goallist = [(5, 7), (1, 5)]
    barrierlist = []
    gridworld.addBarrier(barrierlist)
    fakelist = [(2, 8), (6, 8), (7, 5)]
    # fakelist = [(2, 8), (6, 8), (7, 4)]
    IDSlist = [(0, 4), (3, 3), (4, 3), (4, 4), (7, 3), (7, 7), 
               (7, 8), (8, 2), (8, 7), (9, 5), (9, 6)]
    
    Ulist = []
    for i in range(10):
        for j in range(3, 7):
            Ulist.append((i, j))
    gridworld.addU(Ulist)
    gridworld.gettrans()
    gridworld.addFake(fakelist)
    gridworld.addGoal(goallist)
    gridworld.addIDS(IDSlist)
    reward = gridworld.initial_reward(3)
    # reward = gridworld.getreward_def(3)
    reward_d = gridworld.getreward_def(1)
    # reward = gridworld.initial_reward_withoutDecoy(1)
    reward = gridworld.initial_reward_manual([1.4287, 0, 1.4664])
    # reward = gridworld.initial_reward_manual([2.54, 0, 2.5797])
    # reward = gridworld.getreward_att()
    # policy, V = gridworld.getpolicy_det(reward)
    # policy = gridworld.randomPolicy()   
    policy, V = gridworld.getpolicy(reward)
    # V_def_e = policyImprovement.policyEval_Ent(gridworld, reward, policy)
    # policy_improve, V_improve = policyImprovement.policyImpro(gridworld, V_def_e, reward)
    # V_improve = policyImprovement.policyEval(gridworld, reward_d, policy_improve)
    # print(V_improve[30])
    V_def = gridworld.policy_evaluation(policy, reward_d)
    V_def_e = policyImprovement.policyEval_Ent(gridworld, reward_d, policy)
    print(V_def[30], V_def_e[30])
    return gridworld, V_def, policy
    
    
if __name__ == "__main__":
#    gridworld, V, policy = createGridWorld()
    gridworld, V_def, policy = createGridWorldBarrier_new3()
    Z = gridworld.stVisitFre(policy)
    Z_act = gridworld.stactVisitFre(policy)
#    print(V_def[14], Z[20], Z[48])
#    print(Z[35], Z[54])