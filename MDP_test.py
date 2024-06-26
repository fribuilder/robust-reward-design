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
#        filename = "MdpTransition.txt"
#        output_to_file(self.stotrans, filename)

    def getstate(self):
        # Manually define statespace
        statelist = [
            "q0",
            "q1",
        ]
        return statelist

    def getaction(self):
        A = ["a", "b", "c"]
        return A

    def gettransition(self):
        # Manually define transition
        trans = {}
        for st in self.statespace:
            trans[st] = {}
        trans["q0"]["a"] = "q1"
        trans["q0"]["b"] = "q1"
        trans["q0"]["c"] = "q1"
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
                        stotrans[st][act][self.trans[st][act]] = 1.0
            else:
                stotrans[st] = {}
                for act in self.A:
                    stotrans[st][act] = {}
                    stotrans[st][act]["Sink"] = 1.0
        
        if checkstotrans(stotrans):
            return stotrans
        else:
            print(stotrans)


    def getfakegoals(self, F=[]):
        self.F = F
        return F

    def getgoals(self, G=[]):
        self.G = G
        for st in G:
            self.U.append(st)  # we do not allow sensor placed in G.
        return G
    
    def addU(self, U):
        for st in U:
            self.U.append(st)

    def addIDS(self, IDSlist):
        for ids in IDSlist:
            self.IDS.append(ids)
        for ids in IDSlist:
            self.trans[ids] = {}
            self.stotrans[ids] = {}
            for act in self.A:
                self.trans[ids][act] = "Sink"
                self.stotrans[ids][act] = {}
                self.stotrans[ids][act]["Sink"] = 1.0

    def getcore(self, V, st, act):
        core = 0
        for st_, pro in self.stotrans[st][act].items():
            if st_ != "Sink":
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
                    for act in self.A:
                        policy[st][act] = 1.0/len(self.A)
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
    
    def getreward_att(self, r):
        reward = {}
        reward["q0"] = {}
#        reward["q0"]["a"] = 0.2284
#        reward["q0"]["b"] = 0.2284
#        reward["q0"]["c"] = 0.1432
        reward["q0"]["a"] = 1
        reward["q0"]["b"] = 1
        reward["q0"]["c"] = -3
        reward["q1"] = {}
        reward["q1"]["a"] = 1.0
        reward["q1"]["b"] = 1.0
        reward["q1"]["c"] = 1.0
        return reward
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
            1.1529,
            0,
            1.1529,
        ]  # Exclude true goal
        #        V = [0, 0, 0, 0, 0, 0, 0, 0, 0.981, 0, 0, 1, 1.07, 1.064, 1.069]
        #        V = [0, 0, 0, 0, 0, 0, 0, 0, 0.607, 0, 0, 1, 0.776, 0.678, 0.784]
        return V

    def getpolicy(self, reward, gamma=0.95):
        threshold = 0.00001
        tau = 0.01
#        r = 1  #The reward of taking action to sink state at goal state
        V= self.init_value_att()
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
            print(itcount)
            V1 = V.copy()
            for st in self.statespace:
                for act in self.A:
                    core = (reward[st][act] + gamma * self.getcore(V1, st, act)) / tau
                    Q[st][act] = np.exp(core)
                Q_s = sum(Q[st].values())
                for act in self.A:
                    policy[st][act] = Q[st][act] / Q_s
                V[self.statespace.index(st)] = tau * np.log(Q_s)
            itcount += 1
        return policy, V

    def init_value_att(self):
        V = []
        for st in self.statespace:
            V.append(0)
        return V

    def init_value_def(self):
        V = []
        for st in self.statespace:
            V.append(0)
        return V

    def policyevaluation(self, policy, gamma=0.95):
        threshold = 0.00001
        V = self.init_value_def()
        V1 = V.copy()
        r = -1
        reward = self.getreward_att(r)
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
                
            print("iteration count:", itcount)
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
        r = -1
        reward = self.getreward_att(r)
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
                        core = (reward[st][act] + gamma * self.getcore(V1, st, act)) / tau
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
        Z0 = [1, 0]
        Z_new = [1, 0]
        Z_old = Z_new.copy()
        itcount = 1
        while (
            itcount == 1
            or np.inner(
                np.array(Z_new) - np.array(Z_old), np.array(Z_new) - np.array(Z_old)
            )
            > threshold
        ):
#            print("Itcount:", itcount)
            Z_old = Z_new.copy()
            Z_new = Z0.copy()
            for st in self.statespace:
                index_st = self.statespace.index(st)
                for act in self.A:
                    for st_ in self.statespace:
                        if st in self.stotrans[st_][act].keys():
                            Z_new[index_st] += (Z_old[self.statespace.index(st_)] * policy[st_][act] * self.stotrans[st_][act][st])
            itcount += 1
        return Z_new
    
    def stactVisitFre(self, policy):
        st_visit = self.stVisitFre(policy)
        st_act_visit = {}
        for i in range(len(self.statespace)):
            st_act_visit[self.statespace[i]] ={}
            for act in self.A:
                st_act_visit[self.statespace[i]][act] = st_visit[i] * policy[self.statespace[i]][act]
        return st_act_visit


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




def output_to_file(transition, filename):
#    print("111")
    file1 = open(filename, "w")
    for st in transition.keys():
        for act in transition[st].keys():
            for st_, pro in transition[st][act].items():
                if pro != 0:
                    pro = round(pro, 2)
                    outputtext = "Starting state: " + str(st) + ", takeing action: " + str(act) + ", transfer to state: " + str(st_) + " with probability " + str(pro) + ".\n"
                    file1.write(outputtext)
    file1.close()
    
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
    IDSlist = []
    G1 = ["q1"]
    F1 = []
#    F1 = ["q12", "q14"]
    U = []
    mdp = MDP()
    mdp.getgoals(G1)
    mdp.getfakegoals(F1)
    mdp.addU(U)
    mdp.stotrans = mdp.getstochastictrans()
    mdp.addIDS(IDSlist)
#    V_init = mdp.init_value_att()   #Attacker's true value
#    V_init = mdp.init_value_att_v2()  #Maximize the probability of reaching the decoys and minimize the probability of reaching IDS and true goal
#    V_init = mdp.init_value_att_enu()  # Test the given value returned by maxEnt
    reward = mdp.getreward_att(1)
    policy_att, V_att = mdp.getpolicy(reward)
    V_def = mdp.policyevaluation(policy_att)
    st_visit = mdp.stVisitFre(policy_att)
    st_act_visit = mdp.stactVisitFre(policy_att)
#    st_visit = None
    return mdp, policy_att, V_att, V_def, st_visit, st_act_visit
#    return policy_att, V_att, V_def, st_visit, mdp


if __name__ == "__main__":
    mdp, policy, V_att, V_def, st_visit, st_act_visit = test_att()
#    policy, V_att, V_def, st_visit, mdp = test_att()
