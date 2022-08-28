import numpy as np
from itertools import product
#from MDP import MDP
from GridWorld import GridWorld, createGridWorld, createGridWorldBarrier
import GridWorldV2
import GridWorld_RandomAgent
from MDP_V2 import MDP
import MDP_test

"""
All the state and transition should in the form of index

"""

class World:
    def __init__(self, mdp):
        self.statespace, self.st_dict = self.convert_statespace(mdp.statespace)
        self.actionspace, self.act_dict = self.convert_actionspace(mdp.A)
        self.transition = self.convert_transition(mdp.stotrans)
#        self.transition = self.convert_transition(mdp.trans)  #Gridworld
        self.F = self.convert_fakegoals(mdp.F)
        self.G = self.convert_goals(mdp.G)
        self.Sink = self.convert_sink(mdp.IDS)
#        self.F = self.convert_fakegoals(mdp.Fake)  #Gridworld
#        self.G = self.convert_goals(mdp.Goal)
#        self.Sink = self.convert_sink(mdp.IDS)
        
    def convert_statespace(self, statespace):
        index_st = []
        index_dict = {}
        for i in range(len(statespace)):
            index_st.append(i)
            index_dict[statespace[i]] = i
        return index_st, index_dict
    
    def convert_actionspace(self, actionspace):
        index_act = []
        index_dict = {}
        for i in range(len(actionspace)):
            index_act.append(i)
            index_dict[actionspace[i]] = i
        return index_act, index_dict
    
    def convert_transition(self, stotrans):
        transition = np.zeros((len(self.statespace), len(self.statespace), len(self.actionspace)))
        for st in stotrans.keys():
            for act in stotrans[st].keys():
                for st_, pro in stotrans[st][act].items():
                    if st_ != 'Sink':
                        transition[self.st_dict[st]][self.st_dict[st_]][self.act_dict[act]] = pro
        return transition
    
    def convert_fakegoals(self, F):
        fake_index = []
        for st in F:
            fake_index.append(self.st_dict[st])
        return fake_index
    
    def convert_goals(self, G):
        goal_index = []
        for st in G:
            goal_index.append(self.st_dict[st])
        return goal_index
    
    def convert_policy(self, policy):
        policy_mat = np.zeros((len(self.statespace), len(self.actionspace)))
        for st in policy.keys():
            for act, pro in policy[st].items():
                policy_mat[self.st_dict[st]][self.act_dict[act]] = pro
                
        return policy_mat
    
    def convert_sink(self, IDS):
        ids_index = []
        for st in IDS:
            ids_index.append(self.st_dict[st])
        return ids_index
        
    def statevisiting(self, frequency):
        self.statevisit = frequency
    
    def stateActVisiting(self, frequency):
        self.stateactVisiting = []
        for i in frequency.keys():
            for j in frequency[i].keys():
                self.stateactVisiting.append(frequency[i][j])
def state_features(world):
    """
    The state features, assign each state with a feature
    """

    state_features = np.zeros((len(world.statespace), len(world.F)))
    for i in range(len(world.F)):
        state_features[world.F[i]][i] = 1
#    state_features = state_features * 100
    return state_features

def state_act_feature(world):
    """
    The state act feature, assign each state action with a feature
    """
    state_act_feature = np.identity(len(world.stateactVisiting))
    state_act_feature[2][2] = -1
    return state_act_feature

def state_act_feature_manual(world):
    state_act_feature = np.zeros((len(world.statespace) * len(world.actionspace), 6))
    state_act_feature[32][0] = -1
    state_act_feature[33][1] = -1
    state_act_feature[42][2] = -1
    state_act_feature[43][3] = -1
    state_act_feature[52][-2] = 1
    state_act_feature[56][-1] = 1
    return state_act_feature

def state_act_feature_manual_list(world, modifylist):
    """
    For gridworld, 0 is right, 1 is left, 2 is down, 3 is up
    """
    state_act_feature = np.zeros((len(world.statespace) * len(world.actionspace), len(modifylist)))
    pre = len(modifylist) - len(world.F)
    for i in range(pre):
        state_act_feature[modifylist[i]][i] = -1
    for i in range(pre, len(modifylist)):
        state_act_feature[modifylist[i]][i] = 1
    return state_act_feature
        
def state_act_feature_walkingAgent(world, gridworld, F):
    state_act_feature = np.zeros((len(world.statespace) * len(world.actionspace), len(F)))
    for i in range(len(F)):
        for j in range(len(world.statespace)):
            if gridworld.statespace[j][0] == F[i]:
                state_act_feature[j*4][i] = 1
                state_act_feature[j*4+1][i] = 1
                state_act_feature[j*4+2][i] = 1
                state_act_feature[j*4+3][i] = 1
    return state_act_feature

def state_act_feature_decoyonly(world, gridworld):
    state_act_feature = np.zeros((len(world.statespace) * len(world.actionspace), len(gridworld.F)))
    for i in range(len(gridworld.F)):
        state_act_feature[world.st_dict[gridworld.F[i]] * len(world.actionspace)][i] = 1
    return state_act_feature
                
        

def test_att():
    """
    In this test function, the agent is maximizing the probability of reaching the decoys
    while avoiding the IDS placements and the true goal
    """
    IDSlist = ["q8"]
    G1 = ["q11"]
    F1 = ["q12", "q14"]
#    F1 = ["q8", "q12", "q13", "q14"]
    mdp = MDP()
    mdp.getgoals(G1)
    mdp.getfakegoals(F1)
    mdp.stotrans = mdp.getstochastictrans()
    mdp.addIDS(IDSlist)
    V_init = mdp.init_value_att()
    policy_att, V_att = mdp.getpolicy(V_init)
    return policy_att, V_att, mdp

def test():
#    mdp = test_mdp()
#    policy_wstatt, V_wstatt = mdp.getpolicy_det_min()
    policy_wstatt, V_wstatt, mdp = test_att()
    world = World(mdp)
    policy = world.convert_policy(policy_wstatt)
    st_fre = mdp.stVisitFre(policy_wstatt)
    world.statevisiting(st_fre)
    return world, mdp, policy

def test_gridworld():
    """
    This function is used to test gridworld
    """
#    gridworld, V, policy = createGridWorld()
    gridworld, V, policy = createGridWorldBarrier()
    world_convert = World(gridworld)
    policy_convert = world_convert.convert_policy(policy)
#    Z = gridworld.stvisitFreq(policy)
#    world_convert.statevisiting(Z)
    return world_convert, gridworld, policy_convert

def test_gridworldV2():
    """
    This function is used to test gridworld V2
    """
    gridworld, V, policy = GridWorldV2.createGridWorldBarrier()
    world_convert = World(gridworld)
    policy_convert = world_convert.convert_policy(policy)
    return world_convert, gridworld, policy_convert

def test_gridworld_new():
    """
    This function is used to test gridworld V2 new map
    """
    gridworld, V, policy = GridWorldV2.createGridWorldBarrier_new()
    world_convert = World(gridworld)
    policy_convert = world_convert.convert_policy(policy)
#    Z = gridworld.stvisitFreq(policy)
    Z_act = gridworld.stactVisitFre(policy)
    world_convert.stateActVisiting(Z_act)
    return world_convert, gridworld, policy_convert

def test_gridworld_new2():
    """
    This function is used to test gridworld V2 new map
    """
    gridworld, V, policy = GridWorldV2.createGridWorldBarrier_new2()
    world_convert = World(gridworld)
    policy_convert = world_convert.convert_policy(policy)
#    Z = gridworld.stvisitFreq(policy)
    Z_act = gridworld.stactVisitFre(policy)
    world_convert.stateActVisiting(Z_act)
    return world_convert, gridworld, policy_convert

def test_gridworld_agent():
    """
    This function is used to test gridworld with random walking agent
    """
    gridworld, V, policy = GridWorld_RandomAgent.createGridWorldAgent()
    world_convert = World(gridworld)
    policy_convert = world_convert.convert_policy(policy)
    state = (2, 0)
    init_dist = gridworld.init_dist(state)
    Z_act = gridworld.stactVisitFre(policy, init_dist)
    world_convert.stateActVisiting(Z_act)
    return world_convert, gridworld, policy_convert

def test_mdpV2():
    IDSlist = ["q9"]
    G1 = ["q12"]
#    F1 = []
    F1 = ["q13", "q14"]
    U = ["q0", "q1", "q2", "q3", "q4", "q12", "q13", "q14"]
    mdp = MDP()
    mdp.getgoals(G1)
    mdp.getfakegoals(F1)
    mdp.addU(U)
    mdp.stotrans = mdp.getstochastictrans()
    mdp.addIDS(IDSlist)
#    reward = mdp.getreward_att(1)
    reward = mdp.getworstcase_att(1)
#    reward_value = -0.5
#    reward = mdp.modifystactreward(reward, reward_value)
    policy_att, V_att = mdp.getpolicy(reward)
#    V_def = mdp.policyevaluation(policy_att)
#    st_visit = mdp.stVisitFre(policy_att)
    st_act_visit = mdp.stactVisitFre(policy_att)
    world = World(mdp)
    policy = world.convert_policy(policy_att)
    world.stateActVisiting(st_act_visit)
    return world, mdp, policy

def test_mdpSmall():
    IDSlist = []
    G1 = ["q1"]
    F1 = []
#    F1 = ["q12", "q14"]
    U = []
    mdp = MDP_test.MDP()
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
    st_act_visit = mdp.stactVisitFre(policy_att)
    world = World(mdp)
    policy = world.convert_policy(policy_att)
    world.stateActVisiting(st_act_visit)
    return world, mdp, policy

if __name__ == "__main__":
#    world, policy = test()
#    next_p = world.transition[10, :, 0]
#    next_state = np.random.choice(world.statespace, p=next_p)
#    print(next_state)
#    world, gridworld, policy = test_gridworld()
#    state_feature = state_features(world)
#    print(state_feature)
#    world, gridworld, exp_policy = test_mdpV2()
#    world, gridworld, exp_policy = test_mdpSmall()
    world, gridworld, exp_policy = test_gridworld_new2()
    state_act_feature = state_act_feature_decoyonly(world, gridworld)
#    modifylist = [112, 113, 114, 115, 40, 116]
    
    
#    world, gridworld, exp_policy = test_gridworld_agent()   #Code for walking agent
#    F = [(1, 4), (4, 5)]   #Code for walking agent
#    state_feature = state_act_feature_walkingAgent(world, gridworld, F)  #Code for walking agent