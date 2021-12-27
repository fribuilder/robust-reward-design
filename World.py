# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 11:34:19 2021

@author: 53055
"""

import numpy as np
from itertools import product
from MDP import MDP
from GridWorld import GridWorld, createGridWorld

"""
All the state and transition should in the form of index

"""

class World:
    def __init__(self, mdp):
        self.statespace, self.st_dict = self.convert_statespace(mdp.statespace)
        self.actionspace, self.act_dict = self.convert_actionspace(mdp.A)
#        self.transition = self.convert_transition(mdp.stotrans)
        self.transition = self.convert_transition(mdp.trans)  #Gridworld
#        self.F = self.convert_fakegoals(mdp.F)
#        self.G = self.convert_goals(mdp.G)
#        self.Sink = self.convert_sink(mdp.IDS)
        self.F = self.convert_fakegoals(mdp.Fake)  #Gridworld
        self.G = self.convert_goals(mdp.Goal)
        self.Sink = self.convert_sink(mdp.IDS)
        
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
def state_features(world):
    """
    The state features, assign each state with a feature
    """
#    state_features = np.zeros((len(world.statespace), len(world.F) + len(world.G)))
#    for i in range(len(world.F)):
#        state_features[world.F[i]][i] = 1
#    for i in range(len(world.G)):
#        state_features[world.G[i]][i+len(world.F)] = 1
#    return state_features
    
    state_features = np.zeros((len(world.statespace), len(world.F)))
    for i in range(len(world.F)):
        state_features[world.F[i]][i] = 1
    state_features = state_features * 100
    return state_features
        
#def state_features(world):
#    """
#    Return the feature matrix assigning each state with an individual
#    feature (i.e. an identity matrix of size n_states * n_states).
#
#    Rows represent individual states, columns the feature entries.
#
#    Args:
#        world: A GridWorld instance for which the feature-matrix should be
#            computed.
#
#    Returns:
#        The coordinate-feature-matrix for the specified world.
#    """
#    return np.identity(len(world.statespace))

def test_att():
    """
    In this test function, the agent is maximizing the probability of reaching the decoys
    while avoiding the IDS placements and the true goal
    """
    IDSlist = ["q6"]
    G1 = ["q11"]
    F1 = ["q12", "q13", "q14"]
#    F1 = ["q8", "q12", "q13", "q14"]
    mdp = MDP()
    mdp.getgoals(G1)
    mdp.getfakegoals(F1)
    mdp.stotrans = mdp.getstochastictrans()
    mdp.addIDS(IDSlist)
    policy_att, V_att = mdp.getpolicy()
    return policy_att, V_att, mdp

def test():
#    mdp = test_mdp()
#    policy_wstatt, V_wstatt = mdp.getpolicy_det_min()
    policy_wstatt, V_wstatt, mdp = test_att()
    world = World(mdp)
    policy = world.convert_policy(policy_wstatt)
    st_fre = mdp.stVisitFre(policy_wstatt)
    world.statevisiting(st_fre)
    return world, policy

def test_gridworld():
    """
    This function is used to test gridworld
    """
    gridworld, V, policy = createGridWorld()
    world_convert = World(gridworld)
    policy_convert = world_convert.convert_policy(policy)
#    Z = gridworld.stvisitFreq(policy)
#    world_convert.statevisiting(Z)
    return world_convert, gridworld, policy_convert

if __name__ == "__main__":
#    world, policy = test()
#    next_p = world.transition[10, :, 0]
#    next_state = np.random.choice(world.statespace, p=next_p)
#    print(next_state)
    world, gridworld, policy = test_gridworld()
    state_feature = state_features(world)
    print(state_feature)