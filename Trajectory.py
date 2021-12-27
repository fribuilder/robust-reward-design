# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 15:43:06 2021

@author: 53055
"""

import numpy as np
from itertools import chain

class Trajectory:
    def __init__(self, transition):
        self._t = transition
    
    def transition(self):
        return self._t
    
    def states(self):
        """
        The states visited in this trajectory.

        Returns:
            All states visited in this trajectory as iterator in the order
            they are visited. If a state is being visited multiple times,
            the iterator will return the state multiple times according to
            when it is visited.
        """
        return map(lambda x: x[0], chain(self._t, [(self._t[-1][2], 0, 0)]))

        
    
def generate_trajectory(world, policy, start, final):
    state = start
    
    trajectory = []
    while state not in final:
        action = policy(state)
        next_st = world.statespace
        next_p = world.transition[state, :, action]
        next_state = np.random.choice(next_st, p=next_p)
        trajectory += [(state, action, next_state)]
        state = next_state
    return Trajectory(trajectory)
    

def generate_trajectories(n, world, policy, start, final):
    start_states = np.atleast_1d(start)
    
    def generate_one():
        if len(start_states) == len(world.statespace):
            s = np.random.choice(range(len(world.statespace)), p=start_states)
        else:
            s = np.random.choice(start_states)

        return generate_trajectory(world, policy, s, final)
    
    trajectory_list = []
    for i in range(n):
        trajectory_list.append(generate_one())
    return trajectory_list

def stochastic_policy_adapter(policy):
    """
    A policy adapter for stochastic policies.

    Adapts a stochastic policy given as array or map
    `policy[state, action] -> probability` for the trajectory-generation
    functions.

    Args:
        policy: The stochastic policy as map/array
            `policy[state: Integer, action: Integer] -> probability`
            representing the probability distribution p(action | state) of
            an action given a state.

    Returns:
        A function `(state: Integer) -> action: Integer` acting out the
        given policy, choosing an action randomly based on the distribution
        defined by the given policy.
    """
    return lambda state: np.random.choice([*range(policy.shape[1])], p=policy[state, :])