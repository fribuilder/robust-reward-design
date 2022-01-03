# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 15:17:14 2021

@author: 53055
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 11:06:57 2021

@author: 53055
"""

import numpy as np
from itertools import product

def feature_expectation_from_trajectories(features, trajectories):
    n_state, n_features = features.shape
    fe = np.zeros(n_features)
    
    for t in trajectories:
        for s in t.states():
            fe += features[s, :]

    return fe / len(trajectories)

def initial_probability_from_trajectories(n_state, trajectories):
    p = np.zeros(n_state)

    p[14] = 1.0

    return p

def expected_svf_from_policy(p_transition, p_initial, terminal, p_action, eps=1e-5):
    n_states, _, n_actions = p_transition.shape
#    print(p_action)
    p_transition = np.copy(p_transition)
    for i in terminal:
        p_transition[i, :, :] = 0.0

    
    p_transition = [np.array(p_transition[:, :, a]) for a in range(n_actions)]
    
    d = np.zeros(n_states)

    delta = np.inf
    while delta > eps:
        d_ = [p_transition[a].T.dot(p_action[:, a] * d) for a in range(n_actions)]
        d_ = p_initial + np.array(d_).sum(axis=0)

        delta, d = np.max(np.abs(d_ - d)), d_

    return d

def local_action_probabilities(gridworld, reward):
    reward[20] = 100
    reward[48] = 100
    print("reward in local:", reward)
    V, policy = gridworld.getpolicy(reward)
    policy_mat = np.zeros((len(gridworld.statespace), len(gridworld.A)))
    for i in range(len(gridworld.statespace)):
        for j in range(len(gridworld.A)):
            policy_mat[i][j] = policy[gridworld.statespace[i]][gridworld.A[j]]
    return policy_mat

def compute_expected_svf(gridworld, p_transition, p_initial, terminal, reward, eps = 1e-5):
    p_action = local_action_probabilities(gridworld, reward)
#    print(p_action)
#    input("111")
    return expected_svf_from_policy(p_transition, p_initial, terminal, p_action, eps)

def barrier(theta, c = 1.6, t = 1):
    n_feature = theta.shape
    bar = 1/t * np.ones(n_feature)/(c - sum(theta))
    return bar
def irl(gridworld, p_transition, features, terminal, trajectories, optim, init, eps=1e-5, eps_esvf=1e-5):
    n_state, _, n_action = p_transition.shape
    _, n_feature = features.shape
    print(n_feature)
    e_features = np.array([21.05, 72.73])
    
    p_initial = initial_probability_from_trajectories(n_state, trajectories)
#    print(p_initial)
    theta = init(n_feature)
#    print(theta)
    delta = np.inf
#    theta = theta * 0.6
#    theta[1] = 0.5
    optim.reset(theta)

    iter_count = 0
    while delta > eps:
        print("iter_count:", iter_count)
        theta_old = theta.copy()
        
        # compute per-state reward
        print("theta:", theta)
        reward = features.dot(theta)
        print("reward enter compute:", reward)

        # compute the gradient
        e_svf = compute_expected_svf(gridworld, p_transition, p_initial, terminal, reward, eps_esvf)
        print("e_svf is:", e_svf)
        #Use this without barrier function
        grad = e_features - features.T.dot(e_svf)
        
        #Use this with barrier function
#        bar = barrier(theta)
#        grad = e_features - features.T.dot(e_svf) - bar
        print("grad is:", grad)
#        input("111")
        # perform optimization step and compute delta for convergence
        optim.step(grad)
        print("theta after optimize:", theta)
        delta = np.max(np.abs(theta_old - theta))

        iter_count += 1
    print("theta is:", theta)
    return features.dot(theta)
    
def softmax(x1, x2):
    """
    Computes a soft maximum of both arguments.

    In case `x1` and `x2` are arrays, computes the element-wise softmax.

    Args:
        x1: Scalar or ndarray.
        x2: Scalar or ndarray.

    Returns:
        The soft maximum of the given arguments, either scalar or ndarray,
        depending on the input.
    """
    x_max = np.maximum(x1, x2)
    x_min = np.minimum(x1, x2)
    return x_max + np.log(1.0 + np.exp(x_min - x_max))

def local_causal_action_probabilities(p_transition, terminal, reward, discount, eps=1e-5):
    """
    Compute the local action probabilities (policy) required for the edge
    frequency calculation for maximum causal entropy reinfocement learning.

    This is Algorithm 9.1 from Ziebart's thesis (2010) combined with
    discounting for convergence reasons as proposed in the same thesis.

    Args:
        p_transition: The transition probabilities of the MDP as table
            `[from: Integer, to: Integer, action: Integer] -> probability: Float`
            specifying the probability of a transition from state `from` to
            state `to` via action `action` to succeed.
        terminal: Either the terminal reward function or a collection of
            terminal states. Iff `len(terminal)` is equal to the number of
            states, it is assumed to contain the terminal reward function
            (phi) as specified in Ziebart's thesis. Otherwise `terminal` is
            assumed to be a collection of terminal states from which the
            terminal reward function will be derived.
        reward: The reward signal per state as table
            `[state: Integer] -> reward: Float`.
        discount: A discounting factor as Float.
        eps: The threshold to be used as convergence criterion for the state
            partition function. Convergence is assumed if the state
            partition function changes less than the threshold on all states
            in a single iteration.

    Returns:
        The local action probabilities (policy) as map
        `[state: Integer, action: Integer] -> probability: Float`
    """
    n_states, _, n_actions = p_transition.shape

    # set up terminal reward function
    if len(terminal) == n_states:
        reward_terminal = np.array(terminal, dtype=np.float)
    else:
        reward_terminal = -np.inf * np.ones(n_states)
        for i in terminal:
            reward_terminal[i] = 0.0

    # set up transition probability matrices
    p = [np.array(p_transition[:, :, a]) for a in range(n_actions)]

    # compute state log partition V and state-action log partition Q
    v = -1e200 * np.ones(n_states)  # np.dot doesn't behave with -np.inf

    delta = np.inf
    while delta > eps:
        v_old = v

        q = np.array([reward + discount * p[a].dot(v_old) for a in range(n_actions)]).T

        v = reward_terminal
        for a in range(n_actions):
            v = softmax(v, q[:, a])

        # for some reason numpy chooses an array of objects after reduction, force floats here
        v = np.array(v, dtype=np.float)

        delta = np.max(np.abs(v - v_old))

    # compute and return policy
    return np.exp(q - v[:, None])

def compute_expected_causal_svf(p_transition, p_initial, terminal, reward, discount,
                                eps_lap=1e-5, eps_svf=1e-5):
    """
    Compute the expected state visitation frequency for maximum causal
    entropy IRL.

    This is a combination of Algorithm 9.1 and 9.3 of Ziebart's thesis
    (2010). See `local_causal_action_probabilities` and
    `expected_svf_from_policy` for more details.

    Args:
        p_transition: The transition probabilities of the MDP as table
            `[from: Integer, to: Integer, action: Integer] -> probability: Float`
            specifying the probability of a transition from state `from` to
            state `to` via action `action` to succeed.
        p_initial: The probability of a state being an initial state as map
            `[state: Integer] -> probability: Float`.
        terminal: Either the terminal reward function or a collection of
            terminal states. Iff `len(terminal)` is equal to the number of
            states, it is assumed to contain the terminal reward function
            (phi) as specified in Ziebart's thesis. Otherwise `terminal` is
            assumed to be a collection of terminal states from which the
            terminal reward function will be derived.
        reward: The reward signal per state as table
            `[state: Integer] -> reward: Float`.
        discount: A discounting factor as Float.
        eps_lap: The threshold to be used as convergence criterion for the
            state partition function. Convergence is assumed if the state
            partition function changes less than the threshold on all states
            in a single iteration.
        eps_svf: The threshold to be used as convergence criterion for the
            expected state-visitation frequency. Convergence is assumed if
            the expected state visitation frequency changes less than the
            threshold on all states in a single iteration.
    """
    p_action = local_causal_action_probabilities(p_transition, terminal, reward, discount, eps_lap)
    return expected_svf_from_policy(p_transition, p_initial, terminal, p_action, eps_svf)

def irl_causal(p_transition, features, terminal, trajectories, optim, init, discount,
               eps=1e-4, eps_svf=1e-5, eps_lap=1e-5):
    """
    Compute the reward signal given the demonstration trajectories using the
    maximum causal entropy inverse reinforcement learning algorithm proposed
    Ziebart's thesis (2010).

    Args:
        p_transition: The transition probabilities of the MDP as table
            `[from: Integer, to: Integer, action: Integer] -> probability: Float`
            specifying the probability of a transition from state `from` to
            state `to` via action `action` to succeed.
        features: The feature-matrix (e.g. as numpy array), mapping states
            to features, i.e. a matrix of shape (n_states x n_features).
        terminal: Either the terminal reward function or a collection of
            terminal states. Iff `len(terminal)` is equal to the number of
            states, it is assumed to contain the terminal reward function
            (phi) as specified in Ziebart's thesis. Otherwise `terminal` is
            assumed to be a collection of terminal states from which the
            terminal reward function will be derived.
        trajectories: A list of `Trajectory` instances representing the
            expert demonstrations.
        optim: The `Optimizer` instance to use for gradient-based
            optimization.
        init: The `Initializer` to use for initialization of the reward
            function parameters.
        discount: A discounting factor for the log partition functions as
            Float.
        eps: The threshold to be used as convergence criterion for the
            reward parameters. Convergence is assumed if all changes in the
            scalar parameters are less than the threshold in a single
            iteration.
        eps_lap: The threshold to be used as convergence criterion for the
            state partition function. Convergence is assumed if the state
            partition function changes less than the threshold on all states
            in a single iteration.
        eps_svf: The threshold to be used as convergence criterion for the
            expected state-visitation frequency. Convergence is assumed if
            the expected state visitation frequency changes less than the
            threshold on all states in a single iteration.
    """
    n_states, _, n_actions = p_transition.shape
    _, n_features = features.shape

    # compute static properties from trajectories
#    e_features = feature_expectation_from_trajectories(features, trajectories)
#    p_initial = initial_probabilities_from_trajectories(n_states, trajectories)
#    e_features = np.array([0.413, 0.056, 0.469])
    e_features = np.array([0.29, 0.052, 0.342])
    
    p_initial = initial_probability_from_trajectories(n_states, trajectories)

    # basic gradient descent
    theta = init(n_features)
    delta = np.inf
    theta = theta*0.9
    optim.reset(theta)
    while delta > eps:
        theta_old = theta.copy()

        # compute per-state reward
        reward = features.dot(theta)

        # compute the gradient
        e_svf = compute_expected_causal_svf(p_transition, p_initial, terminal, reward, discount,
                                            eps_lap, eps_svf)

        grad = e_features - features.T.dot(e_svf)

        
        # perform optimization step and compute delta for convergence
        optim.step(grad)
        print("e_svf is:", e_svf)
        print("grad is:", grad)
        print("theta after optimize:", theta)
        input("111")
        delta = np.max(np.abs(theta_old - theta))

    # re-compute per-state reward and return
    return features.dot(theta)