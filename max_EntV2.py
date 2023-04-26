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

#    p[0] = 1.0   #mdp example
    p[12] = 1.0   #gridworld example

    return p

# =============================================================================
# def expected_svf_from_policy(p_transition, p_initial, terminal, p_action, eps=1e-5):
#     n_states, _, n_actions = p_transition.shape
# #    print(p_action)
#     p_transition = np.copy(p_transition)
#     for i in terminal:
#         p_transition[i, :, :] = 0.0
# 
#     
#     p_transition = [np.array(p_transition[:, :, a]) for a in range(n_actions)]
#     
#     d = np.zeros(n_states)
# 
#     delta = np.inf
#     while delta > eps:
#         d_ = [p_transition[a].T.dot(p_action[:, a] * d) for a in range(n_actions)]
#         d_ = p_initial + np.array(d_).sum(axis=0)
# 
#         delta, d = np.max(np.abs(d_ - d)), d_
# 
#     return d
# =============================================================================

def expected_svf_from_policy(mdp, policy):
#    Z = gridWorld.stvisitFreq(policy)
    Z = mdp.stactVisitFre(policy)  #mdp case and gridworld case
    
#    state = (2, 0)  #Gridworld with agent walking
#    init_dist = mdp.init_dist(state)  #Gridworld with agent walking
#    Z = mdp.stactVisitFre(policy, init_dist)  #Gridworld with agent walking
#    print(Z)
    Z_list = dict2list(Z)
    return np.array(Z_list)

def dict2list(Zdict):
    Z = []
    for st in Zdict.keys():
        for act, visit in Zdict[st].items():
            Z.append(visit)
    return Z

def local_action_probabilities(gridworld, reward):
#    print("reward in local:", reward)
#    reward_dict = transferlist2dict(gridworld, reward)
    policy, V = gridworld.getpolicy(reward)
    policy_mat = np.zeros((len(gridworld.statespace), len(gridworld.A)))
    for i in range(len(gridworld.statespace)):
        for j in range(len(gridworld.A)):
            policy_mat[i][j] = policy[gridworld.statespace[i]][gridworld.A[j]]
    return policy_mat, policy

def transferlist2dict(mdp, reward):
    i = 0
    reward_dict = {}
    for st in mdp.statespace:
        reward_dict[st] = {}
        for act in mdp.A:
            reward_dict[st][act] = reward[i]
            i += 1
    return reward_dict

def modifyreward(gridworld, reward, reward_ori):
    index = 0
    for st in gridworld.statespace:
        for act in gridworld.A:
            reward_ori[st][act] = reward[index]
            index += 1
    for st in gridworld.G:
        for act in gridworld.A:
            reward_ori[st][act] = 1.0
    return reward_ori

def modifyreward_grid(gridworld, reward, reward_ori):
    index = 0
    for st in gridworld.statespace:
        for act in gridworld.A:
            reward_ori[st][act] = reward[index]
            index += 1
    for st in gridworld.G:
        for act in gridworld.A:
            reward_ori[st][act] = 1
            
    for st in gridworld.F:
        for act in gridworld.A:
            reward_ori[st][act] = reward_ori[st][gridworld.A[0]]
    return reward_ori
    
def compute_expected_svf(gridworld, p_transition, p_initial, terminal, reward, eps = 1e-5):
    
    reward_ori = gridworld.getreward_att(1)
#    reward_ori = modifyreward(gridworld, reward, reward_ori)  #MDP case
    reward_ori = modifyreward_grid(gridworld, reward, reward_ori)  #gridworld case
#    reward_ori = gridworld.modifystactreward(reward_ori)
    
#    reward_ori[1] = reward[0]
#    print(reward_ori)
#    input("11")
    p_action, policy = local_action_probabilities(gridworld, reward_ori)
#    print(p_action)
#    input("111")
#    return expected_svf_from_policy(p_transition, p_initial, terminal, p_action, eps)
    return expected_svf_from_policy(gridworld, policy)

def barrier(theta, c = 2, t = 1000):
    n_feature = theta.shape
    bar = 1/t * np.ones(n_feature)/(c - sum(theta))
    return bar
def irl(gridworld, p_transition, features, terminal, trajectories, optim, init, e_features, eps=1e-4, eps_esvf=1e-5):
    n_state, _, n_action = p_transition.shape
#    print(e_features)
    e_features = np.array(e_features)
    # print(e_features)
    # input("111")
    _, n_feature = features.shape
#    print(n_feature)
#    print("features:", features)
#    input("111")
#    e_features = np.array([7.2, 89.74])
#    e_features = np.array([97.95])
    
    p_initial = initial_probability_from_trajectories(n_state, trajectories)
#    print(p_initial)
    theta = init(n_feature)
    # theta = theta * 0.1
    theta[0] = 2.5
    theta[1] = 0
    theta[2] = 2.5
    delta = np.inf
    norm = np.inf
#    theta[-1] = 1
#    theta[-2] = 1
    optim.reset(theta)

    iter_count = 0
#    e_features[2] = -e_features[2]
    print(features.T.dot(e_features))
    # input("111")
    while norm > eps:
        print("irl iter_count:", iter_count)
        theta_old = theta.copy()
        
        # compute per-state reward
        # print("theta:", theta)
        reward = features.dot(theta)  #N*2.dot 2*1
        # print("reward enter compute:", reward)

        # compute the gradient
        e_svf = compute_expected_svf(gridworld, p_transition, p_initial, terminal, reward, eps_esvf)
        # print("e_svf is:", e_svf)
        #Use this without barrier function
        grad = features.T.dot(e_features) - features.T.dot(e_svf)  #Test negative feature

#        print(grad)
        
        #Use this with barrier function
#        bar = barrier(theta)
#        grad = (e_features - features.T.dot(e_svf))/100 - bar
        # print("grad is:", grad)
#        input("111")
        # perform optimization step and compute delta for convergence
        optim.step(grad)
#        print("theta after optimize:", theta)
        theta = modify_theta(theta)
        
        delta = np.max(np.abs(theta_old - theta))
        norm = norm_1(theta_old, theta)
#        norm = norm_2(theta_old, theta)
#        print(norm)
        iter_count += 1
        print("theta is:", theta)
    reward = features.dot(theta)
    reward_F = gridworld.getreward_att(1)
    reward_F = modifyreward_grid(gridworld, reward, reward_F)
#    reward_F["q13"]["a"] = reward[52]
#    reward_F["q13"]["b"] = reward[52]
#    reward_F["q13"]["c"] = reward[52]
#    reward_F["q13"]["d"] = reward[52]
#    reward_F["q14"]["a"] = reward[56]
#    reward_F["q14"]["b"] = reward[56]
#    reward_F["q14"]["c"] = reward[56]
#    reward_F["q14"]["d"] = reward[56]
    return reward_F

def modify_theta(theta):
    for i in range(len(theta)):
        theta[i] = max(theta[i], 0)
    return theta

def norm_2(theta_old, theta):
    norm2 = 0
    theta_len = len(theta)
    for i in range(theta_len - 2):
        diff = max(theta_old[i], 0) - max(theta[i], 0)
        norm2 += np.power(diff, 2)
    norm2 += np.power(theta_old[-2] - theta[-2], 2)
    norm2 += np.power(theta_old[-1] - theta[-1], 2)
    return norm2

def norm_1(theta_old, theta):
    norm1 = 0
    theta_len = len(theta)
    for i in range(theta_len - 2):
        diff = max(theta_old[i], 0) - max(theta[i], 0)
        norm1 += abs(diff)
    norm1 += abs(theta_old[-2] - theta[-2])
    norm1 += abs(theta_old[-1] - theta[-1])
    return norm1
    
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