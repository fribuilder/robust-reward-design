import World as W
import max_Ent as M
import max_EntGrid as MG
import plot as P
import Trajectory as T
import solver as S
import optimizer as O

import numpy as np
import matplotlib.pyplot as plt

def setup_MDP():
#    world, gridworld, exp_policy= W.test()     #Attack graph case
    world, gridworld, exp_policy= W.test_gridworldV2()   #GridWorld case
    reward = np.zeros(len(world.statespace))
    terminal = []
    for i in world.F:
        reward[i] = 1
        terminal.append(i)
    for j in world.G:
        reward[j] = 1
        terminal.append(j) 
    for i in world.Sink:
        terminal.append(i)
    return world, gridworld, reward, terminal, exp_policy


def generate_trajectories(world, reward, terminal, policy):
    """
    Generate the worst attack trajectories
    """
    n = 800
    initial = np.zeros(len(world.statespace))
    initial[0] = 1.0
    traj = T.generate_trajectories(n, world, policy, initial, terminal)
    
    return traj

def maxEnt(world, gridworld, terminal, trajectories):
    
    features = W.state_features(world)
    
    init = O.Constant(1.0)
    
#    optim = O.ExpSga(lr=O.linear_decay(lr0=0.01))
    optim = O.Sga(lr=O.linear_decay(lr0=0.01))

#    reward = M.irl(gridworld, world.transition, features, terminal, trajectories, optim, init)  #Attack Graph case
    
    reward = MG.irl(gridworld, world.transition, features, terminal, trajectories, optim, init)   #Gridworld case
    
    
#    discount = 0.7
#    reward = M.irl_causal(world.transition, features, terminal, trajectories, optim, init, discount)
    
    return reward

def main():
    style = {
        'border': {'color': 'red', 'linewidth': 0.5},
    }

    # set-up mdp
    world, gridworld, reward, terminal, exp_policy = setup_MDP()

    # show our original reward
    ax = plt.figure(num='Original Reward').add_subplot(111)
    P.plot_state_values(ax, world, reward, **style)
    plt.draw()
    
    ##Need to create the worst attack policy
    
#    policy = {}
    
    trajectories = generate_trajectories(world, reward, terminal, exp_policy)
    
    ax = plt.figure(num='Expert Trajectories and Policy').add_subplot(111)
    P.plot_stochastic_policy(ax, world, exp_policy, **style)

    for t in trajectories:
        P.plot_trajectory(ax, world, t, lw=5, color='white', alpha=0.025)

    plt.draw()
    
    reward_maxent = maxEnt(world, gridworld, terminal, trajectories)
    
    ax = plt.figure(num='MaxEnt Reward').add_subplot(111)
    P.plot_state_values(ax, world, reward_maxent, **style)
    plt.draw()
    
    plt.show()
    
def test():
    world, gridworld, reward, terminal, exp_policy = setup_MDP()
#    print(terminal)
    expert_policy = T.stochastic_policy_adapter(exp_policy)
    traj = generate_trajectories(world, reward, terminal, expert_policy)
#    state_features = W.state_features(world)
#    print(state_features)
    print("Get all Trajectories")
    reward_maxent = maxEnt(world, gridworld, terminal, traj)
    print("reward: ", reward_maxent)
    return traj, world
    
if __name__ == "__main__":
#    main()
    traj, world = test()

                
    