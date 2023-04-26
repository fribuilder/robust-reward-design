import World as W
import max_Ent as M
import max_EntGrid as MG
import max_EntV2 as M2
import plot as P
import Trajectory as T
import solver as S
import optimizer as O
import pickle
import policyImprovement
import time

import numpy as np
import matplotlib.pyplot as plt

def setup_MDP():
    # world, gridworld, exp_policy= W.test()     #Attack graph case
#    world, gridworld, exp_policy= W.test_gridworldV2()   #GridWorld case
    
#    world, gridworld, exp_policy = W.test_mdpV2()   #mdp case, state act visiting
#    world, gridworld, exp_policy = W.test_mdpSmall()
#    world, gridworld, exp_policy = W.test_gridworld_new2() #Gridworld case 6*6, state act visiting
    world, gridworld, exp_policy = W.test_gridworld_new3() #Gridworld case 10*10, state act visiting
#    world, gridworld, exp_policy = W.test_gridworld_agent() #Gridworld with moving agent case
#    reward_ori = gridworld.getreward_def(1)  #Choose 1 for mdp and 100 for gridworld
#    reward_mod = gridworld.initial_reward(reward_ori)
#    reward = np.zeros(len(world.stateactVisiting))
    reward = []
#    for st in reward_mod.keys():
#        for act, r in reward_mod[st].items():
#            reward.append(r)
#    print(reward)
#    input("111")
    terminal = []
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
#    modifylist = [8, 68]
#    modifylist = [24, 25, 26, 27, 72, 73, 74, 75, 8, 68]
#    modifylist = [99, 103, 107, 112, 113, 114, 115, 40, 116]  #Gridworld example 6*6
    # modifylist = [99, 103, 107, 112, 113, 114, 115, 8, 132]
    # features = W.state_act_feature_manual_list(world, modifylist)  #52*3
    
#    F = [(1, 4), (4, 5)]    #Random walking agent example
#    features = W.state_act_feature_walkingAgent(world, gridworld, F)   #Random walking agent example
    features = W.state_act_feature_decoyonly(world, gridworld)  #Only modify decoy reward
#    print(features)
    
    
    init = O.Constant(1.0)
    
#    optim = O.ExpSga(lr=O.linear_decay(lr0=0.01))
    optim = O.Sga(lr=O.linear_decay(lr0=0.1))
    
    e_feature = world.stateactVisiting
    



#    reward = M.irl(gridworld, world.transition, features, terminal, trajectories, optim, init)  #Attack Graph case
    
#    reward = MG.irl(gridworld, world.transition, features, terminal, trajectories, optim, init)   #Gridworld case
    
    reward = M2.irl(gridworld, world.transition, features, terminal, trajectories, optim, init, e_feature)  

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
#    traj = generate_trajectories(world, reward, terminal, expert_policy)
    traj = []
#    state_features = W.state_features(world)
#    print(state_features)
    print("Get all Trajectories")
    reward_maxent = maxEnt(world, gridworld, terminal, traj)
    print("reward: ", reward_maxent)
    return traj, gridworld, reward_maxent

def synthesis_improve(eps, iter_thre):
    world, gridworld, reward, termial, exp_policy = setup_MDP()
    traj = []
    V_0 = np.inf
    diff = np.inf
    terminal = []
    reward_maxent = maxEnt(world, gridworld, terminal, traj)
    policy_att, V_att = gridworld.getpolicy(reward_maxent)
    st_visit_att = gridworld.stVisitFre(policy_att)
    reward_d = gridworld.getreward_def(1)
    reward_improve = gridworld.getreward_def(3)
    V_def = policyImprovement.policyEval(gridworld, reward_d, policy_att)
    # policy_improve, V_improve = policyImprovement.policyImpro(gridworld, V_def, reward_d)
    V_def_e = policyImprovement.policyEval_Ent(gridworld, reward_improve, policy_att)
    policy_improve, V_improve = policyImprovement.policyImpro(gridworld, V_def_e, reward_improve)
    V_improve = policyImprovement.policyEval(gridworld, reward_d, policy_improve)
    print("V_def[30] is:", V_def[30])
    print("V_improve[30] is:", V_improve[30])
    st_act_visit_imp = gridworld.stactVisitFre(policy_improve)
    st_visit_imp = gridworld.stVisitFre(policy_improve)
    itcount = 1
    V_att_record = [V_att]
    V_def_record = [V_def]
    st_act_visit_att_record = [st_visit_att]
    st_act_visit_imp_record = [st_visit_imp]
    diff_record = []
    while itcount == 1 or diff >= eps:
        print("policy improvement iteration:", itcount)
        V_0 = V_def[30]   #Adding index 12 for 6*6 51 for 10*10, 30 for 10*10
        print(V_0)
        world.stateActVisiting(st_act_visit_imp)
        reward_maxent = maxEnt(world, gridworld, terminal, traj)
        policy_att, V_att = gridworld.getpolicy(reward_maxent)
        st_visit_att = gridworld.stVisitFre(policy_att)
        reward_d = gridworld.getreward_def(1)
        reward_improve = gridworld.getreward_def(3)
        V_def = policyImprovement.policyEval(gridworld, reward_d, policy_att)
        V_def_e = policyImprovement.policyEval_Ent(gridworld, reward_improve, policy_att)

        # policy_improve, V_improve = policyImprovement.policyImpro(gridworld, V_def, reward_d)
        policy_improve, V_improve = policyImprovement.policyImpro(gridworld, V_def_e, reward_improve)
        V_improve = policyImprovement.policyEval(gridworld, reward_d, policy_improve)
        print("V_def[30] is:", V_def[30])
        print("V_improve[30] is:", V_improve[30])
        st_act_visit_imp = gridworld.stactVisitFre(policy_improve)
        st_visit_imp = gridworld.stVisitFre(policy_improve)
        V_att_record.append(V_att)
        V_def_record.append(V_def)
        st_act_visit_att_record.append(st_visit_att)
        st_act_visit_imp_record.append(st_visit_imp)
        diff = abs(V_0 - V_def[30]) #Adding index 12 for 6*6 51 for 10*10, 30 for 10*10
        diff_record.append(diff)
        print("difference is:", V_0 - V_def[30])
        if itcount >= iter_thre:
            break
        itcount += 1
    return V_att_record, V_def_record, diff_record, reward_maxent, st_act_visit_att_record, st_act_visit_imp_record
    
    
if __name__ == "__main__":
    start_time = time.time()
#    traj, gridworld, reward = test()
    V_att, V_def, diff, reward, st_act_visit_att, st_act_visit_imp = synthesis_improve(1e-4, 10)
    end_time = time.time()
    # print(reward)
    print("Running time:", end_time - start_time)
    
    
    
    
    
#    mdp_file = 'gridworld2.pkl'
#    reward_file = 'rewardgrid2_5.pkl'

#    picklefile = open(mdp_file, "wb")
#    pickle.dump(gridworld, picklefile)
#    picklefile.close()
    
#    picklefile = open(reward_file, "wb")
#    pickle.dump(reward, picklefile)
#    picklefile.close()



                
    