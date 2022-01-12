This is the instruction of how to reproduce the experiment result.

First, the transition probability of attack graph example is listed in "MdpTransition.txt" file.

First, there are two sets experiment, which the mdp is initialed in MDP.py and GridWorldV2.py. MDP.py corresponds to the small attack graph, GridWorldV2.py corresponds to the gridworld example. Go over the main function to check how to instantiate the MDP. Make sure you determine "G" in MDP, while "IDS" and "F" is empty.

In order to get the optimal sensor allocation, check the milp.py file, choose the instantiate function you want to use. Run "python milp.py -n 1 --save". Here 1 specified the number of sensors you can allocate.

Once you obtain the sensor allocation. Adding the sensor allocation to the MDP's "IDS" variable. Then manually determine the decoy candidate states, adding these states to variable "F". Change the value function to the preferred attack value, get the state visiting frequency. Decide your own threshold to eliminate decoys. 

Once you decide the final decoy state, obtain the state visiting frequency of the decoy state. Then you can move to the Inverse reinforcement learning part. 

First check main.py, in setup_MDP() function, decide the MDP you want to use. Please check the existing code and comments. Then you need to decide the corresponding IRL module, which is in the maxent() function.

There are two IRL modules, one is max_Ent.py, corresponds to the attack graph case. Another is max_EntGrid.py, corresponds to the gridworld case. 

For both of these two modules. You need to change the "e_features" variable in irl() function to the state visiting frequemcy you obtained in the preferred attack policy. Moreover, in local_action_probability() function. Fix the goal reward to what you defined. For example, in my case, the reward of reaching goal state in attack graph is 1 and the reward of reaching goal state in grid world is 100. 

In order to change the resource constraint and the parameter in barrier function, please change the "c" value for resource constraint and "t" value for parameter in barrier() function.

For other parameters, for example, initial value, learning parameter, gradient descent method, please check the maxEnt() function of main.py.  



