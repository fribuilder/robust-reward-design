import pickle
import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
import GridWorldV2
import MDP

def extract_decoy_value(x_res, mdp):
    decoy_value = np.zeros(len(mdp.F))
    for k, decoy in enumerate(mdp.F):
        decoy_value[k] = x_res[mdp.statespace.index(decoy)]
    return decoy_value

def get_modified_reward(mdp, x_decoy, y, cvxpy=False):
    # Add the modification due to decoy allocation. 
    assert x_decoy.size == len(mdp.F)

    obj_fcn = 0
    act_len = len(mdp.A)
    for k, decoy in enumerate(mdp.F): 
        # k: Index of the decoy
        # decoy: A 2-tuple representing the spatial location of the decoy

        # Obtain the spatial index, denoted by i, of the decoy in the flattened vector.
        i = mdp.statespace.index(decoy)

        # Add the modification given by decoy k
        if cvxpy is True:
            obj_fcn += x_decoy[k] * cp.sum(y[i*act_len:(i+1)*act_len])
        else:
            obj_fcn += x_decoy[k] * np.sum(y[i*act_len:(i+1)*act_len])
    return obj_fcn

def eval_decoy_allocation(mdp, tau, decoy_value):
    '''
    Evaluates a given decoy allocation.

    Return:
    - y_opt: Attacker's best response under decoy_value.
    - def_val: Defender's value.
    '''
    _, y_opt, def_val = ent_gradient(mdp, tau, decoy_value, eval = False)
    return y_opt, def_val
    #reconstruct_policy(y_opt, len(mdp.A))
    
def ent_gradient(mdp,tau,decoy_value,y_test=None,debug=False, eval=False): # add y_test
    '''
    decoy_value: 1D array, size = number of decoy spatial locations.
    '''

    #define variables
    st_len = len(mdp.statespace)
    act_len = len(mdp.A)
    n_decoys = len(mdp.F)
    assert len(decoy_value) == n_decoys
    
    #y stands for occupancy measure
    y = cp.Variable(st_len * act_len)
    # x represent the decoy allocation (only the value without position)
    # x = cp.Parameter(st_len * act_len)

    # x_decoy represents the decoy resource allocation. The length is the same as the number of decoy states.
    x_decoy = cp.Parameter(n_decoys)

    #define constants
    D, E, F = generate_matrix(mdp) # D: Reward vector corresponding to the true goal.
    gamma = 0.95
    init = np.zeros(st_len)
    #init[0] = 1 # mdp case
    init[12] = 1 #6 * 6 case
    #init[30] = 1 #10 * 10 case

    # decoy_index stores the spatial of decoy locations in the flattened vector.
    # Example: For a 6x6 grid world, when decoys are located at (1,4) and (4,5), then decoy_index = [10, 29].
    decoy_index = []  
    for decoy in mdp.F:
        decoy_index.append(mdp.statespace.index(decoy))
    # print(decoy_index)

    R1_np = np.zeros(st_len * act_len)  # R1_np is the reward vector (of length SA) for the defender. It is 1 for all the decoy states.
    for i in decoy_index:
        R1_np[i*act_len:(i+1)*act_len] = 1
    R1_tch = torch.from_numpy(R1_np)
    tau0 = tau

    #define parameter
    x_decoy_tch = torch.from_numpy(decoy_value)
    x_decoy_tch.requires_grad_(True)

    #sum of occpancy measure according to action
    y_a_matrix = np.zeros(st_len*act_len)
    y_a_matrix = np.kron(np.eye(st_len),np.ones((act_len,act_len)))

    #construct problem
    obj_fcn = D @ y - tau0 * cp.sum(cp.rel_entr(y,y_a_matrix @ y)) + get_modified_reward(mdp, x_decoy, y, cvxpy=True)

    objective = cp.Maximize(obj_fcn) #notce + or - set tau0 = 0 
    #1. positive 2.in and out flow must equal 3.replce
    constraints = [y >= 0, E @ y - gamma * F @ y - init == 0]

    if y_test is not None:
        print('Feasibility of y_test (SSE):', np.max(np.abs(E @ y_test - gamma * F @ y_test - init)))
        print('Attacker value under y_test:', D @ y_test + get_modified_reward(mdp, decoy_value, y_test, cvxpy=False))
        print('Defender value under y_test:', R1_np @ y_test)

    '''
    Parameter does not need the constraint
    , x >= 0, cons_decoy_value.T @ x == 0]

    for i in range(st_len):
        for j in range(act_len):
            constraints += [
                x[i*act_len] == x[i*act_len + j]
            ]
    '''

    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp('dcp')

    if eval == False :
        '''First try to solve with cvxpylayer'''
        cvxpylayer = CvxpyLayer(problem, parameters=[x_decoy], variables=[y])

        #solve the problem
        y_tch, = cvxpylayer(x_decoy_tch, solver_args={"eps": 1e-6})
        def_score = R1_tch @ y_tch
        def_score.backward()
        def_score_np = def_score.detach().numpy()

        if debug is True:
            y_np = y_tch.detach().numpy()
            print('Attacker opt value (boundedly rational response):', D @ y_np + get_modified_reward(mdp, decoy_value, y_np, cvxpy=False))
            print('Defender opt value (boundedly rational response):', def_score_np)

        x_grad_np = x_decoy_tch.grad.detach().numpy()
        y_np = y_tch.detach().numpy()

        return x_grad_np, y_np, def_score_np
    if eval == True:
        '''now use gurobi to evaluate the LP problem'''
        x_decoy.value = decoy_value
        problem.solve(solver=cp.GUROBI)

        def_score = R1_np @ y
        return x_grad_np, y, def_score
    
def generate_matrix(mdp):
    st_len = len(mdp.statespace)
    act_len = len(mdp.A)
    
    #sum over all action wrt one state, out visit
    E = np.zeros((st_len, st_len * act_len))
    for i in range(st_len):
        for j in range(act_len):
            E[i][i * act_len + j] = 1
    
    #in visit, corresponds to the upper one
    F = np.zeros((st_len, st_len * act_len))
    for st in mdp.stotrans.keys():
        for act in mdp.stotrans[st].keys():
            for st_, pro in mdp.stotrans[st][act].items():
                if st_ in mdp.statespace:
                    F[mdp.statespace.index(st_)][mdp.statespace.index(st) * act_len + mdp.A.index(act)] = pro
    
    '''rewards'''
    D = np.zeros(st_len * act_len)
    for i in mdp.G:
        goal_index = mdp.statespace.index(i)
        for j in range(act_len):
            D[goal_index * act_len + j] = 1
            
    return D, E, F

def gen_problem_data():
    #policy, V_att, V_def, st_visit, mdp = MDP.test_att()
    mdp, _, _ = GridWorldV2.createGridWorldBarrier_new2()#new3 to new2 to change case
    D, E, F = generate_matrix(mdp)
    return D, E, F, mdp

def index_change(x_res,mdp):
    st_len = len(mdp.statespace)
    act_len = len(mdp.A)
    decoy_value = np.zeros(st_len * act_len)
    for i in range(len(x_res)):
        decoy_value[i*act_len:(i+1)*act_len] = x_res[i]
    return decoy_value

def test_gradient_v2():
    tau = 1e-5

    D, E, F, mdp = gen_problem_data()
    st_len = len(mdp.statespace)
    act_len = len(mdp.A)

    # the decoy value is the optimal solution of the LPSOS1
    with open('x_res_A', 'rb') as fp:
        x_res = pickle.load(fp)
    # x_res[10] = x_res[10] - perturb
    # assert x_res[10] >= 0

    # decoy_value = index_change(x_res, mdp)
    # print(decoy_value)
    # decoy_value = np.zeros(st_len*act_len)
    decoy_value = extract_decoy_value(x_res, mdp)  # Only keep the entries corresponding to valid decoy locations.
    # print(decoy_index)

    with open('y_res_A', 'rb') as fp:
        y_res = pickle.load(fp)
    
    x_grad_np, y_np, def_value = ent_gradient(mdp, tau, decoy_value, y_test=np.array(y_res), debug=True)
    # print('\n','gradient of x:', x_grad_np,'\n\n','occupancy measure:', y_np)

    #compare the optimal solution and the ent-regularized solution

    # print('difference:', np.array(y_res) - y_np)
    return x_grad_np, y_np, def_value

if __name__ == "__main__":
    test_gradient_v2()
 

                    
