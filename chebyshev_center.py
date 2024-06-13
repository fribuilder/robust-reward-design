import pickle
import cvxpy as cp
import numpy as np
import GridWorldV2
import gradient_v2 as gv

def chebyshev():
    with open('y_res', 'rb') as fp:
        y_res = pickle.load(fp)
    m_star = np.array(y_res)
    '''Use this function to find the chebyshev center of the Linear Bilevel Problem,
    this function needs the solution (optimal occupancy measure) derived by the mip first'''

    D, E, F, mdp = gv.gen_problem_data()
    A1, A2, A3, A4, R1_np = generate_matrix_for_constraints()
    # D is the vector of length st_len * act_len, and E and F are 2-d arrays of 
    # (st_len, st_len * act_len)
    st_len = len(mdp.statespace)
    act_len = len(mdp.A)

    init = np.zeros(st_len)
    init[12] = 1 #6 * 6 case

    gamma = cp.Variable() # radius of the Chebyshev center
    center = cp.Variable(st_len*act_len)# center of the Chebyshev center
    mu1 = cp.Variable(A1.shape[0])
    mu2 = cp.Variable(A1.shape[0])
    mu3 = cp.Variable(A1.shape[0])
    mu4 = cp.Variable(A1.shape[0])

    objective = cp.Maximize(gamma)

    R1_np_1 = np.zeros(st_len*act_len) 
    R1_np_1[np.where(R1_np==1)[0][0:act_len]] = 1
    R1_np_2 = np.zeros(st_len*act_len)
    R1_np_2[np.where(R1_np==1)[0][act_len:2*act_len+1]] = 1

    # these y are the vertexes of the y + gamma*e_d where e_d
    # only takes value on the states that allow decoy and the
    # infinity norm of it less than 1
    y1 = center + gamma*R1_np
    y2 = center - gamma*R1_np
    y3 = center + gamma*R1_np_1 - gamma*R1_np_2
    y4 = center - gamma*R1_np_1 + gamma*R1_np_2
 
    constraints = [ y1 @ m_star >= mu1 @ init,
                    y2 @ m_star >= mu2 @ init,
                    y3 @ m_star >= mu3 @ init,
                    y4 @ m_star >= mu4 @ init,
                    mu1 @ A1 >= y1,
                    mu2 @ A1 >= y2,
                    mu3 @ A1 >= y3,
                    mu4 @ A1 >= y4,
                    # for vertexes, there are mu's satisfy the condition
                    A3 @ center <= 4 - 2*gamma,
                    A2 @ center == D,
                    # D is the original goals of MDP
                    A4 @ center == 0,
                    center - gamma * R1_np >= 0]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver = cp.MOSEK)
    
    return center.value, gamma.value, m_star


def generate_matrix_for_constraints():
    D, E, F, mdp = gv.gen_problem_data()
    # D is the vector of length st_len * act_len, and E and F are 2-d arrays of 
    # (st_len, st_len * act_len)
    st_len = len(mdp.statespace)
    act_len = len(mdp.A)
    tau = 0.95
    init = np.zeros(st_len)
    init[12] = 1 #6 * 6 case

    # A1 @ m = init certifies feasible occupancy measure
    A1 = E - tau * F

    decoy_index = []  
    for decoy in mdp.F:
        decoy_index.append(mdp.statespace.index(decoy))

    # R1_np is the reward vector (of length |S|*|A|) for the defender. It is 1 for all the decoy states. 
    R1_np = np.zeros(st_len * act_len)  
    for i in decoy_index:
        R1_np[i*act_len:(i+1)*act_len] = 1

    # A2 pin out states that do not allow decoy
    A2 = np.eye(st_len * act_len) - np.diag(R1_np)

    # A3 sum the deccoy in y
    A3 = np.zeros(st_len*act_len)
    for i in decoy_index:
        A3[i*act_len] = 1


    # A4 makes sure that the decoy in every acctions are the same for the decoy state
    A4 = np.zeros((len(decoy_index)*(act_len-1), st_len*act_len))
    for i,j in enumerate(decoy_index):
        A4[i*(act_len-1):(i+1)*(act_len-1),j*act_len] = 1
    for i in range(len(decoy_index)):
        for j in range(act_len-1):
            A4[i*(act_len-1)+j,decoy_index[i]*act_len+j+1] = -1
    
    return A1,A2,A3,A4,R1_np

if __name__ == '__main__':
    test = True # for convenience of debugging, output matrices

    if test == True:
        D, E, F, mdp = gv.gen_problem_data()
        A1, A2, A3, A4,R1_np = generate_matrix_for_constraints()

    with open('y_res', 'rb') as fp:
        y_res = pickle.load(fp)

    center, gamma, m_star = chebyshev()
    print(center, gamma)


