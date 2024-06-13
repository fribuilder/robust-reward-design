from mip import *
import numpy as np
import MDP
import MDP_V2
import GridWorldV2
def LP(mdp):
    model = Model(solver_name=GRB)
    gamma = 0.95
    st_len = len(mdp.statespace)
    act_len = len(mdp.A)
    decoy_index = []
    for decoy in mdp.F:
        decoy_index.append(mdp.statespace.index(decoy))
    init = np.zeros(st_len)
    init[0] = 1
    init[30] = 1
    y = [model.add_var() for i in range(st_len * act_len)]
    R1 = np.zeros(st_len)
#    for i in decoy_index:
#        R1[i] = 1
    for i in mdp.G:
        R1[mdp.statespace.index(i)] = 1

    model.objective = maximize(xsum(R1[i] * xsum(y[i * act_len + j] for j in range(act_len)) for i in range(st_len)))
    D, E, F = generate_matrix(mdp)
    for i in range(st_len * act_len):
        model += y[i] >= 0

    for i in range(st_len):
        model += xsum((E[i][j] * y[j] - gamma * F[i][j] * y[j]) for j in range(st_len * act_len)) - init[i] == 0

    print("Start optimization")
    # model.max_gap = 0.05
    status = model.optimize()  # Set the maximal calculation time
    print("Finish optimization")
    print(status)
    if status == OptimizationStatus.OPTIMAL:
        print("The model objective is:", model.objective_value)
    elif status == OptimizationStatus.FEASIBLE:
        print('sol.cost {} found, best possible: {}'.format(model.objective_value, model.objective_bound))
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print('no feasible solution found, lower bound is: {}'.format(model.objective_bound))
    else:
        print("The model objective is:", model.objective_value)


def generate_matrix(mdp):
    st_len = len(mdp.statespace)
    act_len = len(mdp.A)

    # sum over all action wrt one state, out visit
    E = np.zeros((st_len, st_len * act_len))
    for i in range(st_len):
        for j in range(act_len):
            E[i][i * act_len + j] = 1

    # in visit, corresponds to the upper one
    F = np.zeros((st_len, st_len * act_len))
    for st in mdp.stotrans.keys():
        for act in mdp.stotrans[st].keys():
            for st_, pro in mdp.stotrans[st][act].items():
                if st_ in mdp.statespace:
                    F[mdp.statespace.index(st_)][mdp.statespace.index(st) * act_len + mdp.A.index(act)] = pro

    D = np.zeros(st_len * act_len)
    for i in mdp.G:
        goal_index = mdp.statespace.index(i)
        for j in range(act_len):
            D[goal_index * act_len + j] = 1

    return D, E, F

if __name__ == "__main__":
#    mdp, policy, V_att, V_def, st_visit, st_act_visit = MDP_V2.test_att()
    mdp, V_def, policy = GridWorldV2.createGridWorldBarrier_new3()
    LP(mdp)

