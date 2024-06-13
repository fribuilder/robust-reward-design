import pickle
import numpy as np 
import gradient_v2 as gv
import chebyshev_center_v2 as cv
import copy
import itertools
import chebyshev_center_v2

with open('m_new', 'rb') as fp:
    m_new = pickle.load(fp)
m_new = np.array(m_new)

D, E, F, mdp = gv.gen_problem_data()
policy_new = gv.reconstruct_policy_numerical(m_new,len(mdp.A))
A1, A2, A3, A4, R1_np = cv.generate_matrix_for_constraints()
# print(policy_new, R1_np@m_new)

with open('y_res', 'rb') as fp:
    y_res = pickle.load(fp)
y_res = np.array(y_res)
policy_det = gv.reconstruct_policy_numerical(y_res,len(mdp.A))
# print(policy_det, R1_np@m_new)
# print(all(policy_det-policy_new == -1) or all(policy_det-policy_new == 1))


reward_def = mdp.getreward_def()

policy_example, V_example = mdp.getpolicy_det(reward_def)
#print(policy_example)

'''not sure that this transformation is correct'''
policy_new_dic = {}
for i, state in enumerate(mdp.statespace):
    policy_new_dic[state]={}
    for j, act in enumerate(mdp.A):
        policy_new_dic[state][act] = policy_new[i*len(mdp.A)+j]
#print('\n', policy_new_dic)

m_new_dic = {}
for i, state in enumerate(mdp.statespace):
    m_new_dic[state]={}
    for j, act in enumerate(mdp.A):
        m_new_dic[state][act] = m_new[i*len(mdp.A)+j]

V = mdp.policy_evaluation(policy_new_dic ,reward_def)

def get_expectation_term(mdp, state, act):
    expect = 0
    for st_, pro in mdp.stotrans[state][act].items():
        if st_ != "Sink":
            expect += 0.95* policy_new_dic[state][act] * mdp.stotrans[state][act][st_] * V[mdp.statespace.index(st_)]
        else:
            expect = 0.95 * policy_new_dic[state][act]*V[mdp.statespace.index(state)]
    return expect

Q = {}
for state in mdp.statespace:
    Q[state]={}
    for act in mdp.A:
        Q[state][act] = reward_def[state][act] + get_expectation_term(mdp, state, act)
#print('\n', Q)


def substitute_line_in_policy(index_of_st, substitution):
    temp = copy.deepcopy(policy_new_dic)
    while (index_of_st != 0):
        for st in mdp.statespace:
            if min(temp[st].values()) != 0:
                for i, (key, value) in enumerate(temp[st].items()):
                    temp[st][key] = substitution[index_of_st-1, i]
                index_of_st -= 1
                break
    return temp

def get_def_policy_set(k):
    det_policy_set = []
    identity = np.eye(4)
    for comb in itertools.product(identity, repeat = k):
        det_policy_set.append(substitute_line_in_policy(k, np.stack(comb)))
    return det_policy_set

def make_det_policy(policy, det_line):
    for st in mdp.statespace:
            if min(policy[st].values()) != 0:
                for i, (key, value) in enumerate(policy[st].items()):
                    policy[st][key] = det_line[i]
    return policy

det_policy_set = get_def_policy_set(4)
for i in range(len(det_policy_set)):
    det_policy_set[i] = make_det_policy(det_policy_set[i],np.array((1,0,0,0)))

#print(det_policy_set[0])

det_m_set = []
for i in range(len(det_policy_set)):
    det_m_set.append(mdp.stactVisitFre(det_policy_set[i]))

def eval_m(m, mdp):
    R = 0
    for st in mdp.statespace:
        for act in mdp.A:
            R += m[st][act]*reward_def[st][act]
    return R

#print(eval_m(det_m_set, mdp))

# m_recover = []
# for st in mdp.statespace:
#     for act in mdp.A:
#         m_recover.append(mdp.stactVisitFre(policy_new_dic)[st][act])
# m_recover = np.array(m_recover)

# print(m_recover@R1_np)
def m2np(m):
    m_np = []
    for st in mdp.statespace:
        for act in mdp.A:
            m_np.append(m[st][act])
    m_np = np.array(m_np)
    return m_np

det_m_np = []
for i in range(len(det_m_set)):
    det_m_np.append(m2np(det_m_set[i]))

for i in range(len(det_m_np)):
    center, gamma, _ = chebyshev_center_v2.chebyshev(det_m_np[i],4,False)
    print(R1_np@det_m_np[i])
    print(center, gamma)

center, gamma, _ = chebyshev_center_v2.chebyshev(y_res,4,False)
print(R1_np@det_m_np[i])
print(center, gamma)
