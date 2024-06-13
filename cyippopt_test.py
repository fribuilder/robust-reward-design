import gradient as grad
import GridWorldV2 as Gd
import numpy as np

tau = 0.001
_, _, _, mdp = grad.test()
st_len = len(mdp.statespace)
act_len = len(mdp.A)

R1_np = np.zeros(st_len * act_len)

decoy_index = []
for decoy in mdp.F:
    decoy_index.append(mdp.statespace.index(decoy))

for i in decoy_index:
    R1_np[i*act_len:(i+1)*act_len] = 1
cons_decoy_value = np.ones((1, st_len * act_len)) - R1_np
st_len, act_len, cons_decoy_value

x = np.ones(144)

cons_eq_ele = np.zeros((act_len-1,act_len))
cons_eq_ele[:,0] = 1
for i in range(act_len-1):
    cons_eq_ele[i,i+1] = -1

cons_eq = np.kron(np.eye(st_len),cons_eq_ele)

np.concatenate((cons_decoy_value, cons_eq, 
                np.ones((1, st_len * act_len))))

y = [0]
y += [0]

print([np.ones(144)@x])