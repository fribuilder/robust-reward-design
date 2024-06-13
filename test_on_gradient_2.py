import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer

y = cp.Variable(2)
a = cp.Parameter(2)
objective = cp.Minimize(cp.sum(cp.power((y - 2* np.array([[1,2],[2,4]]) @ a), 2)))
problem = cp.Problem(objective)
assert problem.is_dpp()
cvxpylayer = CvxpyLayer(problem, parameters=[a], variables=[y])

a = 2 * np.ones(2)
a_tch = torch.tensor(a)
a_tch.requires_grad_(True)

[y_tch] = cvxpylayer(a_tch, solver_args={"eps": 1e-8})

#test on define high level problem
b = np.ones(2,)
b_tch = torch.from_numpy(b)

sum_y = y_tch[0]**2
print(sum_y)
sum_y.backward()

a_np = a_tch.detach().numpy()
a_grad_np = a_tch.grad.detach().numpy()
print(a_np, a_grad_np)