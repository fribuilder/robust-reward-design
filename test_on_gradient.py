import cvxpy as cp
import numpy as np

y = cp.Variable(2)
a = cp.Parameter(2)

#objective = np.array([1,1]) @ cp.power(y-a, 2)
objective = cp.sum(cp.power(y-a, 2))
prob = cp.Problem(cp.Minimize(objective))
print(prob.is_dcp(dpp=True))

a.value = np.array([0.5,0.5])
prob.solve(requires_grad=True)

def f(y, a):
    return (y-a)**2
print(y.value)

y.gradient = 2*(y.value - a.value)
prob.backward()

print(a.gradient)