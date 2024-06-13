import cyipopt
import numpy as np
import gradient_new as grad
import pickle


class ent_reg_problem:
    def __init__(self, tau, init, mdp):
        self.tau = tau
        self.mdp = mdp
        self.st_len = len(self.mdp.statespace)
        self.act_len = len(self.mdp.A)
        self.init = init

    def objective(self, x):
        _, _, def_value = grad.ent_gradient(self.mdp, self.tau, x, self.init)
        return -def_value

    def gradient(self, x):
        x_grad_np, _, _ = grad.ent_gradient(self.mdp, self.tau, x, self.init)
        return -x_grad_np

    def constraints(self, x):
        constraints = np.ones(len(self.mdp.F)) @ x
        return constraints
    
    def jacobian(self,x):
        return np.ones(len(self.mdp.F))
    
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_dr, ls_trails):
        '''prints information at every IPOPT iteration'''


def test_cyipopt_optimize(test, perturb, tau, init, mdp, x_res):
    tau = tau
    init = init
    
    prob = ent_reg_problem(tau, init, mdp)

    x0 = grad.extract_decoy_value(x_res, prob.mdp)
    print(x0)

    if test == True:
        print('Evaluate the objective for the initial guess:')
        _, obj_val0 = grad.eval_decoy_allocation(prob.mdp, prob.tau, x0, init)
        print(obj_val0)

        att_value = prob.objective(x0)
        print(att_value)  
    
    lb = [0] * len(x0)
    ub = [2e19] * len(x0)

    cl = [0]
    cu = [4]

    ent_reg_ipopt = cyipopt.Problem(
        n = len(x0),
        m = len(cl),
        problem_obj = prob,
        lb = lb,
        ub = ub,
        cl = cl,
        cu = cu,
    )

    ent_reg_ipopt.add_option('mu_strategy', 'adaptive')
    ent_reg_ipopt.add_option('tol', 1e-4)

    x, info = ent_reg_ipopt.solve(x0)
    # print('x:',x)

    print('Evaluate the objective for the optimal allocation (boundedly rational attacker):')
    _, obj_val_opt = grad.eval_decoy_allocation(prob.mdp, prob.tau, x, init)
    print(obj_val_opt)

    print('Evaluate the objective for the optimal allocation (rational attacker):')
    _, obj_val_opt_rational = grad.eval_decoy_allocation(prob.mdp, 0.0, x, init)
    print(obj_val_opt_rational)

    print('decoy allocation in boundedly rational case:', x)

    with open ('x_ent', 'wb') as fp:
        pickle.dump(x,fp)
    return

# if __name__ == '__main__':
#     test_cyipopt_optimize(test=False, perturb=False)
