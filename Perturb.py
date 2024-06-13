import gradient_v2 as grad
import pickle 
import numpy as np

with open('x_res', 'rb') as fp:
    x_res = pickle.load(fp)

_, _, _, mdp = grad.gen_problem_data()
x0 = grad.extract_decoy_value(x_res, mdp)
tau = 1e-2

rng = np.random.default_rng(seed=0)
ntrials = 10
pert_rel = 0.01
obj_vals_pert_rational = np.zeros(ntrials)
obj_vals_pert_bounded = np.zeros(ntrials)
x_pert_all = np.zeros((x0.size, ntrials))

def pert(pert_mode, step = 0.001):
    for i in range(ntrials):
        match pert_mode:
            case 'entrywise':
                delta_x = pert_rel * np.linalg.norm(x0) * rng.uniform(-1, 1, x0.shape)
            case 'directional':
                v = rng.standard_normal(x0.shape)
                v_norm = v / np.linalg.norm(v)  # Generate a random direction.
                delta_x = pert_rel * np.linalg.norm(x0) * v_norm
            case 'gradient':
                gradient_x0, _, _ = grad.ent_gradient(mdp, tau, x0)
                delta_x = (i+1)*step*gradient_x0
            case 'line search':
                with open('x_ent', 'rb') as fp:
                    x_ent = pickle.load(fp)
                    delta_x = np.linspace(0, x_ent-x0, ntrials + 1)[i+1]

        x_pert = np.clip(x0 + delta_x, 0, None)  # Ensure that the resulting x is entrywise nonnegative.
        x_pert_all[:,i] = x_pert
        # Evaluate x_pert under a rational attacker
        _, obj_vals_pert_rational[i] = grad.eval_decoy_allocation(mdp, 0.0, x_pert)

        # Evaluate x_pert under a boundedly rational attacker
        _, obj_vals_pert_bounded[i] = grad.eval_decoy_allocation(mdp, tau, x_pert)     

    print(obj_vals_pert_rational)
    print("Value under the best perturbation:", np.max(obj_vals_pert_rational))    

    print(obj_vals_pert_bounded)
    print("Value (boundedly rational) under the best perturbation:", np.max(obj_vals_pert_bounded))    

if __name__ == '__main__':
    pert(pert_mode='line search')
    