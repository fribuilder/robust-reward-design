#!/usr/bin/env python3
from itertools import product
from mip import Model, BINARY, minimize, xsum, OptimizationStatus
from MDP import MDP
import GridWorld
import os  # requried for system path
import argparse  # required for parsing arguments
import json  # required for saving the human readable results


def get_save_path(save_dir, name):
    """
    Return the path to the save file.

    :param save_dir: the path to the save directory
    :type save_dir: str
    :param name: the name of the save file
    :type name: str
    :return: the path to the save file
    :rtype: str
    """
    return os.path.join(save_dir, "{}.json".format(name))


def parse_arguments():
    """
    Parse and return argments

    :return: argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        "We formulate MILP problem to solve the optimal sensor allocation problem."
    )

    parser.add_argument("--gamma", type=float, help="Discounting factor", default=0.95)
    parser.add_argument(
        "-lb",
        "--lower-bound",
        type=str,
        help="Lower bound of the value function",
        default=-1,
    )
    parser.add_argument(
        "-ub",
        "--upper-bound",
        type=float,
        help="Upper bound of the value function",
        default=1,
    )
    parser.add_argument(
        "--nu",
        type=str,
        help="Upper bound of the value function",
    )
    parser.add_argument(
        "-n",
        "--num-ids",
        type=int,
        help="The Num. of sensors available to be allocated",
        required=True,
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.getcwd(),
        help="If specified, json saved in this directory.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="If specified, store the results.",
    )
    # parse the args
    args = parser.parse_args()

    return args


def solve(mdp, args):
    """
    Solve the optimal sensor allocation by formulating a MILP problem given the MDP and the nu.

    :param mdp: the MDP to solve the optimal sensor allocations
    :type mdp: mdp.MDP
    :param args: the arguments
    :type args: argparse.Namespace
    """
    if args.nu is None:
        # uniform distribution
        nu = [1 / len(mdp.statespace)] * len(mdp.statespace)
    else:
        nu = [float(i) for i in args.nu.split(",")]
        nu = [float(i) / sum(nu) for i in nu]
    print("The initial distribution nu: {}".format(nu))
    assert len(mdp.statespace) == len(nu)

    # for convience I create the SxAxS space
    mdp.s_a_ns = list(product(mdp.statespace, mdp.A, mdp.statespace))

    # Create an optimization model
    m = Model()

    # Decleare the decision variables
    v = [m.add_var(lb=0, ub=1) for _ in enumerate(mdp.statespace)]  # V(s)
    x = [m.add_var(var_type=BINARY) for _ in enumerate(mdp.statespace)]  #  x(s)
    w = [m.add_var(lb=0) for _ in enumerate(mdp.s_a_ns)]

    # Objective function
    m.objective = minimize(xsum(nu[i] * v[i] for i, _ in enumerate(mdp.statespace)))

    for i, s in enumerate(mdp.statespace):
        if s in mdp.G:
            m += v[i] == 1  # values of final states are 1
        for a in mdp.A:
            m += v[i] >= xsum(
                mdp.stotrans[s][a][ns] * w[mdp.s_a_ns.index((s, a, ns))]
                for ns in mdp.stotrans[s][a]
            )
            for j, ns in enumerate(mdp.statespace):
                k = mdp.s_a_ns.index((s, a, ns))

                # index of s: i
                # index of ns: j
                # index of (s, a, ns): k
                # big-M method
                m += w[k] >= args.lower_bound * (1 - x[i])
                m += w[k] <= args.upper_bound * (1 - x[i])
                m += w[k] - (args.gamma * v[j]) >= args.lower_bound * x[i]
                m += w[k] - (args.gamma * v[j]) <= args.upper_bound * x[i]

    print("U: {}".format(mdp.U))
    print("G: {}".format(mdp.G))
    # we do not allow placing sensors in U
    for _, s in enumerate(mdp.U):
        m += x[mdp.statespace.index(s)] == 0

    # Set the constraint on the Num. of the IDSs
    print("num: {}".format(args.num_ids))
    m += xsum(x[i] for i, s in enumerate(mdp.statespace)) <= args.num_ids

    m.max_gap = 1e-10
    print("=" * 10 + " Start optimization " + "=" * 10)
    status = m.optimize(max_seconds=300)  # Set the maximal calculation time
    if status == OptimizationStatus.OPTIMAL:
        print("optimal solution cost {} found".format(m.objective_value))
        # construct the solution
        sensor_allcitions = [
            s for i, s in enumerate(mdp.statespace)
            if s not in mdp.U and x[i].x > 0
        ]
        print({s:x[i].x for i, s in enumerate(mdp.statespace) if x[i].x > 0})
        print("The optimal sensor allocation: {}".format(sensor_allcitions))

        # only save result if it is specified
        if args.save:
            path = get_save_path(
                args.save_dir,
                "sensor_allocation_{}_{}".format(args.gamma, args.num_ids),
            )
            detailed_sol = {
                "objective value": m.objective_value,
                "sensor locations": sensor_allcitions,
                "value": {"q" + str(i): v[i].x for i, _ in enumerate(mdp.statespace)},
                "args": vars(args),
            }
            # write into file
            with open(path, "w") as f:
                json.dump(detailed_sol, f)

    elif status == OptimizationStatus.FEASIBLE:
        print(
            "sol.cost {} found, best possible: {}".format(
                m.objective_value, m.objective_bound
            )
        )
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print(
            "no feasible solution found, lower bound is: {}".format(m.objective_bound)
        )
    elif status == OptimizationStatus.INFEASIBLE:
        print("The problem is infeasible.")


def main(args):
    """
    Main function
    :param args: the arguments
    :type args: argparse.Namespace.
    """

    G1 = ["q12"]
    F1 = []
    mdp = MDP()
    mdp.getgoals(G1)
    mdp.getfakegoals(F1)
    mdp.stotrans = mdp.getstochastictrans()

    if os.path.exists(args.save_dir):
        print("Warning, dir already exists, files may be overwritten.")
    else:
        print("Creating dir since it does not exist.")
        os.makedirs(args.save_dir)

    # solve the MILP problem
    solve(mdp, args)


def GridWorldCase(args):
    """
    Function used to test GridWorld Case
    """
    gridworld, V, policy = GridWorld.createGridWorldBarrier()
    if os.path.exists(args.save_dir):
        print("Warning, dir already exists, files may be overwritten.")
    else:
        print("Creating dir since it does not exist.")

    solve(gridworld, args)


if __name__ == "__main__":
    # parse the arguments
    args = parse_arguments()
    #    main(args)
    GridWorldCase(args)
