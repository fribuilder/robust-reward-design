#!/usr/bin/env python3
from itertools import product
from mip import Model, BINARY, minimize, xsum, OptimizationStatus
from MDP import MDP
import os
import argparse  # required for parsing arguments
import json


def get_save_path(save_dir, name):
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
        default=100,
    )
    parser.add_argument(
        "--nu", help="Upper bound of the value function", default="uniform"
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


def solve(mdp, nu, args):
    """
    Solve the optimal sensor allocation by formulating a MILP problem given the MDP and the nu.

    :param mdp: the MDP to solve the optimal sensor allocations
    :type mdp: mdp.MDP
    :param nu: the initial state distribution
    :type nu: list
    :param args: the arguments
    :type args: argparse.Namespace
    :return: the optimal sensor allocaitons
    :rtype: list
    """
    # for convience I create the SxAxS space
    mdp.s_a_ns = list(product(mdp.statespace, mdp.A, mdp.statespace))

    # Create an optimization model
    m = Model()

    # Decleare the decision variables
    v = [m.add_var(lb=0) for _ in enumerate(mdp.statespace)]  # V(s)
    x = [m.add_var(var_type=BINARY) for _ in enumerate(mdp.statespace)]  #  x(s)
    w = [m.add_var(lb=0) for _ in enumerate(mdp.s_a_ns)]

    # Objective function
    m.objective = minimize(xsum(nu[i] * v[i] for i, _ in enumerate(mdp.statespace)))

    for s in mdp.stotrans:
        i = mdp.statespace.index(s)  # the index of the s in S
        for a in mdp.stotrans[s]:
            try:
                m += v[i] >= xsum(
                    mdp.stotrans[s][a][ns] * w[mdp.s_a_ns.index((s, a, ns))]
                    for ns in mdp.stotrans[s][a]
                )
            except Exception as err:
                print(s, a, ns)
                print((s, a, ns) in mdp.s_a_ns)
                print(len(w), mdp.s_a_ns.index((s, a, ns)))
                print(err)
                exit(-1)

        for j, ns in enumerate(mdp.statespace):
            k = mdp.s_a_ns.index((s, a, ns))

            # index of s: i
            # index of ns: j
            # index of (s, a, ns): k
            # big-M method
            m += w[k] >= args.lower_bound * (1 - x[i])
            m += w[k] <= args.upper_bound * (1 - x[i])
            m += w[k] - (mdp.R[ns] + args.gamma * v[j]) >= args.lower_bound * x[i]
            m += w[k] - (mdp.R[ns] + args.gamma * v[j]) <= args.upper_bound * x[i]

    # # forbid allocating sensors in the goal
    # for _, ns in enumerate(mdp.G):
    #     j = mdp.statespace.index(ns)
    #     x[j] = 0

    # Set the constraint on the Num. of the IDSs
    m += (
        xsum(x[i] for i, s in enumerate(mdp.statespace) if s not in mdp.G)
        <= args.num_ids
    )

    print("=" * 10 + " Start optimization " + "=" * 10)
    m.max_gap = 1e-10
    status = m.optimize(max_seconds=300)  # Set the maximal calculation time
    if status == OptimizationStatus.OPTIMAL:
        print("optimal solution cost {} found".format(m.objective_value))
        # construct the solution
        sol = [
            mdp.statespace[i]
            for i, s in enumerate(mdp.statespace)
            if s not in mdp.G and x[i].x > 0
        ]


        # only save result if it is specified
        if args.save:
            path = get_save_path(
                args.save_dir, "sensor_allocation_{}_{}".format(args.gamma, args.num_ids)
            )
            save_file = {"objective value":m.objective_value, "sensor locations": sol, "value": [v[i].x for i, _ in enumerate(mdp.statespace)]}

            with open(path, "w") as f:
                json.dump(, f)

        print("The optimal sensor allocation: {}".format(sol))
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


def main():
    G1 = ["q11"]
    mdp = MDP()
    mdp.getgoals(G1)
    mdp.stotrans = mdp.getstochastictrans()

    args = parse_arguments()

    if args.nu == "uniform":
        # uniform distribution
        nu = [1 / len(mdp.statespace)] * len(mdp.statespace)
    else:
        nu = args.nu
    # solve the MILP problem
    sensor_allocation = solve(mdp, nu, args)


if __name__ == "__main__":
    main()
