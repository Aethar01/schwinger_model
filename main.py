#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
import argparse

from qclib.sim import run_sims_t, extrapolate_N_to_infty


def plot_results_theta(thetas, results_by_N, Ns, results_inf, **kwargs):
    plt.figure(figsize=(7, 5))
    if results_inf is not None:
        # for N, res in zip(Ns, results_by_N):
        #     plt.plot(thetas / (2 * np.pi), res, "o--", label=f"N={N}")
        plt.plot(thetas / (2 * np.pi), results_inf,
                 "o-", label="N->inf extrapolated")
    else:
        for N, res in zip(Ns, results_by_N):
            plt.plot(thetas / (2 * np.pi), res, "o-", label=f"N={N}")

    plt.title(f"(g,m,N,w) = ({kwargs['g']},{
              kwargs['m']},{kwargs['N']},{kwargs['w']})")
    plt.xlabel(r"$\theta / 2\pi$")
    plt.ylabel(
        r"$\langle\bar{\psi}\psi\rangle - \langle\bar{\psi}\psi\rangle_{\rm free}$")
    plt.legend()
    plt.tight_layout()
    if kwargs["output"]:
        plt.savefig(kwargs["output"], dpi=200)
    plt.show()


def plot_results_t(ts, results_by_t, results_inf, **kwargs):
    plt.figure(figsize=(7, 5))
    if results_inf is not None:
        plt.plot(ts, results_inf,
                 "o-", label="N->inf extrapolated")
    else:
        for res, T in zip(results_by_t, kwargs['Ts']):
            # idx = math.ceil(5 / (kwargs["dt"]))
            # ts = ts[:idx]
            # res = res[:idx]
            plt.plot(ts, res, "o-",
                     label=f"T={T}, dt={kwargs['dt']}")

    plt.title(f"(g,m,N,w) = ({kwargs['g']},{
              kwargs['m']},{kwargs['N']},{kwargs['w']})")
    plt.xlabel(r"$t$")
    plt.ylabel(
        r"$\langle\bar{\psi}\psi\rangle$")
    plt.legend()
    plt.tight_layout()
    if kwargs["output"]:
        plt.savefig(kwargs["output"], dpi=200)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # parser.add_argument("-N", "--Nqubits", type=int)
    parser.add_argument("-a", "--a", type=float,
                        default=1.0, help="Lattice spacing")
    parser.add_argument("-T", "--time", type=float,
                        default=15, help="Total time")
    parser.add_argument("-dt", "--timestep", type=float,
                        default=0.3, help="Time of each trotter step")
    parser.add_argument("-m0", "--m0", type=float, default=1.0, help="IDK")
    parser.add_argument("-g", "--coupling_constant",
                        type=float, default=1.0, help="Coupling constant")
    parser.add_argument("-m", "--mass", type=float, default=0.1, help="Mass")
    parser.add_argument("-s", "--shots", type=int,
                        default=2048, help="Number of shots")
    parser.add_argument("-i", "--inf", action="store_true",
                        help="Extrapolate number of Qubits to infinity")
    parser.add_argument("-N", "--Nqubits", type=int,
                        default=16, help="Number of Qubits")
    parser.add_argument("-w", "--w", type=float, default=None,
                        help="w (if set overwrites a)")
    parser.add_argument("-d", "--draw", action="store_true",
                        help="Draw the circuit and exit")
    parser.add_argument("plot", choices=[
                        "4", "5", "5_single"], help="Plot type, choices are which variable to plot on the x-axis")
    parser.add_argument("-o", "--output", type=str,
                        help="Save figure to this path")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print parameters")
    parser.add_argument("-t", "--theta", type=float,
                        default=0.0, help="Initial angle")
    parser.add_argument("-mt", "--max_t", type=int,
                        default=5, help="Maximum time to run")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.m0 is not None and args.m0 < 0:
        raise ValueError("m0 must be non-negative")

    if args.w is not None:
        args.a = 1.0 / (2 * args.w)
    a = args.a or 1.0
    params = {
        "N": args.Nqubits or 4,
        "a": a,
        "w": args.w or 1.0 / (2 * a),
        "T": args.time or 150,
        "dt": args.timestep or 0.3,
        "m0": args.m0 or 1.0,
        "g": args.coupling_constant or 1.0,
        "m": args.mass or 0.0,
        "shots": args.shots or 2048,
        "draw": args.draw or False,
        "output": args.output or None,
        "theta": args.theta or 0.0
    }

    if args.verbose:
        print("Parameters:")
        for k, v in params.items():
            print(f"\t{k}: {v}")

    match args.plot:
        case "4":
            print("Not implemented")
            exit(1)
        #     if args.inf:
        #         all_results = []
        #         args.Nqubits = list(range(4, args.Nqubits + 1, 4))
        #         Ns = args.Nqubits
        #         for N in Ns:
        #             print(f"Running simulations for N={N}")
        #             params["N"] = N
        #             thetas, results = run_sims_theta(**params)
        #             all_results.append(results)
        #         results_inf = extrapolate_N_to_infty(all_results, Ns)
        #         plot_results_theta(thetas, all_results, Ns,
        #                            results_inf, **params)
        #     else:
        #         Ns = [args.Nqubits]
        #         thetas, results = run_sims_theta(**params)
        #         results = [results]
        #         plot_results_theta(thetas, results, Ns, None, **params)
        case "5_single":
            if args.inf:
                print("Not implemented")
                exit(1)
            else:
                ts, results = run_sims_t(**params, max_t=args.max_t)
                results = [results]
                plot_results_t(ts, results, None, **params, Ts=[args.time])
        case "5":
            if args.inf:
                args.Nqubits = list(range(4, args.Nqubits + 1, 4))
                Ns = args.Nqubits
                Ts = [args.time, args.time + 50, args.time + 100]
                results_by_T = []
                for T in Ts:
                    print(f"Running simulations for T={T}")
                    results_by_N = []
                    for N in Ns:
                        print(f"Running simulations for N={N}")
                        params["T"] = T
                        params["N"] = N
                        ts, results = run_sims_t(**params, max_t=args.max_t)
                        results_by_N.append(results)
                    results_inf = extrapolate_N_to_infty(results_by_N, Ns)
                    results_by_T.append(results_inf)
                params["N"] = r"\inf"
                plot_results_t(ts, results_by_T, None, **params, Ts=Ts)
            else:
                Ts = [args.time, args.time + 50, args.time + 100]
                results_by_T = []
                for T in Ts:
                    print(f"Running simulations for T={T}")
                    params["T"] = T
                    ts, results = run_sims_t(**params)
                    results_by_T.append(results)
                plot_results_t(ts, results_by_T, None, **params, Ts=Ts)


if __name__ == "__main__":
    main()
