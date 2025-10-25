#!/usr/bin/env python

from qiskit import QuantumCircuit, transpile
import numpy as np
from qiskit_aer import AerSimulator
from matplotlib import pyplot as plt
from scipy.special import ellipk
from tqdm import tqdm
import argparse
from qiskit.quantum_info import Statevector


def zz_gate(circ, i, j, coeff):
    """
    Implements exp(-i * coeff * Z_i Z_j).
    i => control qubit
    j => target qubit
    """
    circ = circ.copy()
    circ.cx(i, j)
    circ.rz(coeff * 2, j)
    circ.cx(i, j)
    return circ


def xx_gate(circ, i, j, coeff):
    """
    Implements exp(-i * coeff * X_i X_j).
    i => control qubit
    j => target qubit
    """
    circ = circ.copy()
    circ.cx(j, i)
    circ.rx(coeff * 2, j)
    circ.cx(j, i)
    return circ


def yy_gate(circ, i, j, coeff):
    """
    Implements exp(-i * coeff * Y_i Y_j).
    i => control qubit
    j => target qubit
    """
    circ = circ.copy()
    pi = np.pi
    circ.rz(-pi / 2, i)
    circ.rz(-pi / 2, j)
    circ = xx_gate(circ, i, j, coeff)
    circ.rz(pi / 2, i)
    circ.rz(pi / 2, j)
    return circ


def xxyy_gate(circ, i, j, coeff):
    """
    Implements exp(-i * coeff * (X_i X_j + Y_i Y_j)).
    i => control qubit
    j => target qubit
    """
    circ = circ.copy()
    circ = xx_gate(circ, i, j, coeff)
    circ = yy_gate(circ, i, j, coeff)
    return circ


def c_n(t, T, m0, m, theta, n, J, N):
    cn = 1/2 * ((1 - t / T) * m0 + t / T * m) * np.cos(t / T * theta) * \
        ((-1) ** n) - J / 2 * sum([k % 2 for k in range(n, N-1)])
    return cn


def trotter_step(circ, t, theta, *, N, T, dt, w, m0, m, J):
    """
    Performs one Trotter step:
    U(t) = exp(-i H_+-(t) dt / 2) exp(-i H_ZZ dt) exp(-i H_Z dt) exp(-i H_+-(t) dt / 2)
    """
    circ = circ.copy()

    def w_bar(t, T, w, n, m0, m, theta):
        wbar = t / T * w - (((-1) ** n) / 2) * ((1 - t / T)
                                                * m0 + m) * np.sin(theta * t / T)
        return wbar

    def H_Z(circ):
        circ.copy()
        for n in range(N):
            cn = c_n(t, T, m0, m, theta, n, J, N)
            circ.rz(2 * cn * dt, n)
        return circ

    def H_plus_minus_half(circ):
        circ.copy()
        for n in range(0, N-1, 2):
            wbar = w_bar(t, T, w, n, m0, m, theta)
            circ = yy_gate(circ, n + 1, n, wbar * 0.5 * dt / 2)
        for n in range(1, N-1, 2):
            wbar = w_bar(t, T, w, n, m0, m, theta)
            circ = yy_gate(circ, n + 1, n, wbar * 0.5 * dt / 2)

        circ.barrier()

        for n in range(0, N-1, 2):
            wbar = w_bar(t, T, w, n, m0, m, theta)
            circ = xx_gate(circ, n + 1, n, wbar * 0.5 * dt / 2)
        for n in range(1, N-1, 2):
            wbar = w_bar(t, T, w, n, m0, m, theta)
            circ = xx_gate(circ, n + 1, n, wbar * 0.5 * dt / 2)

        # for n in range(0, N-1, 2):
        #     wbar = w_bar(t, T, w, n, m0, m, theta) * 0.5
        #     circ = xxyy_gate(circ, n + 1, n, wbar * dt / 2)
        # for n in range(1, N-1, 2):
        #     wbar = w_bar(t, T, w, n, m0, m, theta) * 0.5
        #     circ = xxyy_gate(circ, n + 1, n, wbar * dt / 2)
        return circ

    def H_ZZ(circ):
        circ.copy()
        for n in range(0, N-1):
            for k in range(n):
                circ = zz_gate(circ, n, k, J * dt / 2)
        return circ

    circ.barrier()
    circ.barrier()

    circ = H_plus_minus_half(circ)
    circ.barrier()
    circ.barrier()

    circ = H_Z(circ)
    circ.barrier()
    circ.barrier()

    circ = H_ZZ(circ)
    circ.barrier()
    circ.barrier()

    circ = H_plus_minus_half(circ)

    return circ


def compute_z_expectations(counts, N, shots):
    """
    Return <Z_n> for each qubit n from measurement counts.
    <Z_n> = P(0)_n - P(1)_n
    <Z_n> = (N_0 - N_1) / (N_0 + N_1)
    """
    expectations = np.zeros(N)
    for bitstring, count in counts.items():
        bits = bitstring[::-1]
        for n, b in enumerate(bits):
            expectations[n] += (1 if b == "0" else -1) * count
    expectations /= shots
    return expectations


def chiral_condensate(expectations, a):
    """
    <psi_bar psi> = (1 / (2 * N * a)) * sum_n ((-1) ** n) <Z_n>
    """
    N = len(expectations)
    vals = [(-1) ** n * expectations[n] for n in range(N)]
    return np.sum(vals) / (2 * N * a)


def run_sim(g, m, theta, *, N, T, dt, m0, a, w, shots, draw: bool, t=None):
    """Run one simulation for given parameters and return the VEV - VEV_free."""
    J = g**2 * a / 2

    qc = QuantumCircuit(N)

    dim = 2 ** N

    state_vector = np.zeros(dim, dtype=complex)

    for n in range(N):
        state_vector[n] = c_n(t, T, m0, m, theta, n, J, N)

    norm = np.linalg.norm(state_vector)

    if norm < 1e-9:
        print("Warning: State vector norm is close to zero.")
        exit(1)
    else:
        normalized_state = state_vector / norm

    state = Statevector(normalized_state)
    qc.initialize(state)

    # Initial state |0101...>
    # for n in range(1, N, 2):
    #     qc.x(n)

    for n in range(N):
        cn = c_n(t, T, m0, m, theta, n, J, N)
        qc.rz(2 * cn * dt, n)

    steps = int(t / dt)

    for step in range(steps):
        qc = trotter_step(qc, step * dt, theta, N=N, T=T,
                          dt=dt, w=w, m0=m0, m=m, J=J)

    qc.measure_all()

    # print(steps)
    if steps == 1 and draw:
        print(qc.draw(output="mpl"))
        plt.show()
        exit()

    sim = AerSimulator()
    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()

    # print(f"t={t}, counts={counts}")

    expectations = compute_z_expectations(counts, N, shots)
    psi_bar_psi = chiral_condensate(expectations, a)
    if m != 0:
        psi_free = -(m * np.cos(theta) / np.pi) * \
            1 / np.sqrt(1 + (m*a*np.cos(theta))**2) * \
            ellipk((1 - (m*a*np.sin(theta))**2) / (1 + (m*a*np.cos(theta))**2))
    else:
        psi_free = 0

    psi_bar_psi_minus_free = psi_bar_psi - psi_free
    return psi_bar_psi_minus_free


def run_sims_theta(g, m, *, N, T, dt, m0, a, w, shots, draw: bool, **kwargs):
    """Run multiple simulations across a theta range."""
    thetas = [i * 0.05 * 2 * np.pi for i in range(0, 11)]
    results = []
    t = T / dt
    for theta in tqdm(thetas):
        res = run_sim(g, m, theta, N=N, T=T, dt=dt,
                      m0=m0, a=a, w=w, shots=shots, draw=draw, t=t)
        results.append(res)
    return np.array(thetas), np.array(results)


def run_sims_t(g, m, *, N, T, dt, m0, a, w, shots, draw: bool, **kwargs):
    """Run multiple simulations across a t range."""
    ts = [i * dt for i in range(int(5 / dt))]
    # print(ts)
    results = []
    for t in tqdm(ts):
        res = run_sim(g, m, kwargs["theta"], N=N, T=T, dt=dt,
                      m0=m0, a=a, w=w, shots=shots, draw=draw, t=t)
        results.append(res)
    return np.array(ts), np.array(results)


def extrapolate_N_to_infty(all_results, Ns):
    Ns = np.array(Ns, dtype=float)
    invN = 1.0 / Ns
    independent_var = np.arange(len(all_results[0]))
    extrapolated = []

    for i in range(len(independent_var)):
        y = np.array([res[i] for res in all_results])
        coeffs = np.polyfit(invN, y, 2)
        extrapolated.append(np.polyval(coeffs, 0.0))  # evaluate at 1/N = 0
    return np.array(extrapolated)


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
        for res in results_by_t:
            # idx = math.ceil(5 / (kwargs["dt"]))
            # ts = ts[:idx]
            # res = res[:idx]
            plt.plot(ts, res, "o-",
                     label=f"T={kwargs['T']}, dt={kwargs['dt']}")

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
                        "4", "5"], help="Plot type, choices are which variable to plot on the x-axis")
    parser.add_argument("-o", "--output", type=str,
                        help="Save figure to this path")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print parameters")
    parser.add_argument("-t", "--theta", type=float,
                        default=0.0, help="Initial angle")
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
            if args.inf:
                all_results = []
                args.Nqubits = list(range(4, args.Nqubits + 1, 4))
                Ns = args.Nqubits
                params.clear("N")
                for N in Ns:
                    print(f"Running simulations for N={N}")
                    thetas, results = run_sims_theta(N=N, **params)
                    all_results.append(results)
                results_inf = extrapolate_N_to_infty(all_results, Ns)
                plot_results_theta(thetas, all_results, Ns,
                                   results_inf, **params)
            else:
                Ns = [args.Nqubits]
                thetas, results = run_sims_theta(**params)
                results = [results]
                plot_results_theta(thetas, results, Ns, None, **params)
        case "5":
            if args.inf:
                exit(1)
            else:
                ts, results = run_sims_t(**params)
                results = [results]
                plot_results_t(ts, results, None, **params)


if __name__ == "__main__":
    main()
