#!/usr/bin/env python

from qiskit import QuantumCircuit, transpile
import numpy as np
from qiskit_aer import AerSimulator
from matplotlib import pyplot as plt
from scipy.special import ellipk
from tqdm import tqdm
from sys import argv
import argparse


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


def xy_gate(circ, i, j, coeff):
    """
    Implements exp(-i * coeff * (X_i X_j + Y_i Y_j)).
    i => control qubit
    j => target qubit
    """
    circ = circ.copy()
    circ = xx_gate(circ, i, j, coeff)
    circ = yy_gate(circ, i, j, coeff)
    return circ


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

    def c_n(t, T, m0, m, theta, n, J):
        cn = 1/2 * ((1 - t / T) * m0 + t / T * m) * np.cos(t / T * theta) * \
            ((-1) ** n) - J / 2 * sum([k % 2 for k in range(n, N-1)])
        return cn

    def H_Z(circ):
        circ.copy()
        for n in range(N):
            cn = c_n(t, T, m0, m, theta, n, J)
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
        return circ

    def H_ZZ(circ):
        circ.copy()
        for n in range(1, N-1):
            for k in range(n):
                circ = zz_gate(circ, k, n, J * dt / 2)

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
    """Return <Z_n> for each qubit n from measurement counts."""
    expectations = np.zeros(N)
    for bitstring, count in counts.items():
        bits = bitstring[::-1]
        for n, b in enumerate(bits):
            expectations[n] += (1 if b == "0" else -1) * count
    expectations /= shots
    return expectations


def chiral_condensate(expectations, a):
    N = len(expectations)
    vals = [(-1) ** n * expectations[n] for n in range(N)]
    return np.sum(vals) / (2 * N * a)


def run_sim(g, m, theta, *, N, T, dt, m0, a, w, shots, draw: bool):
    """Run one simulation for given parameters and return the VEV - VEV_free."""
    J = g**2 * a / 2

    qc = QuantumCircuit(N)
    # Initial state |0101...>
    for n in range(1, N, 2):
        qc.x(n)

    steps = int(T / dt)
    for step in range(steps):
        qc = trotter_step(qc, step * dt, theta, N=N, T=T,
                          dt=dt, w=w, m0=m0, m=m, J=J)

    qc.measure_all()

    if draw:
        print(qc.draw(output="mpl"))
        plt.show()
        exit()

    sim = AerSimulator()
    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()

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


def run_sims(g, m, *, N, T, dt, m0, a, w, shots, draw: bool):
    """Run multiple simulations across a theta range."""
    thetas = [i * 0.05 * 2 * np.pi for i in range(0, 11)]
    results = []
    for theta in tqdm(thetas):
        res = run_sim(g, m, theta, N=N, T=T, dt=dt,
                      m0=m0, a=a, w=w, shots=shots, draw=draw)
        results.append(res)
    return np.array(thetas), np.array(results)


def extrapolate_N_to_infty(all_results, Ns):
    Ns = np.array(Ns, dtype=float)
    invN = 1.0 / Ns
    thetas = np.arange(len(all_results[0]))
    extrapolated = []

    for i in range(len(thetas)):
        y = np.array([res[i] for res in all_results])
        coeffs = np.polyfit(invN, y, 2)
        extrapolated.append(np.polyval(coeffs, 0.0))  # evaluate at 1/N = 0
    return np.array(extrapolated)


def plot_results(thetas, results_by_N, Ns, results_inf, **kwargs):
    plt.figure(figsize=(7, 5))
    if results_inf is not None:
        # for N, res in zip(Ns, results_by_N):
        #     plt.plot(thetas / (2 * np.pi), res, "o--", label=f"N={N}")
        plt.plot(thetas / (2 * np.pi), results_inf,
                 "o-", label="N->inf extrapolated")
    else:
        for N, res in zip(Ns, results_by_N):
            plt.plot(thetas / (2 * np.pi), res, "o-", label=f"N={N}")

    plt.title(f"Chiral condensate, g={kwargs['g']}, m={
              kwargs['m']}, a={kwargs['a']}")
    plt.xlabel(r"$\theta / 2\pi$")
    plt.ylabel(
        r"$\langle\bar{\psi}\psi\rangle - \langle\bar{\psi}\psi\rangle_{\rm free}$")
    plt.legend()
    plt.tight_layout()
    if Ns and results_inf is not None:
        plt.savefig("images/results_extrapolated.png", dpi=200)
    else:
        plt.savefig("images/results.png", dpi=200)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-N", "--Nqubits", type=int)
    parser.add_argument("-a", "--a", type=float)
    parser.add_argument("-T", "--time", type=float)
    parser.add_argument("-dt", "--timestep", type=float)
    parser.add_argument("-m0", "--m0", type=float)
    parser.add_argument("-g", "--coupling_constant", type=float)
    parser.add_argument("-m", "--mass", type=float)
    parser.add_argument("-s", "--shots", type=int)
    parser.add_argument("-i", "--inf", action="store_true")
    parser.add_argument("-N", "--Nqubits", type=int)
    parser.add_argument("-d", "--draw", action="store_true", help="Draw the circuit and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.m0 is not None and args.m0 < 0:
        raise ValueError("m0 must be non-negative")

    a = args.a or 1.0
    params = {
        # "N": args.Nqubits or 16,
        "a": a,
        "w": 1.0 / (2 * a),
        "T": args.time or 150,
        "dt": args.timestep or 0.3,
        "m0": args.m0 or 1.0,
        "g": args.coupling_constant or 1.0,
        "m": args.mass or 0.0,
        "shots": args.shots or 2048,
        "draw": args.draw or False
    }

    if args.Nqubits and args.inf:
        Ns = list(range(4, args.Nqubits + 1, 4))
    else:
        if args.Nqubits:
            Ns = [args.Nqubits]
        else:
            Ns = [4]

    if args.inf:
        all_results = []
        for N in Ns:
            print(f"Running simulations for N={N}")
            thetas, results = run_sims(N=N, **params)
            all_results.append(results)
        results_inf = extrapolate_N_to_infty(all_results, Ns)
        plot_results(thetas, all_results, Ns, results_inf, **params)
    else:
        thetas, results = run_sims(N=Ns[0], **params)
        results = [results]
        plot_results(thetas, results, Ns, None, **params)


if __name__ == "__main__":
    main()
