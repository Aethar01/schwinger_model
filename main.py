#!/usr/bin/env python

from qiskit import QuantumCircuit, transpile
import numpy as np
from qiskit_aer import AerSimulator
from matplotlib import pyplot as plt
from scipy.special import ellipk
from tqdm import tqdm
from sys import argv


def zz_gate(circ, i, j, coeff):
    """Implements exp(-i J dt Z_i Z_j / 2)."""
    circ = circ.copy()
    circ.cx(i, j)
    circ.rz(2 * coeff, j)
    circ.cx(i, j)
    return circ


def xy_gate(circ, i, j, coeff):
    """Implements exp(-i w dt (X_i X_j + Y_i Y_j) / 2)."""
    circ = circ.copy()
    pi = np.pi
    circ.cx(i, j)
    circ.rx(2 * coeff, i)
    circ.cx(i, j)

    circ.rz(-pi / 2, i)
    circ.rz(-pi / 2, j)
    circ.cx(i, j)
    circ.rx(2 * coeff, i)
    circ.cx(i, j)
    circ.rz(pi / 2, i)
    circ.rz(pi / 2, j)
    return circ


def trotter_step(circ, t, theta, *, N, T, dt, w, m0, m, J):
    """
    Performs one Trotter step:
    U(t) = exp(-i H_+-(t) dt / 2) exp(-i H_ZZ dt) exp(-i H_Z dt) exp(-i H_+-(t) dt / 2)
    """
    circ = circ.copy()
    t_frac = t / T
    wt = w * t_frac
    mt = (1 - t_frac) * m0 + t_frac * m
    thetat = theta * t_frac

    # H_+- half-time evolution
    for n in range(N - 1):
        coeff = 0.5 * dt * (wt - ((-1) ** n) * (mt / 2) * np.sin(thetat))
        circ = xy_gate(circ, n, n + 1, coeff)

    # H_ZZ full-time evolution
    for n in range(1, N):
        for k in range(n):
            circ = zz_gate(circ, k, n, J * dt / 2)

    # H_Z full-time evolution
    for n in range(N):
        coeff = 0.5 * dt * (mt * np.cos(thetat) * (-1) ** n)
        circ.rz(2 * coeff, n)

    # H_+- half-time evolution
    for n in range(N - 1):
        coeff = 0.5 * dt * (wt - ((-1) ** n) * (mt / 2) * np.sin(thetat))
        circ = xy_gate(circ, n, n + 1, coeff)

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


def run_sim(g, m, theta, *, N, T, dt, m0, a, w, shots=2048):
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

    sim = AerSimulator()
    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()

    expectations = compute_z_expectations(counts, N, shots)
    psi_bar_psi = chiral_condensate(expectations, a)
    psi_free = -(m * np.cos(theta) / np.pi) * \
        1 / np.sqrt(1 + (m*a*np.cos(theta))**2) * \
        ellipk((1 - (m*a*np.sin(theta))**2) / (1 + (m*a*np.cos(theta))**2))
    psi_bar_psi_minus_free = psi_bar_psi - psi_free
    return psi_bar_psi_minus_free


def run_sims(g, m, *, N, T, dt, m0, a, w):
    """Run multiple simulations across a theta range."""
    thetas = [i * 0.05 * 2 * np.pi for i in range(0, 11)]
    results = []
    for theta in tqdm(thetas):
        res = run_sim(g, m, theta, N=N, T=T, dt=dt, m0=m0, a=a, w=w)
        results.append(res)
    return np.array(thetas), np.array(results)


def plot_results(thetas, results, **kwargs):
    plt.plot(thetas / (2 * np.pi), results, "o-", label="simulation")
    plt.title(f"Paper figure 4 with T={kwargs['T']}, dt={
              kwargs['dt']}, g={kwargs['g']}, m={kwargs['m']}")
    plt.xlabel(r"$\theta / 2\pi$")
    plt.ylabel(r"$\langle\bar{\psi}\psi\rangle$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/results.png", dpi=200)
    plt.show()


def main():
    a = 1.0
    params = {
        "N": 16,
        "a": a,
        "w": 1.0 / (2 * a),
        "T": 150,
        "dt": 0.3,
        "m0": 1.0,
        "g": 1.0,
        "m": 0.1
    }

    thetas, results = run_sims(**params)
    plot_results(thetas, results, **params)


if __name__ == "__main__":
    main()
