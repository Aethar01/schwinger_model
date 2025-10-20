#!/usr/bin/env python

from qiskit import QuantumCircuit, transpile
import numpy as np
from qiskit_aer import AerSimulator
from matplotlib import pyplot as plt


# CONSTANTS
pi = np.pi
N = 16
g = 1.0
m = 0.5
theta = 0.0
a = 1.0
w = 1/(2*a)
J = g**2 * a / 2

# ADIABATIC PARAMETERS
T = 5
dt = 0.1
steps = int(T/dt)
m0 = 1.0


# GATES
def zz_gate(circ, i, j, coeff):
    """
    Implements exp(-i J dt Z_i Z_j / 2)
    """
    circ.cx(i, j)
    circ.rz(2 * coeff, j)
    circ.cx(i, j)


def xy_gate(circ, i, j, coeff):
    """
    Implements exp(-i w dt (X_i X_j + Y_i Y_j) / 2)
    """
    circ.cx(i, j)
    circ.rx(2 * coeff, i)
    circ.cx(i, j)

    circ.rz(-pi/2, i)
    circ.rz(-pi/2, j)
    circ.cx(i, j)
    circ.rx(2 * coeff, i)
    circ.cx(i, j)
    circ.rz(pi/2, i)
    circ.rz(pi/2, j)


def trotter_step(circ, t):
    """
    U(t) = exp(-i H_+-(t) dt / 2) exp(-i H_ZZ dt) exp(-i H_Z dt) exp(-i H_+-(t) dt / 2)
    """
    t_frac = t / T
    wt = w * t_frac
    mt = (1 - t_frac) * m0 + t_frac * m
    thetat = theta * t_frac

    # H_+- half-time evolution
    for n in range(N - 1):
        coeff = 0.5 * dt * (wt - ((-1)**n) * (mt/2) * np.sin(thetat))
        xy_gate(circ, n, n + 1, coeff)

    # H_ZZ full-time evolution
    for n in range(1, N):
        for k in range(n):
            zz_gate(circ, k, n, J * dt / 2)

    # H_Z full-time evolution
    for n in range(N):
        coeff = 0.5 * dt * (mt * np.cos(thetat) * (-1)**n)
        circ.rz(2 * coeff, n)

    # H_+- half-time evolution
    for n in range(N - 1):
        coeff = 0.5 * dt * (wt - ((-1)**n) * (mt/2) * np.sin(thetat))
        xy_gate(circ, n, n + 1, coeff)


# qc = QuantumCircuit(N)
#
# # initial satate |0101...>
# for n in range(1, N, 2):
#     qc.x(n)
#
# for step in range(steps):
#     trotter_step(qc, step * dt)
#
# qc.barrier()
# qc.measure_all()
#
# # print(qc.draw())
#
# sim = AerSimulator()
# compiled = transpile(qc, sim)
# result = sim.run(compiled, shots=1000).result()
# counts = result.get_counts()
#
# plot_histogram(counts, filename="_histogram.png")

def compute_z_expectations(counts, N, shots):
    """Return <Z_n> for each qubit n from measurement counts."""
    expectations = np.zeros(N)
    for bitstring, count in counts.items():
        bits = bitstring[::-1]
        for n, b in enumerate(bits):
            expectations[n] += (1 if b == "0" else -1) * count
    expectations /= shots
    return expectations


def chiral_condensate(expectations, a=1.0):
    N = len(expectations)
    vals = [(-1) ** n * expectations[n] for n in range(N)]
    return np.sum(vals) / (2 * N * a)


def run_sim(g, m, theta, shots=2048):
    qc = QuantumCircuit(N)
    # initial state |0101...>
    for n in range(1, N, 2):
        qc.x(n)

    for step in range(steps):
        trotter_step(qc, step * dt)

    qc.measure_all()
    sim = AerSimulator()
    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()

    # compute <Z_n> and chiral condensate
    expectations = compute_z_expectations(counts, N, shots)
    psi_bar_psi = chiral_condensate(expectations)
    return psi_bar_psi


def run_sims(g, m):
    thetas = [i * 0.05 * 2 * np.pi for i in range(0, 11)]
    results = [run_sim(g, m, theta) for theta in thetas]
    return np.array(thetas), np.array(results)


def plot_results(thetas, results):
    plt.plot(thetas / (2 * np.pi), results, "o-", label="simulation")
    plt.xlabel(r"$\theta / 2\pi$")
    plt.ylabel(r"$\langle\bar{\psi}\psi\rangle$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    thetas, results = run_sims(g=1.0, m=0.1)
    plot_results(thetas, results)
