import numpy as np
from qiskit.circuit import QuantumCircuit


def zz_gate(circ: QuantumCircuit, i, j, coeff) -> QuantumCircuit:
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


def xx_gate(circ: QuantumCircuit, i, j, coeff) -> QuantumCircuit:
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


def yy_gate(circ: QuantumCircuit, i, j, coeff) -> QuantumCircuit:
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


# def xxyy_gate(circ, i, j, coeff):
#     """
#     Implements exp(-i * coeff * (X_i X_j + Y_i Y_j)).
#     i => control qubit
#     j => target qubit
#     """
#     circ = circ.copy()
#     circ = xx_gate(circ, i, j, coeff)
#     circ = yy_gate(circ, i, j, coeff)
#     return circ


def c_n(t, T, m0, m, theta, n, J, N):
    cn = 1/2 * ((1 - t / T) * m0 + t / T * m) * np.cos(t / T * theta) * \
        ((-1) ** n) - J / 2 * sum([k % 2 for k in range(n, N-1)])
    return cn


def trotter_step(circ: QuantumCircuit, t, theta, *, N, T, dt, w, m0, m, J) -> QuantumCircuit:
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
        circ = circ.copy()
        for n in range(N):
            cn = c_n(t, T, m0, m, theta, n, J, N)
            circ.rz(2 * cn * dt, n)
        return circ

    def H_plus_minus_half(circ):
        circ = circ.copy()
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
        circ = circ.copy()
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
        # bits = bitstring
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
    return np.sum(vals) / (N * a)
