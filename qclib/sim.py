from qiskit import QuantumCircuit, transpile
import numpy as np
from qiskit_aer import AerSimulator
from matplotlib import pyplot as plt
from scipy.special import ellipk
from tqdm import tqdm
from qiskit.quantum_info import Statevector

from .lib import c_n, trotter_step, compute_z_expectations, chiral_condensate


def run_sim(g, m, theta, *, N, T, dt, m0, a, w, shots, draw: bool, t=None):
    """Run one simulation for given parameters and return the VEV - VEV_free."""
    J = g**2 * a / 2

    qc = QuantumCircuit(N)

    dim = 2 ** N

    state_vector = np.zeros(dim, dtype=complex)

    for n in range(N):
        state_vector[n] = c_n(0, T, m0, m, theta, n, J, N)

    norm = np.linalg.norm(state_vector)

    if norm < 1e-9:
        print("Warning: State vector norm is close to zero.")
        exit(1)
    else:
        normalized_state = state_vector / norm

    state = Statevector(normalized_state)
    # print(state)
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
    t = T
    for theta in tqdm(thetas):
        res = run_sim(g, m, theta, N=N, T=T, dt=dt,
                      m0=m0, a=a, w=w, shots=shots, draw=draw, t=t)
        results.append(res)
    return np.array(thetas), np.array(results)


def run_sims_t(g, m, *, N, T, dt, m0, a, w, shots, draw: bool, **kwargs):
    """Run multiple simulations across a t range."""
    ts = [i * dt for i in range(int(kwargs["max_t"] / dt))]
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
