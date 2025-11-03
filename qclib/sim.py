from qiskit import QuantumCircuit, transpile
import numpy as np
from qiskit_aer import AerSimulator
from matplotlib import pyplot as plt
from scipy.special import ellipk
from tqdm import tqdm
from qiskit.quantum_info import Statevector

from .lib import c_n, trotter_step, compute_z_expectations, chiral_condensate


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


def setup_simulation(params, t):
    N = params["N"]
    T = params["T"]
    dt = params["dt"]
    m0 = params["m0"]
    a = params["a"]
    w = params["w"]
    g = params["g"]
    m = params["m"]
    theta = params["theta"]
    J = params["J"]
    qc = QuantumCircuit(N)
    dim = 2 ** N
    state_vector = np.zeros(dim, dtype=complex)
    for n in range(N):
        state_vector[n] = c_n(t=0, T=T, m0=m0, m=m, theta=theta, n=n, J=J, N=N)
    norm = np.linalg.norm(state_vector)
    if norm < 1e-9:
        print("Warning: State vector norm is close to zero.")
        exit(1)
    else:
        normalized_state = state_vector / norm
    state = Statevector(normalized_state)
    qc.initialize(state)

    # for n in [0, 1, 2]:
    #     qc.x(n)
    # for n in range(0, N, 2):
    #     qc.x(n)

    steps = int(t / dt)
    for step in range(steps):
        qc = trotter_step(qc, step * dt, theta, N=N, T=T,
                          dt=dt, w=w, m0=m0, m=m, J=J)
    qc.measure_all()
    return qc


def run_sim(qcs, params):
    sim = AerSimulator()

    print("Transpiling circuits...")
    compiled = [
        transpile(qc, sim)
        for qc in tqdm(qcs)
    ]

    print("Running simulations...")
    all_counts = [
        sim.run(qc, shots=params["shots"]).result().get_counts()
        for qc in tqdm(compiled)
    ]

    all_expectations = [
        compute_z_expectations(counts, params["N"], params["shots"])
        for counts in all_counts
    ]

    all_psi_bar_psi = [
        chiral_condensate(expectations, params["a"])
        for expectations in all_expectations
    ]

    if params["m"] != 0:
        psi_free = -(params["m"] * np.cos(params["theta"]) / np.pi) * \
            1 / np.sqrt(1 + (params["m"]*params["a"]*np.cos(params["theta"]))**2) * \
            ellipk((1 - (params["m"]*params["a"]*np.sin(params["theta"]))**2) /
                   (1 + (params["m"]*params["a"]*np.cos(params["theta"]))**2))
    else:
        psi_free = 0

    psi_bar_psi_minus_free = np.array(all_psi_bar_psi) - psi_free
    return psi_bar_psi_minus_free


def run_sims_t(**params):
    dt = params["dt"]
    max_t = params["max_t"]
    ts = [i * dt for i in range(int(max_t / dt))]
    print("Building circuits...")
    qcs = [setup_simulation(params, t) for t in tqdm(ts)]

    if params["draw"]:
        print(qcs[1].draw(output="mpl"))
        plt.show()
        exit()

    res = run_sim(qcs, params)
    return np.array(ts), np.array(res)
