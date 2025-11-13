from qiskit import QuantumCircuit, transpile
import numpy as np
from qiskit_aer import AerSimulator
from matplotlib import pyplot as plt
from scipy.special import ellipk
from tqdm import tqdm
from qiskit.quantum_info import Statevector
import pickle
import hashlib
import json
import inspect
from pathlib import Path

from .lib import c_n, trotter_step, compute_z_expectations, chiral_condensate


def _hash_source(*funcs):
    """Hash the source code of given functions for cache invalidation."""
    source_str = "".join(inspect.getsource(f) for f in funcs)
    return hashlib.sha256(source_str.encode()).hexdigest()


def _circuit_cache_key(params, t):
    """Create a unique hash key for the transpiled circuit based on parameters, time, and function sources."""
    relevant_params = {k: params[k] for k in [
        "N", "T", "dt", "m0", "a", "w", "m", "theta", "J", "shots"
    ]}
    relevant_params["t"] = t

    param_str = json.dumps(relevant_params, sort_keys=True)
    code_hash = _hash_source(c_n, trotter_step)
    combined_str = param_str + code_hash
    return hashlib.sha256(combined_str.encode()).hexdigest()


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


def setup_or_load_simulation(params, t):
    # LOAD
    _cache_dir = params["cache_dir"]
    relevant_params = {k: params[k] for k in [
        "N", "T", "dt", "m0", "a", "w", "m", "theta", "J", "shots"
    ]}
    relevant_params["t"] = t
    param_str = json.dumps(relevant_params, sort_keys=True)
    code_hash = _hash_source(c_n, trotter_step)
    key = hashlib.sha256((param_str + code_hash).encode()).hexdigest()
    cache_file = _cache_dir / f"{key}.pkl"

    # Load cached transpiled circuit if available
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            transpiled_qc = pickle.load(f)
        return transpiled_qc

    # OR BUILD
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

    sim = AerSimulator()

    # print("Transpiling circuits...")
    transpiled_qc = transpile(qc, sim)

    with open(cache_file, "wb") as f:
        pickle.dump(transpiled_qc, f)

    return transpiled_qc


def run_sim(qcs, params):
    sim = AerSimulator()

    print("Running simulations...")
    all_counts = [
        sim.run(qc, shots=params["shots"]).result().get_counts()
        for qc in tqdm(qcs)
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


# def run_sim(qcs, params):
#     sim = AerSimulator()
#
#     print("Transpiling circuits...")
#
#     compiled = []
#     for qc in tqdm(qcs):
#         # compute cache key per circuit
#         key = _circuit_cache_key(
#             params, t=0)
#         cache_file = _cache_dir / f"{key}.pkl"
#
#         if cache_file.exists():
#             with open(cache_file, "rb") as f:
#                 transpiled_qc = pickle.load(f)
#         else:
#             transpiled_qc = transpile(qc, sim)
#             with open(cache_file, "wb") as f:
#                 pickle.dump(transpiled_qc, f)
#         compiled.append(transpiled_qc)
#
#     print("Running simulations...")
#     all_counts = [
#         sim.run(qc, shots=params["shots"]).result().get_counts()
#         for qc in tqdm(compiled)
#     ]
#
#     all_expectations = [
#         compute_z_expectations(counts, params["N"], params["shots"])
#         for counts in all_counts
#     ]
#
#     all_psi_bar_psi = [
#         chiral_condensate(expectations, params["a"])
#         for expectations in all_expectations
#     ]
#
#     if params["m"] != 0:
#         psi_free = -(params["m"] * np.cos(params["theta"]) / np.pi) * \
#             1 / np.sqrt(1 + (params["m"]*params["a"]*np.cos(params["theta"]))**2) * \
#             ellipk((1 - (params["m"]*params["a"]*np.sin(params["theta"]))**2) /
#                    (1 + (params["m"]*params["a"]*np.cos(params["theta"]))**2))
#     else:
#         psi_free = 0
#
#     psi_bar_psi_minus_free = np.array(all_psi_bar_psi) - psi_free
#     return psi_bar_psi_minus_free


def run_sims_t(**params):
    dt = params["dt"]
    max_t = params["max_t"]
    ts = [i * dt for i in range(int(max_t / dt))]
    print("Building circuits...")
    qcs = [setup_or_load_simulation(params, t) for t in tqdm(ts)]

    if params["draw"]:
        print(qcs[1].draw(output="mpl"))
        plt.show()
        exit()

    res = run_sim(qcs, params)
    return np.array(ts), np.array(res)


def run_sims_theta(**params):
    thetas = [i * 2 * np.pi for i in np.arange(0, 0.51, 0.05)]
    print("Building circuits...")
    qcs = []
    for theta in tqdm(thetas):
        params["theta"] = theta
        qcs.append(setup_or_load_simulation(params, params["T"]))

    if params["draw"]:
        print(qcs[1].draw(output="mpl"))
        plt.show()
        exit()

    res = run_sim(qcs, params)
    return np.array(thetas), np.array(res)


def run_sims_g(**params):
    gs = [i for i in np.arange(0, 2.1, 0.25)]
    print("Building circuits...")
    qcs = []
    for g in tqdm(gs):
        params["g"] = g
        qcs.append(setup_or_load_simulation(params, params["T"]))

    if params["draw"]:
        print(qcs[1].draw(output="mpl"))
        plt.show()
        exit()

    res = run_sim(qcs, params)
    return np.array(gs), np.array(res)
