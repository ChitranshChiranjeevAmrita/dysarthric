import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD

# . Time Domain 0 to T
T = 1000
fs = 1 / T
t = np.arange(1, T + 1) / T

# . center frequencies of components
f_1 = 20
f_2 = 140
f_3 = 230

# . Synthesized Signals
v_1 = (np.cos(2 * np.pi * f_1 * t))
v_2 = (np.cos(2 * np.pi * f_2 * t))
v_3 = (np.cos(2 * np.pi * f_3 * t))

# . adding them all
v = v_1 + v_2 + v_3 + 0.1 * np.random.randn(v_1.size)

#. some sample parameters for VMD
alpha = 5000      # moderate bandwidth constraint
tau = 0           # noise-tolerance (no strict fidelity enforcement)
K = 3              # 3 modes
DC = 0             # no DC part imposed
init = 1           # initialize omegas uniformly
tol = 1e-7

#. Run actual VMD code
u, u_hat, omega = VMD(v, alpha, tau, K, DC, init, tol)