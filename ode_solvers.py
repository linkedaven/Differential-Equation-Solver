"""
Six ODE solvers, from scratch (except DOP853), compared on a test problem
with a known exact solution.

Test problem (classic Burden & Faires example -- nonlinear, closed-form
exact solution, good for checking order of convergence):

    y' = y - t^2 + 1,   y(0) = 0.5,   0 <= t <= 2
    exact: y(t) = (t + 1)^2 - 0.5*e^t

All fixed-step methods share the same signature:
    t, y = solver(f, t0, y0, t_end, h)

and work for scalar y (float) or vector y (numpy array), since y is always
promoted to a numpy array internally -- so the exact same functions can be
dropped into a system of ODEs (e.g. projectile motion, a pendulum, ...).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ----------------------------------------------------------------------
# Test problem
# ----------------------------------------------------------------------
def f(t, y):
    return y - t**2 + 1

def y_exact(t):
    return (t + 1)**2 - 0.5 * np.exp(t)

t0, y0, t_end = 0.0, 0.5, 2.0

# ========================================================================
# 1) EULER METHOD  (explicit, 1st order)
#       y_{n+1} = y_n + h*f(t_n, y_n)
# ========================================================================
def euler(f, t0, y0, t_end, h):
    n = int(round((t_end - t0) / h))
    t = t0 + h * np.arange(n + 1)
    y = np.zeros((n + 1,) + np.shape(y0))
    y[0] = y0
    for i in range(n):
        y[i + 1] = y[i] + h * np.asarray(f(t[i], y[i]))
    return t, y

# ========================================================================
# 2) MODIFIED EULER METHOD  (Heun's predictor-corrector, 2nd order)
#       predictor:  y* = y_n + h*f(t_n, y_n)
#       corrector:  y_{n+1} = y_n + h/2*(f(t_n,y_n) + f(t_{n+1}, y*))
# ========================================================================
def modified_euler(f, t0, y0, t_end, h):
    n = int(round((t_end - t0) / h))
    t = t0 + h * np.arange(n + 1)
    y = np.zeros((n + 1,) + np.shape(y0))
    y[0] = y0
    for i in range(n):
        k1 = np.asarray(f(t[i], y[i]))
        y_pred = y[i] + h * k1
        k2 = np.asarray(f(t[i + 1], y_pred))
        y[i + 1] = y[i] + (h / 2) * (k1 + k2)
    return t, y

# ========================================================================
# 3) RK2  (midpoint method, 2nd order)
#       k1 = f(t_n, y_n)
#       k2 = f(t_n + h/2, y_n + h/2*k1)
#       y_{n+1} = y_n + h*k2
# ========================================================================
def rk2(f, t0, y0, t_end, h):
    n = int(round((t_end - t0) / h))
    t = t0 + h * np.arange(n + 1)
    y = np.zeros((n + 1,) + np.shape(y0))
    y[0] = y0
    for i in range(n):
        k1 = np.asarray(f(t[i], y[i]))
        k2 = np.asarray(f(t[i] + h / 2, y[i] + h / 2 * k1))
        y[i + 1] = y[i] + h * k2
    return t, y

# ========================================================================
# 4) RK4  (classic 4th order Runge-Kutta)
#       k1 = f(t_n, y_n)
#       k2 = f(t_n+h/2, y_n+h/2*k1)
#       k3 = f(t_n+h/2, y_n+h/2*k2)
#       k4 = f(t_n+h,   y_n+h*k3)
#       y_{n+1} = y_n + h/6*(k1+2k2+2k3+k4)
# ========================================================================
def rk4(f, t0, y0, t_end, h):
    n = int(round((t_end - t0) / h))
    t = t0 + h * np.arange(n + 1)
    y = np.zeros((n + 1,) + np.shape(y0))
    y[0] = y0
    for i in range(n):
        k1 = np.asarray(f(t[i], y[i]))
        k2 = np.asarray(f(t[i] + h / 2, y[i] + h / 2 * k1))
        k3 = np.asarray(f(t[i] + h / 2, y[i] + h / 2 * k2))
        k4 = np.asarray(f(t[i] + h, y[i] + h * k3))
        y[i + 1] = y[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y

# ========================================================================
# 5) RKF45  (Runge-Kutta-Fehlberg, embedded 4th/5th order, adaptive step)
#    Uses the classic Fehlberg coefficients; step size is grown/shrunk to
#    keep the local error estimate |y5-y4| within `tol`.
# ========================================================================
def rkf45(f, t0, y0, t_end, tol=1e-6, h_init=0.1, h_min=1e-6, h_max=1.0):
    t, y = t0, np.asarray(y0, dtype=float)
    h = h_init
    t_vals, y_vals = [t], [y.copy()]

    while t < t_end - 1e-12:
        h = min(h, t_end - t)

        k1 = h * np.asarray(f(t, y))
        k2 = h * np.asarray(f(t + h / 4, y + k1 / 4))
        k3 = h * np.asarray(f(t + 3 * h / 8, y + 3 * k1 / 32 + 9 * k2 / 32))
        k4 = h * np.asarray(f(t + 12 * h / 13,
                               y + 1932 * k1 / 2197 - 7200 * k2 / 2197 + 7296 * k3 / 2197))
        k5 = h * np.asarray(f(t + h,
                               y + 439 * k1 / 216 - 8 * k2 + 3680 * k3 / 513 - 845 * k4 / 4104))
        k6 = h * np.asarray(f(t + h / 2,
                               y - 8 * k1 / 27 + 2 * k2 - 3544 * k3 / 2565
                               + 1859 * k4 / 4104 - 11 * k5 / 40))

        y4 = y + 25 * k1 / 216 + 1408 * k3 / 2565 + 2197 * k4 / 4104 - k5 / 5
        y5 = y + 16 * k1 / 135 + 6656 * k3 / 12825 + 28561 * k4 / 56430 - 9 * k5 / 50 + 2 * k6 / 55

        err = np.max(np.abs(y5 - y4))
        err = max(err, 1e-14)  # avoid divide-by-zero

        if err <= tol or h <= h_min:
            t, y = t + h, y5          # accept the higher-order estimate
            t_vals.append(t)
            y_vals.append(y.copy())

        # standard step-size controller
        factor = 0.9 * (tol / err) ** 0.2
        factor = min(max(factor, 0.2), 5.0)
        h = min(max(h * factor, h_min), h_max)

    return np.array(t_vals), np.array(y_vals)

# ========================================================================
# 6) DOP853  (Dormand-Prince 8th order, adaptive)
#    A hand-coded Butcher tableau for an order-8(5,3) method needs ~13
#    stages with dozens of high-precision coefficients -- writing that out
#    by hand is impractical and error-prone. We call SciPy's `solve_ivp`
#    with method='DOP853', which is the same well-tested implementation
#    used throughout the scientific Python ecosystem.
# ========================================================================
def dop853(f, t0, y0, t_end, rtol=1e-10, atol=1e-12, max_step=np.inf):
    sol = solve_ivp(f, [t0, t_end], np.atleast_1d(y0), method='DOP853',
                     rtol=rtol, atol=atol, max_step=max_step, dense_output=False)
    y = sol.y.T
    if np.shape(y0) == ():          # scalar problem -> return a flat array
        y = y[:, 0]
    return sol.t, y

# ========================================================================
# Run all six methods and compare
# ========================================================================
h = 0.2  # fixed step size for the four fixed-step methods

methods = {
    'Euler':          lambda: euler(f, t0, y0, t_end, h),
    'Modified Euler':  lambda: modified_euler(f, t0, y0, t_end, h),
    'RK2':            lambda: rk2(f, t0, y0, t_end, h),
    'RK4':            lambda: rk4(f, t0, y0, t_end, h),
    'RKF45':          lambda: rkf45(f, t0, y0, t_end, tol=1e-6),
    'DOP853':         lambda: dop853(f, t0, y0, t_end),
}

results = {name: fn() for name, fn in methods.items()}

# ----------------------------------------------------------------------
# Table: final value, error at t_end, number of steps used
# ----------------------------------------------------------------------
print(f"{'Method':<16}{'y(t_end) approx':>18}{'Error':>14}{'Steps used':>14}")
for name, (t, y) in results.items():
    y_end = y[-1] if np.ndim(y) == 1 else y[-1].item()
    err = abs(y_end - y_exact(t_end))
    print(f"{name:<16}{y_end:>18.8f}{err:>14.2e}{len(t) - 1:>14d}")

# ----------------------------------------------------------------------
# Plots: solution curves, and error vs exact on a log scale
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

t_fine = np.linspace(t0, t_end, 400)
ax1.plot(t_fine, y_exact(t_fine), 'k--', lw=1.5, label='Exact')
for name, (t, y) in results.items():
    ax1.plot(t, y, marker='o', ms=3, lw=1.2, label=name)
ax1.set_xlabel('t')
ax1.set_ylabel('y')
ax1.set_title("Solution: y' = y - t² + 1,  y(0) = 0.5")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

for name, (t, y) in results.items():
    err = np.abs(np.ravel(y) - y_exact(t))
    err = np.clip(err, 1e-16, None)
    ax2.semilogy(t, err, marker='o', ms=3, lw=1.2, label=name)
ax2.set_xlabel('t')
ax2.set_ylabel('|error| (log scale)')
ax2.set_title('Error vs exact solution')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.show()
