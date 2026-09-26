import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ----------------------------------------------------------------------
# Dark theme (same palette as the other projects)
# ----------------------------------------------------------------------
plt.style.use('dark_background')

BG_COLOR = '#1e1e1e'
PANEL_COLOR = '#2b2b2b'
TEXT_COLOR = '#e0e0e0'
GRID_COLOR = '#444444'
ACCENT1 = '#54a0ff'
ACCENT2 = '#ff9f43'
ACCENT3 = '#1dd1a1'

# ----------------------------------------------------------------------
# Change the function here to test for different functions
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
# Plots: solution curves, and error vs exact on a log scale
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor(BG_COLOR)

def style_axis(ax, title):
    ax.set_facecolor(BG_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.4)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.set_title(title)

line_colors = [ACCENT1, ACCENT2, ACCENT3, '#e15fed', '#feca57', '#c8d6e5']

t_fine = np.linspace(t0, t_end, 400)
ax1.plot(t_fine, y_exact(t_fine), '--', color=TEXT_COLOR, lw=1.5, label='Exact')
for (name, (t, y)), color in zip(results.items(), line_colors):
    ax1.plot(t, y, marker='o', ms=3, lw=1.2, color=color, label=name)
ax1.set_xlabel('t')
ax1.set_ylabel('y')
style_axis(ax1, "Solution: y' = y - t² + 1,  y(0) = 0.5")
leg1 = ax1.legend(fontsize=8, facecolor=PANEL_COLOR, edgecolor=GRID_COLOR)
for txt in leg1.get_texts():
    txt.set_color(TEXT_COLOR)

for (name, (t, y)), color in zip(results.items(), line_colors):
    err = np.abs(np.ravel(y) - y_exact(t))
    err = np.clip(err, 1e-16, None)
    ax2.semilogy(t, err, marker='o', ms=3, lw=1.2, color=color, label=name)
ax2.set_xlabel('t')
ax2.set_ylabel('|error| (log scale)')
style_axis(ax2, 'Error vs exact solution')
leg2 = ax2.legend(fontsize=8, facecolor=PANEL_COLOR, edgecolor=GRID_COLOR)
for txt in leg2.get_texts():
    txt.set_color(TEXT_COLOR)
ax2.grid(True, which='both', color=GRID_COLOR, alpha=0.4)

plt.tight_layout()

# ----------------------------------------------------------------------
# Center the OS window on screen (Tk / Qt / Wx backends)
# ----------------------------------------------------------------------
def center_window(fig):
    try:
        manager = fig.canvas.manager
        backend = plt.get_backend().lower()
        window = manager.window

        if 'tk' in backend:
            window.update_idletasks()
            width = window.winfo_reqwidth()
            height = window.winfo_reqheight()
            if width <= 1 or height <= 1:
                window.update()
                width = window.winfo_width()
                height = window.winfo_height()
            screen_w = window.winfo_screenwidth()
            screen_h = window.winfo_screenheight()
            x = max(0, (screen_w - width) // 2)
            y = max(0, (screen_h - height) // 2)
            window.geometry(f"{width}x{height}+{x}+{y}")

        elif 'qt' in backend:
            screen = window.screen() if hasattr(window, 'screen') else None
            if screen is None:
                from matplotlib.backends.qt_compat import QtWidgets
                screen = QtWidgets.QApplication.primaryScreen()
            screen_geo = screen.availableGeometry()
            frame_geo = window.frameGeometry()
            frame_geo.moveCenter(screen_geo.center())
            window.move(frame_geo.topLeft())

        elif 'wx' in backend:
            window.CentreOnScreen()

    except Exception:
        pass  # non-critical -- if this fails, the window just opens wherever it normally would

center_window(fig)

plt.show()