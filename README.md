
# ODE Solvers Comparison

A from-scratch implementation and comparison of six numerical methods for
solving ordinary differential equations:

- **Euler method** — explicit forward Euler (1st order)
- **Modified Euler method** — Heun's predictor-corrector (2nd order)
- **RK2** — midpoint method (2nd order)
- **RK4** — classic 4-stage Runge-Kutta (4th order)
- **RKF45** — Runge-Kutta-Fehlberg, embedded 4th/5th order with adaptive
  step size
- **DOP853** — Dormand-Prince 8th order, adaptive (via
  `scipy.integrate.solve_ivp`, since its ~13-stage Butcher tableau isn't
  practical to hand-code)

All six solvers are run on the same test problem, and their accuracy is
compared against the known closed-form solution.

## Features

- Each method implemented as its own function with a shared signature
  (`solver(f, t0, y0, t_end, h)`), so any of them can be dropped into a
  different ODE or system of ODEs
- Works for both scalar and vector-valued `y`, since state is handled as a
  NumPy array internally
- Adaptive step-size control for RKF45, implemented from the classic
  Fehlberg coefficients (Butcher tableau)
- Side-by-side accuracy comparison: a results table (final value, error,
  steps used) plus solution and error-vs-time plots

## Requirements

- Python 3.9+
- See `requirements.txt`

## Setup

```
pip install -r requirements.txt
```

## Usage

```
python ode_solvers.py
```

This prints a comparison table to the console and opens a plot with two
panels: the six numerical solutions against the exact solution, and each
method's error over time on a log scale.

## Test problem

$$
y' = y - t^{2} + 1, \quad y(0) = 0.5, \quad 0 \le t \le 2
$$

$$
\text{exact: } \; y(t) = (t + 1)^{2} - 0.5\,e^{t}
$$

This is a standard textbook example (Burden & Faires) chosen because it has
a closed-form solution, making it easy to verify that each method converges
at its expected order.

## Methods

**Euler** (1st order):

$$
y_{n+1} = y_{n} + h\,f(t_{n}, y_{n})
$$

**Modified Euler** — Heun's predictor-corrector (2nd order):

$$
y^{*} = y_{n} + h\,f(t_{n}, y_{n})
$$

$$
y_{n+1} = y_{n} + \frac{h}{2}\Big(f(t_{n}, y_{n}) + f(t_{n+1}, y^{*})\Big)
$$

**RK2** — midpoint method (2nd order):

$$
k_{1} = f(t_{n}, y_{n})
$$

$$
k_{2} = f\!\left(t_{n} + \frac{h}{2},\, y_{n} + \frac{h}{2}k_{1}\right)
$$

$$
y_{n+1} = y_{n} + h\,k_{2}
$$

**RK4** — classic Runge-Kutta (4th order):

$$
k_{1} = f(t_{n}, y_{n})
$$

$$
k_{2} = f\!\left(t_{n} + \frac{h}{2},\, y_{n} + \frac{h}{2}k_{1}\right)
$$

$$
k_{3} = f\!\left(t_{n} + \frac{h}{2},\, y_{n} + \frac{h}{2}k_{2}\right)
$$

$$
k_{4} = f(t_{n} + h,\, y_{n} + h\,k_{3})
$$

$$
y_{n+1} = y_{n} + \frac{h}{6}\Big(k_{1} + 2k_{2} + 2k_{3} + k_{4}\Big)
$$

**RKF45** — Runge-Kutta-Fehlberg (adaptive, embedded 4th/5th order):

Uses the classic Fehlberg coefficients to compute a 4th-order estimate
$y_{n+1}^{(4)}$ and a 5th-order estimate $y_{n+1}^{(5)}$ at each step; the
difference $\big|y_{n+1}^{(5)} - y_{n+1}^{(4)}\big|$ drives a step-size
controller that keeps the local error within a chosen tolerance.

**DOP853** — Dormand-Prince (adaptive, 8th order):

Called through `scipy.integrate.solve_ivp(method='DOP853')`, the same
well-tested implementation used throughout the scientific Python ecosystem.

## Results

Error at $t = 2$ decreases in exactly the order expected as method order
increases (Euler → Modified Euler / RK2 → RK4 → RKF45 → DOP853), with DOP853
landing at essentially machine precision.

## License

MIT — see [LICENSE](LICENSE).
