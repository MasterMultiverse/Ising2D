import random
import math
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# PARAMETERS
# =====================================

size = 50
sweeps = 500

Tc = 2.2692

# amplituda oscilacije temperature
A = 0.8

# amplituda slucajnog suma
noise_amp = 0.2

# period oscilacije
period = 50

# =====================================
# CREATE LATTICE
# =====================================

s = np.zeros((size, size), dtype=int)

# =====================================
# FUNCTIONS
# =====================================

def initialize():

    for i in range(size):
        for j in range(size):
            s[i, j] = 1 if random.random() < 0.5 else -1


def deltaU(i, j):

    top = s[(i - 1) % size, j]
    bottom = s[(i + 1) % size, j]
    left = s[i, (j - 1) % size]
    right = s[i, (j + 1) % size]

    return 2 * s[i, j] * (top + bottom + left + right)


def magnetization():

    return np.sum(s) / (size * size)


def total_energy():

    E = 0

    for i in range(size):
        for j in range(size):

            right = s[i, (j + 1) % size]
            down = s[(i + 1) % size, j]

            E -= s[i, j] * (right + down)

    return E


def temperature(sweep):

    sinus_part = A * math.sin(
        2 * math.pi * sweep / period
    )

    noise = random.uniform(
        -noise_amp,
        noise_amp
    )

    return Tc + sinus_part + noise


# =====================================
# INITIALIZATION
# =====================================

initialize()

magnetizations = []
temperatures = []
energies = []

# =====================================
# REAL-TIME LATTICE DISPLAY
# =====================================

plt.ion()

fig, ax = plt.subplots(figsize=(7, 7))

img = ax.imshow(
    s,
    cmap="gray",
    vmin=-1,
    vmax=1,
    interpolation="nearest"
)

plt.colorbar(img)

title = ax.set_title("Initializing...")

# =====================================
# MONTE CARLO LOOP
# =====================================

for sweep in range(sweeps):

    T = temperature(sweep)

    temperatures.append(T)

    for _ in range(size * size):

        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)

        Ediff = deltaU(i, j)

        if Ediff <= 0:

            s[i, j] *= -1

        elif random.random() < math.exp(-Ediff / T):

            s[i, j] *= -1

    M = magnetization()
    E = total_energy()

    magnetizations.append(abs(M))
    energies.append(E)

    img.set_data(s)

    title.set_text(
        f"T = {T:.4f}   Tc = {Tc:.4f}\n"
        f"Sweep {sweep+1}/{sweeps}\n"
        f"|M| = {abs(M):.4f}   E = {E:.0f}"
    )

    plt.pause(0.01)

    print(
        f"Sweep {sweep+1:4d} | "
        f"T = {T:7.4f} | "
        f"|M| = {abs(M):7.4f} | "
        f"E = {E:8.0f}"
    )

# =====================================
# FINAL OUTPUT
# =====================================

print("\n================================")
print("Simulation Complete")
print("================================")

print(
    f"Final Magnetization = "
    f"{magnetization():.4f}"
)

print(
    f"Final Energy = "
    f"{total_energy():.0f}"
)

# =====================================
# SORT FOR M(T) CURVE
# =====================================

pairs = sorted(
    zip(temperatures, magnetizations)
)

T_sorted = [p[0] for p in pairs]
M_sorted = [p[1] for p in pairs]

# =====================================
# ANALYSIS GRAPHS
# =====================================

fig2, (ax1, ax2, ax3, ax4) = plt.subplots(
    4,
    1,
    figsize=(10, 14)
)

# -------------------------------------
# Magnetization vs Sweep
# -------------------------------------

ax1.plot(
    magnetizations,
    color="blue",
    linewidth=1.5
)

ax1.set_title(
    "Magnetization vs Monte Carlo Sweep"
)

ax1.set_ylabel("|M|")

ax1.grid(True)

# -------------------------------------
# Temperature vs Sweep
# -------------------------------------

ax2.plot(
    temperatures,
    color="red",
    linewidth=1.5
)

ax2.axhline(
    Tc,
    color="black",
    linestyle="--",
    linewidth=2,
    label=f"Tc={Tc}"
)

ax2.set_title(
    "Temperature vs Sweep"
)

ax2.set_ylabel("T")

ax2.legend()

ax2.grid(True)

# -------------------------------------
# Energy vs Sweep
# -------------------------------------

ax3.plot(
    energies,
    color="green",
    linewidth=1.5
)

ax3.set_title(
    "Energy vs Sweep"
)

ax3.set_ylabel("E")

ax3.grid(True)

# -------------------------------------
# Magnetization vs Temperature
# -------------------------------------

ax4.scatter(
    temperatures,
    magnetizations,
    color="purple",
    alpha=0.4,
    s=12
)

ax4.plot(
    T_sorted,
    M_sorted,
    color="black",
    linewidth=2
)

ax4.axvline(
    Tc,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Tc={Tc}"
)

ax4.set_title(
    "Phase Transition: |M|(T)"
)

ax4.set_xlabel(
    "Temperature"
)

ax4.set_ylabel(
    "|M|"
)

ax4.legend()

ax4.grid(True)

plt.tight_layout()

# =====================================
# SHOW RESULTS
# =====================================

plt.ioff()
plt.show()

input("\nPress Enter to close...")