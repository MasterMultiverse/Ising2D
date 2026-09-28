import random
import math
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# PARAMETERS
# -------------------------
size = 50
T = 2.269
sweeps = 100

# -------------------------
# CREATE LATTICE
# -------------------------
s = np.zeros((size, size), dtype=int)


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


# -------------------------
# INITIALIZE SYSTEM
# -------------------------
initialize()

# NEW
magnetizations = []

# -------------------------
# SETUP GRAPHICS
# -------------------------
plt.ion()

fig, ax = plt.subplots(figsize=(7, 7))

img = ax.imshow(
    s,
    cmap="gray",
    vmin=-1,
    vmax=1,
    interpolation="nearest"
)

title = ax.set_title("Initializing...")

plt.colorbar(img)

# -------------------------
# MAIN MONTE CARLO LOOP
# -------------------------
for sweep in range(sweeps):

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

    # NEW
    magnetizations.append(M)

    img.set_data(s)

    title.set_text(
        f"T={T:.2f}   Sweep={sweep+1}/{sweeps}\n"
        f"Magnetization={M:.4f}   Energy={E:.0f}"
    )

    plt.pause(0.01)

    print(
        f"Sweep {sweep+1:3d} | "
        f"Magnetization = {M:7.4f} | "
        f"Energy = {E:8.0f}"
    )

# -------------------------
# FINAL RESULTS
# -------------------------
print("\nSimulation Complete")
print(f"Final Magnetization = {magnetization():.4f}")
print(f"Final Energy = {total_energy():.0f}")

# NEW - second window
plt.figure(figsize=(8, 4))
plt.plot(magnetizations)
plt.xlabel("Sweep")
plt.ylabel("Magnetization")
plt.title("Magnetization vs Monte Carlo Sweep")
plt.grid(True)

plt.ioff()
plt.show()

input("Press Enter to close...")