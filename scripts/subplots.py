import matplotlib.pyplot as plt
import numpy as np



# subplots
# fig = Figure object, die gesamte Zeichenfläche
# Axes = jeweils ein einzelner Plotbereich "Koordinatensystem"
# Artist = alles, was gezeichnet wird



x = np.linspace(0, 10, 200)
"""
fig, axes = plt.subplots(2, 2)
# Anordnung: nrows, ncols

axes[0,0].plot(x, np.sin(x))
axes[0,1].plot(x, np.cos(x))
axes[1,0].plot(x, np.tan(x))
axes[1,1].plot(x, np.exp(x))

#plt.show()

# Größe kontrollieren: figsize=(8,6)
# Einheit: Inch
fig, axes = plt.subplots(2,2, figsize=(9,6))

# Abstände werden mit constrained_layout automatisch angepasst
fig, axes = plt.subplots(2,2, constrained_layout=True)

# x und y Achsen können für alle subplots gleich skaliert werden, wichtig für Vergleichsplots
fig, axes = plt.subplots(2,2, sharex=True, sharey=True)



axes[0,0].plot(x, np.sin(x))
axes[0,1].plot(x, np.cos(x))
axes[1,0].plot(x, np.tan(x))
axes[1,1].plot(x, np.exp(x))

plt.tight_layout()
"""


fig = plt.figure(figsize=(8,6))
# GridSpec um größe der Subplots anzupassen
gs = fig.add_gridspec(2, 2)
# height_ratios um die Gewichtung zu ändern, zB obere Reihe doppelt so hoch
#gs = fig.add_gridspec(2, 2, height_ratios=[2,1])

ax1 = fig.add_subplot(gs[0, :])   # obere Reihe, volle Breite
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

plt.show()