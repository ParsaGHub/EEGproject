import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

results_dir = Path(__file__).parent / 'results'

degrees = list(range(0, 91, 5))
theta_deg = np.arange(180)

fig, ax = plt.subplots(figsize=(12, 7), facecolor='#0a0a0a')
ax.set_facecolor('#0a0a0a')

cmap = cm.plasma
colors = [cmap(i / (len(degrees) - 1)) for i in range(len(degrees))]

for deg, color in zip(degrees, colors):
    d = np.load(results_dir / f'Analytical_deg{deg:03d}.npz')
    phi = d['phi_20'].reshape(180, 180)[:, 0] * 1e6
    lw = 1.4 if deg not in (0, 45, 90) else 2.2
    alpha = 0.6 if deg not in (0, 45, 90) else 1.0
    ax.plot(theta_deg, phi, color=color, linewidth=lw, alpha=alpha)

# highlight labels for 0, 45, 90
for deg, color in zip([0, 45, 90], [cmap(0.0), cmap(0.5), cmap(1.0)]):
    d = np.load(results_dir / f'Analytical_deg{deg:03d}.npz')
    phi = d['phi_20'].reshape(180, 180)[:, 0] * 1e6
    ax.plot(theta_deg, phi, color=color, linewidth=2.5, label=f'{deg}° ', zorder=5)

sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 90))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label('Dipole angle (°)', color='white', fontsize=12, labelpad=10)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

ax.axhline(0, color='#444444', linewidth=0.8, linestyle='--')
ax.set_xlim(0, 90)
ax.set_xlabel('Polar angle θ (°)', color='white', fontsize=13)
ax.set_ylabel('Scalp potential (µV)', color='white', fontsize=13)
ax.set_title('Scalp Potential vs θ — All Dipole Orientations', color='white', fontsize=15, pad=15)
ax.tick_params(colors='white', labelsize=11)
ax.legend(labels=['0° (radial)', '45°', '90° (tangential)'],
          facecolor='#1a1a1a', edgecolor='#444444', labelcolor='white', fontsize=10)
for spine in ax.spines.values():
    spine.set_edgecolor('#333333')

plt.tight_layout()
out = results_dir / 'all_angles_single.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved to {out}")
