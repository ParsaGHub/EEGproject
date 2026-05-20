import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

results_dir = Path(__file__).parent / 'results'

degrees = list(range(0, 91, 5))
theta, phi_angle = np.mgrid[0:180:1, -90:90:1]

ncols = 5
nrows = 4

fig = plt.figure(figsize=(22, 18), facecolor='#0d0d0d')
fig.suptitle('Scalp Potential — Analytical Four-Sphere Model  (skull conductivity ratio 1:20)',
             color='white', fontsize=18, y=0.99)

gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.45, wspace=0.3)

data = []
for deg in degrees:
    d = np.load(results_dir / f'Analytical_deg{deg:03d}.npz')
    phi = d['phi_20'].reshape(180, 180) * 1e6
    data.append(phi)

# use a robust percentile-based scale so mid-range variation is visible
all_vals = np.concatenate([p.ravel() for p in data])
vmax = np.percentile(np.abs(all_vals), 99)

im = None
for i, (deg, phi) in enumerate(zip(degrees, data)):
    row, col = divmod(i, ncols)
    ax = fig.add_subplot(gs[row, col])
    im = ax.contourf(phi_angle, theta, phi, levels=50, cmap='seismic',
                     vmin=-vmax, vmax=vmax)
    ax.invert_yaxis()  # θ=0 (top of head) at top
    ax.set_title(f'{deg}°', color='white', fontsize=13, pad=5, fontweight='bold')
    ax.set_xlabel('φ (°)', color='#bbbbbb', fontsize=8)
    ax.set_ylabel('θ (°)', color='#bbbbbb', fontsize=8)
    ax.tick_params(colors='#aaaaaa', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#555555')
    ax.set_facecolor('#111111')

# colorbar in the last empty slot
cbar_ax = fig.add_axes([0.83, 0.08, 0.018, 0.2])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('Potential (µV)', color='white', fontsize=11, labelpad=8)
cb.ax.yaxis.set_tick_params(color='#aaaaaa', labelsize=8)
plt.setp(cb.ax.yaxis.get_ticklabels(), color='#aaaaaa')

out = results_dir / 'all_angles_contour.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved to {out}")
