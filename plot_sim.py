import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
import sys

def time_convert(time, M, R):
    ''' Converts from n-body time to real time (in units of Myr).
    Parameters
    ----------
    time : float/array
        N-body times
    M : float
        Mass of the system in units of solar masses
    R : float
        Radius of the system in pc
    '''
    mass = M * 1.988e30
    radius = R * 3.086e16
    G = 6.6743e-11
    Myr_sec = 31536000000000.0
    return time * np.sqrt(radius**3 / (mass * G)) / Myr_sec

# use the below when running from a bash script
SMBHMass = int(sys.argv[1])
SMBHMass = 10.**SMBHMass
f_edd = float(sys.argv[2])
alpha = float(sys.argv[3])

# use the below when running in an IDE
# SMBHMass = 8
# SMBHMass = 10.**SMBHMass
# f_edd = 0.5
# alpha = 0.01

folder = f'OUTPUT_M{int(np.log10(SMBHMass))}-f{f_edd:.2f}-a{alpha:.3f}/'

r_s = 2 * 4.3e-3 * SMBHMass / 9e10  # 2GM / c^2     units of pc
Nr_s = 2e3      # number of schwarzschild radii to initialise the sim with respect to
lenscale = Nr_s * r_s
rawdata = np.genfromtxt(folder+'orbits.txt')
N = int(rawdata[:, 1].max())
fig, axes = plt.subplots(figsize=(10, 10), nrows=2, sharex=True, gridspec_kw={'hspace':0, 'height_ratios':[1.5, 1]})

cmap_decision = 'scicomap'
### now to decide on the colours for the semi-major axis plot
if cmap_decision == 'stitched':
    # below stitches two colour maps together 
    N1 = N // 2; N2 = N - N1
    colours = np.concatenate([plt.cm.viridis(np.linspace(0, 1, N1)), plt.cm.plasma(np.linspace(0, 1, N2))])
elif cmap_decision == 'single':
    # alternatively just use one colourmap
    colours = plt.cm.turbo(np.linspace(0, 1, N))
elif cmap_decision == 'scicomap':
    ## below uses a colourmap from an external package
    import scicomap as sc   # import SciCoMap
    # cmap = sc.ScicoCircular(cmap='phase')
    # cmap = sc.ScicoDiverging(cmap='guppy')
    # colours = cmap.get_mpl_color_map()(np.linspace(0, 1, N))
    ## pride but omit 7% either side of the end since it's too dark
    cmap = sc.ScicoDiverging(cmap='pride')
    colours = cmap.get_mpl_color_map()(np.linspace(0.07, 0.93, N))
np.random.shuffle(colours)

# rawdata[:, 5] *= SMBHMass
# norm = plt.Normalize(rawdata[:, 5].min(), rawdata[:, 5].max())
# cmap = 'viridis'
fig2, ax2 = plt.subplots()

for i in range(1, N+1):
    particle_data = rawdata[np.where(rawdata[:, 1] == i)[0], :]
    semi_majors = particle_data[:, 2] * Nr_s
    eccentricities = particle_data[:, 3]
    inclinations = particle_data[:, 4]  # convert from rad to degrees
    real_times = time_convert(particle_data[:, 0], SMBHMass, lenscale)
    line, = axes[0].plot(real_times, semi_majors, lw=0.5, color=colours[i-1])
    axes[1].plot(real_times, eccentricities, lw=0.5, color=colours[i-1])
    ax2.plot(real_times, inclinations, lw=0.5, color=colours[i-1])
    for j in [0, 1]:
        axes[j].axvline(real_times[-1], ls='--', lw=0.5, c='k')
    ax2.axvline(real_times[-1], ls='--', lw=0.5, c='k')
        
    
    ### now to overlay the mass changes
    masses = particle_data[:, 5] * SMBHMass
    axes[0].text(real_times[0], semi_majors[0], f"{masses[0]:.1f}", c=line.get_color(), fontsize=4, ha='right')
    unique_masses = np.unique(masses)
    if len(unique_masses) > 1:
        for j in range(1, len(unique_masses)):
            index = np.argwhere(masses == unique_masses[j]).flatten()[0]
            mass_text = f"{masses[index - 1]:.1f} -> {masses[index]:.1f}"
            axes[0].text(real_times[index], semi_majors[index], mass_text, c=line.get_color(), fontsize=4, ha='center')
    
    
axes[0].set(yscale='log', ylabel="Semi-Major Axis ($R_s$)")
axes[1].set(yscale='log', xlabel="Time (Myr)", ylabel='Eccentricity')
# fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes[0], label='Mass ($M_\odot$)', 
#              location='top', orientation='horizontal', aspect=50, pad=0)
fig.savefig(folder+'NBody_a-e_Plot.png', dpi=500, bbox_inches='tight')
ax2.set(xlabel="Time (Myr)", ylabel='Inclination (degrees)')
fig2.savefig(folder+'NBody_inclination_Plot.png', dpi=500, bbox_inches='tight')


### now to plot the binary mass and binary mass ratio plot
merger_data = np.genfromtxt(folder+'mergers.txt')
mergers = merger_data.shape[0]

binary_masses = np.array([merger_data[i, 3] + merger_data[i, 4] for i in range(mergers)])
binary_m_ratio = np.array([min(merger_data[i, 3], merger_data[i, 4]) / max(merger_data[i, 3], merger_data[i, 4]) for i in range(mergers)])
effective_spins = merger_data[:, 7]
remnant_spins = merger_data[:, 8]

fig, ax = plt.subplots()
n_bins = 10
xbins = np.logspace(np.log10(min(binary_masses)), np.log10(max(binary_masses)), n_bins)
ybins = np.logspace(np.log10(min(binary_m_ratio)), np.log10(max(binary_m_ratio)), n_bins)
_, _, _, cbar = ax.hist2d(binary_masses, binary_m_ratio, cmin=1, bins=[xbins, ybins])
ax.set(xscale='log', yscale='log', xlabel='Binary Mass, $m_1 + m_2$ ($M_\odot$)', ylabel='Mass Ratio $q$ ($m_1 / m_2$)')
fig.colorbar(cbar, label='Counts')
fig.savefig(folder+'Q_vs_BinaryMass.png', dpi=400, bbox_inches='tight')


fig, ax = plt.subplots()
ax.hist(remnant_spins, bins=20, ec='k')
ax.set(xlabel="Dimensionless Spin Parameter", ylabel='Frequency')
fig.savefig(folder+'BH_Spins.png', dpi=400, bbox_inches='tight')


def callister(x, y):
    # q-vs-chi_eff probability density function from callister (2021) - Who Ordered That? Unequal-mass Binary Black Hole Mergers Have Larger Effective Spins
    return y**1.08 * np.exp(-(x - (0.19 - 0.46*(y - 0.5)))**2 / (2 * (10**(-1.06 - 0.83*(y - 0.5)))**2))
fig, ax = plt.subplots()
x = np.linspace(-1, 1, 100)
y = np.linspace(0, 1, 100)
z = np.ndarray((len(x), len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        z[i, j] = callister(x[i], y[j])
ax.contourf(x, y, z.T, cmap='binary', alpha=0.3, levels=[0.01, 0.05, 0.16, 0.5, 0.84, 1], antialiased=True)
cbar = ax.scatter(effective_spins, binary_m_ratio, c=merger_data[:, 11])
ax.set(xlabel='$\chi_{eff}$', ylabel='Mass Ratio $q$ ($m_1 / m_2$)', xlim=[-1, 1], ylim=[0, 1])
# ax.legend()
fig.colorbar(cbar, label='BH Generation')
fig.savefig(folder+'BH_Effective_Spins.png', dpi=400, bbox_inches='tight')

plt.close('all')
