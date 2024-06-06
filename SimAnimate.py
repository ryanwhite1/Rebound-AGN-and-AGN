# # -*- coding: utf-8 -*-
# """
# Created on Fri Feb  2 12:09:32 2024

# @author: ryanw
# """
# import matplotlib
# matplotlib.use('Agg')
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import animation
# import time

# # plt.ioff()
    
# folder = 'OUTPUT_M8-f0.50-a0.010'
# pos_file = f'{folder}/positions.txt'

# positions_data = np.genfromtxt(pos_file)
# positions_data[:, 2] *= 1e8
# hash1 = positions_data[1, 1]
# hash2 = positions_data[0, 1]
# m1, m2 = positions_data[1, 2], positions_data[0, 2]
# # hash1 = positions_data[6, 1]
# # hash2 = positions_data[7, 1]

# particle1_txyz = positions_data[:, [0, 3, 4, 5]][np.where(positions_data[:, 1] == hash1)]
# particle2_txyz = positions_data[:, [0, 3, 4, 5]][np.where(positions_data[:, 1] == hash2)]

# tmin = 35000
# tmax = 37000
# # tmin = 0
# # tmax = 2000
# particle1_txyz = particle1_txyz[np.where((particle1_txyz[:, 0] <= tmax) & (particle1_txyz[:, 0] >= tmin))]
# particle2_txyz = particle2_txyz[np.where((particle2_txyz[:, 0] <= tmax) & (particle2_txyz[:, 0] >= tmin))]

# if particle1_txyz.shape != particle2_txyz.shape:
#     raise RuntimeError("Particle position arrays are different shapes!")

# # particle2_txyz -= particle1_txyz
# # SMBH_txyz = - particle1_txyz
# # particle1_txyz = np.zeros(particle1_txyz.shape)

# # # particle2_txyz -= particle1_txyz
# # SMBH_txyz = np.zeros(particle1_txyz.shape)
# # # particle1_txyz = 

# # particle1_txyz -= particle1_txyz
# # particle2_txyz -= particle1_txyz
# SMBH_txyz = np.zeros(particle1_txyz.shape)

# mean_pos = (m1 * particle1_txyz + m2 * particle2_txyz) / (m1 + m2)
# positions = np.zeros((len(particle1_txyz[:, 0]), 4, 2))
# positions[:, :, 0] = particle1_txyz
# positions[:, :, 1] = particle2_txyz
# # positions[:, :, 2] = SMBH_txyz - particle1_txyz

# scales = [50, 30]
# # scales = [50, 30, 100]

# fig = plt.figure(figsize=(12, 12), frameon=False)   # we want no frame so that it's a clean black animation
# ax = fig.add_subplot(projection='3d')   # 3d axes
# fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None) # removes (most of the) blank border around the plot
# # now do an initial scatter so we can get the axis limits to use through the animation
# particles = ax.scatter(positions[0, 1, :], positions[0, 2, :], positions[0, 3, :], s=scales, marker='.', c='k')
# xmin, xmax = min(positions[:, 1, :].flatten()), max(positions[:, 1, :].flatten())
# ymin, ymax = min(positions[:, 2, :].flatten()), max(positions[:, 2, :].flatten())
# zmin, zmax = min(positions[:, 3, :].flatten()), max(positions[:, 3, :].flatten())
# limmin, limmax = min([xmin, ymin, zmin]), max([xmax, ymax, zmax])   # get the minimum and maximums for the axis limits

# every = 1
# length = 500
# # now calculate some parameters for the animation frames and timing
# nt = len(positions[:, 0, 0]) # number of timesteps
# frames = np.arange(0, nt, every)    # iterable for the animation function. Chooses which frames (indices) to animate.
# fps = len(frames) // length  # fps for the final animation

# # ax.set_facecolor('k')   # black background, since space is blach duh

# theta = np.linspace(0, 2*np.pi, 360)
# r1 = np.sqrt(sum(particle1_txyz[0, 1:]**2))
# r2 = np.sqrt(sum(particle2_txyz[0, 1:]**2))
# Zzero = np.zeros(len(theta))
# circle1 = ax.plot(r1 * np.cos(theta), r1 * np.sin(theta), Zzero, lw=1, c='tab:blue')[0]
# circle2 = ax.plot(r1 * np.cos(theta), r1 * np.sin(theta), Zzero, lw=1, c='tab:red')[0]



# # def animate(i):
# #     if (i // every)%20 == 0:
# #         print(f"{i // every} / {len(frames)}")
        
# #     ax.clear()
# #     ax.scatter(positions[i, 1, :], positions[i, 2, :], positions[i, 3, :], s=scales, marker='.', c='w')
# #     ax.set_xlim(limmin, limmax); ax.set_ylim(limmin, limmax); ax.set_zlim(limmin, limmax)
# #     # if times[0]:    # plot the current time in the corner if we want to!
# #     #     ax.text(0.7 * limmax, 0.9 * limmax, 0, "$T = " + str(round(times[1][i], 2)) + "$", fontsize=24, color='w')
    
# #     ax.grid(False)
# #     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# #     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# #     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# #     ax.elev = 90    # sets the viewing angle to be top-down
# #     return fig,


# ax.grid(False)
# ax.set_axis_off()
# ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.elev = 90    # sets the viewing angle to be top-down
# ax.set_xlim(limmin, limmax); ax.set_ylim(limmin, limmax); ax.set_zlim(limmin, limmax)

# def animate(i):
#     if (i // every)%20 == 0:
#         print(f"{i // every} / {len(frames)}")
#     particles._offsets3d = (positions[i, 1, :], positions[i, 2, :], positions[i, 3, :])
#     r1 = np.sqrt(sum(particle1_txyz[i, 1:]**2)); r2 = np.sqrt(sum(particle2_txyz[i, 1:]**2))
#     circle1.set_data_3d(r1 * np.cos(theta), r1 * np.sin(theta), Zzero)
#     circle2.set_data_3d(r2 * np.cos(theta), r2 * np.sin(theta), Zzero)
#     return (particles, circle1, circle2)

# ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True, repeat=False)
# ani.save(f"{folder}/animation.mp4", writer='ffmpeg', fps=fps)
# plt.close('all')














# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:09:32 2024

@author: ryanw
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time

# plt.ioff()
    
folder = 'OUTPUT_M8-f0.50-a0.010'
pos_file = f'{folder}/positions.txt'

positions_data = np.genfromtxt(pos_file)
positions_data[:, 2] *= 1e8
hash1 = positions_data[1, 1]
hash2 = positions_data[0, 1]
m1, m2 = positions_data[1, 2], positions_data[0, 2]
# hash1 = positions_data[6, 1]
# hash2 = positions_data[7, 1]

particle1_txyz = positions_data[:, [0, 3, 4, 5]][np.where(positions_data[:, 1] == hash1)]
particle2_txyz = positions_data[:, [0, 3, 4, 5]][np.where(positions_data[:, 1] == hash2)]

tmin = 35000
tmax = 39000
# tmin = 0
# tmax = 2000
particle1_txyz = particle1_txyz[np.where((particle1_txyz[:, 0] <= tmax) & (particle1_txyz[:, 0] >= tmin))]
particle2_txyz = particle2_txyz[np.where((particle2_txyz[:, 0] <= tmax) & (particle2_txyz[:, 0] >= tmin))]

if particle1_txyz.shape != particle2_txyz.shape:
    raise RuntimeError("Particle position arrays are different shapes!")


particle1_spherical, particle2_spherical = np.zeros(particle1_txyz.shape), np.zeros(particle1_txyz.shape)

particle1_spherical[:, 0] = particle1_txyz[:, 0]
particle1_spherical[:, 1] = np.sqrt(sum([particle1_txyz[:, i]**2 for i in [1, 2, 3]]))
particle1_spherical[:, 2] = np.sign(particle1_txyz[:, 2]) * np.arccos(particle1_txyz[:, 1] / np.sqrt(sum([particle1_txyz[:, i]**2 for i in [1, 2]])))
particle2_spherical[:, 0] = particle2_txyz[:, 0]
particle2_spherical[:, 1] = np.sqrt(sum([particle2_txyz[:, i]**2 for i in [1, 2, 3]]))
particle2_spherical[:, 2] = np.sign(particle2_txyz[:, 2]) * np.arccos(particle2_txyz[:, 1] / np.sqrt(sum([particle2_txyz[:, i]**2 for i in [1, 2]])))

# particle2_spherical[:, 2] -= particle1_spherical[:, 2]
# particle1_spherical[:, 2] = 0

from astropy.stats import circmean
angle_arr = np.array([particle1_spherical[:, 2], particle2_spherical[:, 2]])
weights = np.zeros(angle_arr.shape)
weights[0, :] = m1 / (m1 + m2); weights[1, :] = m2 / (m1 + m2)
# weighted_angular_shift = (m1 * particle1_spherical[:, 2] + m2 * particle2_spherical[:, 2]) / (2 * (m1 + m2))
weighted_angular_shift = circmean(angle_arr, axis=0, weights=weights)
particle2_spherical[:, 2] -= weighted_angular_shift
particle1_spherical[:, 2] -= weighted_angular_shift

particle1_txyz[:, 1] = particle1_spherical[:, 1] * np.cos(particle1_spherical[:, 2])
particle1_txyz[:, 2] = particle1_spherical[:, 1] * np.sin(particle1_spherical[:, 2])
particle2_txyz[:, 1] = particle2_spherical[:, 1] * np.cos(particle2_spherical[:, 2])
particle2_txyz[:, 2] = particle2_spherical[:, 1] * np.sin(particle2_spherical[:, 2])



SMBH_txyz = np.zeros(particle1_txyz.shape)

positions = np.zeros((len(particle1_txyz[:, 0]), 4, 3))
positions[:, :, 0] = particle1_txyz
positions[:, :, 1] = particle2_txyz
positions[:, :, 2] = SMBH_txyz

scales = [200 * m1/(m1 + m2), 200 * m2/(m1 + m2), 1000]
# scales = [50, 30, 100]

fig = plt.figure(figsize=(12, 12), frameon=False)   # we want no frame so that it's a clean black animation
ax = fig.add_subplot(projection='3d')   # 3d axes
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None) # removes (most of the) blank border around the plot
# now do an initial scatter so we can get the axis limits to use through the animation
particles = ax.scatter(positions[0, 1, :], positions[0, 2, :], positions[0, 3, :], s=scales, marker='.', c='k')
xmin, xmax = min(positions[:, 1, :].flatten()), max(positions[:, 1, :].flatten())
ymin, ymax = min(positions[:, 2, :].flatten()), max(positions[:, 2, :].flatten())
zmin, zmax = min(positions[:, 3, :].flatten()), max(positions[:, 3, :].flatten())
limmin, limmax = min([xmin, ymin, zmin]), max([xmax, ymax, zmax])   # get the minimum and maximums for the axis limits

every = 1
length = 30
# now calculate some parameters for the animation frames and timing
nt = len(positions[:, 0, 0]) # number of timesteps
frames = np.arange(0, nt, every)    # iterable for the animation function. Chooses which frames (indices) to animate.
fps = len(frames) // length  # fps for the final animation

# ax.set_facecolor('k')   # black background, since space is blach duh

theta = np.linspace(0, 2*np.pi, 360)
r1 = np.sqrt(sum(particle1_txyz[0, 1:]**2))
r2 = np.sqrt(sum(particle2_txyz[0, 1:]**2))
Zzero = np.zeros(len(theta))
circle1 = ax.plot(r1 * np.cos(theta), r1 * np.sin(theta), Zzero, lw=1, c='tab:blue')[0]
circle2 = ax.plot(r1 * np.cos(theta), r1 * np.sin(theta), Zzero, lw=1, c='tab:red')[0]



# def animate(i):
#     if (i // every)%20 == 0:
#         print(f"{i // every} / {len(frames)}")
        
#     ax.clear()
#     ax.scatter(positions[i, 1, :], positions[i, 2, :], positions[i, 3, :], s=scales, marker='.', c='w')
#     ax.set_xlim(limmin, limmax); ax.set_ylim(limmin, limmax); ax.set_zlim(limmin, limmax)
#     # if times[0]:    # plot the current time in the corner if we want to!
#     #     ax.text(0.7 * limmax, 0.9 * limmax, 0, "$T = " + str(round(times[1][i], 2)) + "$", fontsize=24, color='w')
    
#     ax.grid(False)
#     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.elev = 90    # sets the viewing angle to be top-down
#     return fig,


ax.grid(False)
ax.set_axis_off()
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.elev = 90    # sets the viewing angle to be top-down
ax.set_xlim(limmin, limmax); ax.set_ylim(limmin, limmax); ax.set_zlim(limmin, limmax)

def animate(i):
    if (i // every)%20 == 0:
        print(f"{i // every} / {len(frames)}")
    particles._offsets3d = (positions[i, 1, :], positions[i, 2, :], positions[i, 3, :])
    r1 = np.sqrt(sum(particle1_txyz[i, 1:]**2)); r2 = np.sqrt(sum(particle2_txyz[i, 1:]**2))
    circle1.set_data_3d(r1 * np.cos(theta), r1 * np.sin(theta), Zzero)
    circle2.set_data_3d(r2 * np.cos(theta), r2 * np.sin(theta), Zzero)
    return (particles, circle1, circle2)

ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True, repeat=False)
ani.save(f"{folder}/animation.mp4", writer='ffmpeg', fps=fps)
plt.close('all')

