# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 11:30:52 2024

@author: ryanw
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.interpolate as interp
import warnings
warnings.filterwarnings("ignore")
from pagn import Sirko 
import pagn.constants as ct

# set LaTeX font for our figures
# plt.rcParams.update({"text.usetex": True})
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['mathtext.fontset'] = 'cm'

G = 6.67e-11
G_pc = 4.3e-3
G_cgs = G * 1e3
M_odot = 1.98e30
M_odot_cgs = M_odot * 1e3
c = 299792458.
c_cgs = c * 100
mu = 0.62               # average molecular weight ?
m_H = 1.6735575e-24     # hydrogen mass, cgs units
m_cgs = mu * m_H
thomson_cross_sec = 6.65246e-29     # SI units
thomson_cgs = thomson_cross_sec * 1e4
stef_boltz = 5.67037e-5    # cgs units
k = 1.38065e-16        # cgs

def angvel(r, M):
    '''
    '''
    return np.sqrt(G_cgs * M * M_odot_cgs / r**3)

def disk_model(M, f_edd, alpha, b):
    '''
    '''

    n = 1e3
    
    sk = Sirko.SirkoAGN(Mbh=M*ct.MSun, le=f_edd, alpha=alpha, b=b)
    sk.solve_disk(n)
    
    radii = sk.R
    log_radii = sk.R / sk.Rs
    t_eff = sk.Teff4**(1/4)
    temps = sk.T 
    rho = sk.rho 
    H = sk.h 
    h = H / radii
    Sigma = 2 * rho * H
    kappa = sk.kappa 
    tau = sk.tauV
    cs = sk.cs 
    Q = sk.Q 
    cs2 = cs*cs
    pgas = (ct.Kb / ct.massU) * temps / cs2
    prad = tau * ct.sigmaSB / (2 * ct.c) * sk.Teff4 / (rho * cs2)
    beta = pgas / (prad + pgas)

    return [log_radii, t_eff, temps, tau, kappa, Sigma, cs, rho, h, Q, beta, prad, pgas]


def save_disk_model(disk_params, location='', name='', save_all=False):
    '''
    '''
    path = os.path.dirname(os.path.abspath(__file__)) + location
    if not os.path.isdir(path):
        os.mkdir(path)
    param_names = ['log_radii', 't_eff', 'temps', 'tau', 'kappa', 'Sigma', 'cs', 'rho', 'h', 'Q', 'beta', 'prad', 'pgas', 'pressure']
    filenames = [path + param + '_' + name + '.csv' for param in param_names]
    if save_all:
        indices = np.arange(0, len(param_names))
    else:   # only saves parameter arrays for those relevant to migration in the nbody code
        indices = [0, 2, 4, 5, 8, 13]
    for index in indices:
        if index == 0:
            np.savetxt(filenames[index], np.log10(disk_params[index]), delimiter=',')
        elif index == 13:
            np.savetxt(filenames[index], np.log10(10**disk_params[11] + 10**disk_params[12]), delimiter=',')
        else:
            np.savetxt(filenames[index], disk_params[index], delimiter=',')


def plot_disk_model(disk_params, axes=[], save=False, location=''):
    '''
    '''
    log_radii, t_eff, temps, tau, kappa, Sigma, cs, rho, h, Q, beta, prad, pgas = disk_params

    if len(axes) == 0:
        fig, axes = plt.subplots(nrows=6, sharex=True, figsize=(5, 10), gridspec_kw={'hspace': 0})

    axes[0].plot(log_radii, 10**temps)
    axes[1].plot(log_radii, 10**Sigma)
    axes[2].plot(log_radii, 10**h)
    axes[3].plot(log_radii, 10**kappa)
    axes[4].plot(log_radii, 10**tau)
    axes[5].plot(log_radii, 10**Q)

    for i, ax in enumerate(axes):
        ax.set(xscale='log', yscale='log')
        if i != 0:
            ax.xaxis.set_tick_params(which='both', reset=True)
    axes[0].set(ylabel='$T_{\mathrm{mid}}$ (K)')
    axes[1].set(ylabel='$\Sigma$ (g/cm$^2$)')
    axes[2].set(ylabel='$h$ ($H$/$r$)')
    axes[3].set(ylabel='$\kappa$ (cm$^2$/g)')
    axes[4].set(ylabel=r'$\tau$')
    axes[5].set(ylabel='Toomre, $Q$', xlabel='$r$/$R_s$')

    if save:
        path = os.path.dirname(os.path.abspath(__file__)) + location
        if not os.path.isdir(path):
            os.mkdir(path)
        fig.savefig(path + 'disk_model.png', dpi=400, bbox_inches='tight')

def plot_many_models():
    ''' Used to plot a range of disk models for display in the paper. Saves the images to the "Images" folder.
    '''
    # set LaTeX font for our figures
    plt.rcParams.update({"text.usetex": True})
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'
    fig, axes = plt.subplots(nrows=6, sharex=True, figsize=(5, 12), gridspec_kw={'hspace': 0})
    masses = [1e6, 1e7, 1e8, 1e9]
    fracs = [0.1, 0.5, 1]
    # alphas = [0.01, 0.1]
    alphas = [0.01]

    colours = ['tab:orange', 'tab:red', 'tab:purple', 'tab:blue']
    ls = ['-', '--', ':']
    lw = [1, 0.5]
    for i, M in enumerate(masses):
        for j, f_edd in enumerate(fracs):
            for k, alpha in enumerate(alphas):
                log_radii, t_eff, temps, tau, kappa, Sigma, cs, rho, h, Q, beta, prad, pgas = disk_model(M, f_edd, alpha, 0)

                axes[0].plot(log_radii, 10**temps, color=colours[i], ls=ls[j], lw=lw[k], rasterized=True)
                axes[1].plot(log_radii, 10**Sigma, color=colours[i], ls=ls[j], lw=lw[k], rasterized=True)
                axes[2].plot(log_radii, 10**h, color=colours[i], ls=ls[j], lw=lw[k], rasterized=True)
                axes[3].plot(log_radii, 10**kappa, color=colours[i], ls=ls[j], lw=lw[k], rasterized=True)
                axes[4].plot(log_radii, 10**tau, color=colours[i], ls=ls[j], lw=lw[k], rasterized=True)
                axes[5].plot(log_radii, 10**Q, color=colours[i], ls=ls[j], lw=lw[k], rasterized=True)

    axes[0].set(ylabel='$T_{\mathrm{mid}}$ (K)')
    axes[1].set(ylabel='$\Sigma$ (g/cm$^2$)')
    axes[2].set(ylabel='$h$ ($H$/$r$)')
    axes[3].set(ylabel='$\kappa$ (cm$^2$/g)')
    axes[4].set(ylabel=r'$\tau$')
    axes[5].set(ylabel='Toomre, $Q$', xlabel='$R/R_s$')
    for i, ax in enumerate(axes):
        ax.set(xscale='log', yscale='log')

    from matplotlib.lines import Line2D
    custom_lines1 = [Line2D([0], [0], color=colours[0]),
                     Line2D([0], [0], color=colours[1]),
                     Line2D([0], [0], color=colours[2]),
                     Line2D([0], [0], color=colours[3])]
    custom_lines2 = [Line2D([0], [0], color='k', ls=ls[0]),
                     Line2D([0], [0], color='k', ls=ls[1]),
                     Line2D([0], [0], color='k', ls=ls[2])]
    axes[0].legend(custom_lines1, ['$M=10^6 M_\odot$',
                   '$M=10^7 M_\odot$', '$M=10^8 M_\odot$', '$M=10^9 M_\odot$'])
    axes[-1].legend(custom_lines2, ['$f_{\mathrm{edd}} = 0.1$',
                    '$f_{\mathrm{edd}} = 0.5$', '$f_{\mathrm{edd}} = 1$'])

    fig.savefig("Images/SGDiskModels.png", dpi=400, bbox_inches='tight')
    fig.savefig("Images/SGDiskModels.pdf", dpi=400, bbox_inches='tight')
    
def calculate_torques(M, f_edd, visc):
    bh_mass = 10 * M_odot_cgs
    gamma_coeff = 5/3 

    rs = 2 * G_cgs * M * M_odot_cgs / c_cgs**2
    log_radii, t_eff, temps, tau, kappa, Sigma, cs, rho, h, Q, beta, prad, pgas = disk_model(M, f_edd, visc, 0)

    spl_sigma = interp.CubicSpline(np.log10(log_radii), Sigma, extrapolate=True)
    spl_temp = interp.CubicSpline(np.log10(log_radii), temps, extrapolate=True)
    spl_dens = interp.CubicSpline(np.log10(log_radii), rho, extrapolate=True)
    spl_h = interp.CubicSpline(np.log10(log_radii), h, extrapolate=True)
    spl_kappa = interp.CubicSpline(np.log10(log_radii), kappa, extrapolate=True)
    # spl_tau = interp.CubicSpline(np.log10(log_radii), tau, extrapolate=True)
    spl_P = interp.CubicSpline(np.log10(log_radii), np.log10(10**prad + 10**pgas), extrapolate=True) # total pressure
    spl_cs = interp.CubicSpline(np.log10(log_radii), cs, extrapolate=True)

    def alpha(r): return -spl_sigma.derivative()(np.log10(r))
    def beta(r): return -spl_temp.derivative()(np.log10(r))
    def P_deriv(r): return -spl_P.derivative()(np.log10(r))
    
    log_radii = np.logspace(1, 5, 1000)
    torques = np.zeros(len(log_radii))

    for ii, r in enumerate(log_radii):
        logr = np.log10(r)
        Gamma_0 = ((10/M) / 10**spl_h(logr))**2 * 10**spl_sigma(logr) * (r*rs)**4 * angvel(r*rs, M)**2
        
        ### Migration from pardekooper
        # c_v = 14304 / 1000
        # tau_eff = 3 * 10**spl_tau(logr) / 8 + np.sqrt(3)/4 + 0.25 / 10**spl_tau(logr)
        # Theta = (c_v * 10**spl_sigma(logr) * angvel(r*rs, M) * tau_eff) / (12. * np.pi * stef_boltz * 10**(3 * spl_temp(logr)));
        # Gamma_iso = -0.85 - alpha(r) - 0.9 * beta(r)
        # xi = beta(r) - (gamma_coeff - 1) * alpha(r)
        # Gamma_ad = (-0.85 - alpha(r) - 1.7 * beta(r) + 7.9 * xi / gamma_coeff) / gamma_coeff;
        # Gamma = Gamma_0 * (Gamma_ad * Theta*Theta + Gamma_iso) / ((Theta + 1)*(Theta + 1));
        
        ### Migration from Jimenez
        H = 10**spl_h(logr) * r*rs
        chi = 16. * gamma_coeff * (gamma_coeff - 1.) * stef_boltz * 10**(4 * spl_temp(logr)) / (3. * 10**(2 * spl_dens(logr)) * 10**spl_kappa(logr) * (angvel(r*rs, M) * H)**2)
        chi_chi_c = chi / (H**2 * angvel(r*rs, M))
        fx = (np.sqrt(chi_chi_c / 2.) + 1. / gamma_coeff) / (np.sqrt(chi_chi_c / 2.) + 1.);
        Gamma_lindblad = - (2.34 - 0.1 * alpha(r) + 1.5 * beta(r)) * fx;
        Gamma_simp_corot = (0.46 - 0.96 * alpha(r) + 1.8 * beta(r)) / gamma_coeff;
        Gamma = Gamma_0 * (Gamma_lindblad + Gamma_simp_corot);

        ### Thermal torques
        dPdr = P_deriv(r)
        x_c = dPdr * H**2 / (3 * gamma_coeff * r*rs)
        L = 4. * np.pi * G_cgs * bh_mass * m_H * c_cgs / thomson_cgs;     # accretion assuming completely ionized hydrogen
        # L = accretion * 4 * np.pi * G_cgs * bh_mass * c_cgs / 10**spl_kappa(logr)       # accretion assuming the AGN disk composition
        # below are equations 17-23 from gilbaum 2022
        R_BHL = 2 * G_cgs * bh_mass / (H * angvel(r*rs, M))**2
        R_H = r*rs * np.cbrt(10 / (3 * M))
        b_H = np.sqrt(R_BHL * R_H)
        mdot_RBHL = np.pi * min(R_BHL, b_H) * min(R_BHL, b_H, H) * (H * angvel(r*rs, M))
        L_RBHL = 0.1 * c_cgs**2 * mdot_RBHL
        L = min(L_RBHL, L)
        
        Lc = 4. * np.pi * G_cgs * bh_mass * 10**spl_dens(logr) * chi / gamma_coeff
        # print(L/Lc)
        lambda_ = np.sqrt(2. * chi / (3 * gamma_coeff * angvel(r*rs, M)));
        Gamma_thermal = 1.61 * (gamma_coeff - 1) / gamma_coeff * x_c / lambda_ * (L/Lc - 1.) * Gamma_0 / 10**spl_h(logr);

        ### GR Inspiral torque
        Gamma_GW = Gamma_0 * (-32 / 5 * (c_cgs / 10**spl_cs(logr))**3 * 10**(6 * spl_h(logr)) * (2*r)**-4 * M*M_odot_cgs / (10**spl_sigma(logr) * (r*rs)**2))
        
        Gamma += Gamma_thermal + Gamma_GW
        torques[ii] = Gamma
    return log_radii, torques

def plot_torques(M, f_edd, visc, disk_params, save=False, location=''):
    '''
    Open problems:
        1. Not sure whether to model based on total pressure or just gas pressure. Evgeni modelled by gas pressure, and this
            means that there are some migration traps in the inner disk; these migration traps disappear when modelling via total pressure
        2. Need to plot regions of parameter space that contain at least one migration trap (and at what radius!)
    '''
    

    fig, ax = plt.subplots()

    log_radii, torques = calculate_torques(M, f_edd, visc)
        
    pos_vals = torques > 0 
    neg_vals = torques <= 0 
    pos_torques = [torques[i] if pos_vals[i] else np.nan for i in range(len(torques))]
    neg_torques = [-torques[i] if neg_vals[i] else np.nan for i in range(len(torques))]
    ax.plot(log_radii, pos_torques, label='$+$ve', rasterized=True)
    ax.plot(log_radii, neg_torques, ls='--', label='$-$ve', rasterized=True)
    
    ax.set(xscale='log', yscale='log', xlabel="log$(R/R_s)$", ylabel='abs($\Gamma$)')
    ax.legend()
    ax.grid()
    if save:
        path = os.path.dirname(os.path.abspath(__file__)) + location
        if not os.path.isdir(path):
            os.mkdir(path)
        fig.savefig(path + 'Torque_Model.png', dpi=400, bbox_inches='tight')
        fig.savefig(path + 'Torque_Model.pdf', dpi=400, bbox_inches='tight')

def plot_many_torques():
    '''
    Open problems:
        1. Not sure whether to model based on total pressure or just gas pressure. Evgeni modelled by gas pressure, and this
            means that there are some migration traps in the inner disk; these migration traps disappear when modelling via total pressure
        2. Need to plot regions of parameter space that contain at least one migration trap (and at what radius!)
    '''
    plt.rcParams.update({"text.usetex": True})
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'

    fig, ax = plt.subplots()
    # fig2, ax2 = plt.subplots()

    masses = [1e6, 1e7, 1e8, 1e9]
    fracs = [0.1]
    # alphas = [0.01, 0.1]
    alphas = [0.01]
    colours = ['tab:orange', 'tab:red', 'tab:purple', 'tab:blue']

    for i, M in enumerate(masses):
        for j, f_edd in enumerate(fracs):
            for jj, visc in enumerate(alphas):
                log_radii, torques = calculate_torques(M, f_edd, visc)
                
                pos_vals = torques > 0 
                neg_vals = torques <= 0 
                pos_torques = [torques[i] if pos_vals[i] else np.nan for i in range(len(torques))]
                neg_torques = [-torques[i] if neg_vals[i] else np.nan for i in range(len(torques))]
                ax.plot(log_radii, pos_torques, c=colours[i], label=f'$M=10^{int(np.log10(M))}$', rasterized=True)
                ax.plot(log_radii, neg_torques, c=colours[i], ls='--', rasterized=True)
    ax.set(xscale='log', yscale='log', xlabel="log$(R/R_s)$", ylabel='abs($\Gamma$)')
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    p1 = Line2D([0], [0], color='k', ls='-', label='$+$ve'); handles.append(p1)
    p2 = Line2D([0], [0], color='k', ls='--', label='$-$ve'); handles.append(p2)
    ax.legend(handles=handles)
    ax.grid()
    fig.savefig('Images/Torque_Model.png', dpi=400, bbox_inches='tight')
    fig.savefig('Images/Torque_Model.pdf', dpi=400, bbox_inches='tight')
    
    # ax2.set(xscale='log')

def plot_migration_traps(visc):
    from matplotlib.colors import LogNorm
    import numpy.ma as ma
    plt.rcParams.update({"text.usetex": True})
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'
    fig, ax = plt.subplots()
    n_M = 20
    n_edd = 20
    bh_mass = 10 * M_odot_cgs
    gamma_coeff = 5/3 
    trap_rads = np.ones((n_M, n_edd))
    masses = np.logspace(6, 9, n_M)
    fractions = np.logspace(-2, 0, n_edd)
    for i, M in enumerate(masses):
        for j, f_edd in enumerate(fractions):
            log_radii, torques = calculate_torques(M, f_edd, visc)
                
            if torques[-1] < 0:
                for n_torque, torque in enumerate(torques[::-1]):
                    k = len(torques) - n_torque
                    if torque > 0:
                        trap_rads[i, j] = log_radii[k]
                        break
    x, y = np.meshgrid(masses, fractions)
    
    Zm = ma.masked_where(trap_rads == 1, trap_rads).T
    
    contour = ax.pcolormesh(x, y, Zm, cmap='viridis', 
                            norm=LogNorm(vmin=Zm.min(), vmax=Zm.max()),
                            rasterized=True)
    ax.set(xlabel='SMBH Mass', ylabel='Eddington Fraction', xscale='log', yscale='log')
    fig.colorbar(contour, label='Migration Trap Location ($R_s$)')
    fig.savefig(f'Images/MigrationTraps-alph{visc}.png', dpi=400, bbox_inches='tight')
    fig.savefig(f'Images/MigrationTraps-alph{visc}.pdf', dpi=400, bbox_inches='tight')
                
                    
            

plot_many_models()
# plot_many_torques()
# plot_migration_traps(0.01)
# plot_migration_traps(0.1)
