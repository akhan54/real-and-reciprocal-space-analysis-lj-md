"""
Publication-Quality Plotting for Structural Analysis
=====================================================

This module provides plotting functions for visualizing:
- Radial distribution functions g(r)
- Static structure factors S(q)
- Bond-orientational order parameters ψ6
- Temperature-dependent phase behavior
- Energy landscapes

All plots follow scientific publishing standards with proper labels,
units, and clear visual presentation.

Author: Visualization tools for materials science research
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# Set publication-quality defaults
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14


def plot_radial_distribution(r, g_r, title=None, filename=None, show=True):
    """
    Plot radial distribution function g(r)
    
    Parameters
    ----------
    r : array-like
        Radial distances
    g_r : array-like
        RDF values
    title : str or None, optional
        Plot title
    filename : str or None, optional
        If provided, save figure to this file
    show : bool, optional
        Whether to display the plot (default: True)
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    ax.plot(r, g_r, 'b-', linewidth=2, label='g(r)')
    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Ideal gas')
    
    ax.set_xlabel(r'Distance $r$ ($\sigma$)', fontsize=12)
    ax.set_ylabel(r'$g(r)$', fontsize=12)
    ax.set_title(title if title else 'Radial Distribution Function', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, ax


def plot_structure_factor(q, S_q, title=None, filename=None, show=True):
    """
    Plot static structure factor S(q)
    
    Parameters
    ----------
    q : array-like
        Wavevector magnitudes
    S_q : array-like
        Structure factor values
    title : str or None, optional
        Plot title
    filename : str or None, optional
        If provided, save figure to this file
    show : bool, optional
        Whether to display the plot (default: True)
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    ax.plot(q, S_q, 'r-', linewidth=2)
    
    ax.set_xlabel(r'Wavevector $q$ ($\sigma^{-1}$)', fontsize=12)
    ax.set_ylabel(r'$S(q)$', fontsize=12)
    ax.set_title(title if title else 'Static Structure Factor', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # Highlight first peak if present
    if len(S_q) > 10:
        peak_idx = np.argmax(S_q[q > 2.0]) + np.sum(q <= 2.0)
        if peak_idx < len(q):
            ax.axvline(q[peak_idx], color='orange', linestyle='--', alpha=0.7, 
                      label=f'Peak at q = {q[peak_idx]:.2f}')
            ax.legend()
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, ax


def plot_combined_structure(r, g_r, q, S_q, psi6=None, title=None, filename=None, show=True):
    """
    Create a combined plot showing g(r), S(q), and optionally ψ6
    
    Parameters
    ----------
    r, g_r : array-like
        Radial distribution data
    q, S_q : array-like
        Structure factor data
    psi6 : float or None, optional
        Bond-orientational order parameter value
    title : str or None, optional
        Overall figure title
    filename : str or None, optional
        If provided, save figure to this file
    show : bool, optional
        Whether to display the plot (default: True)
    """
    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)
    
    # g(r) plot
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(r, g_r, 'b-', linewidth=2)
    ax1.axhline(y=1.0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel(r'Distance $r$ ($\sigma$)', fontsize=12)
    ax1.set_ylabel(r'$g(r)$', fontsize=12)
    ax1.set_title('Radial Distribution Function', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # S(q) plot
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(q, S_q, 'r-', linewidth=2)
    ax2.set_xlabel(r'Wavevector $q$ ($\sigma^{-1}$)', fontsize=12)
    ax2.set_ylabel(r'$S(q)$', fontsize=12)
    ax2.set_title('Static Structure Factor', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add ψ6 annotation if provided
    if psi6 is not None:
        fig.text(0.5, 0.96, f'ψ₆ = {psi6:.4f}', 
                ha='center', va='top', fontsize=12, fontweight='bold')
    
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, (ax1, ax2)


def plot_temperature_sweep(sweep_results, filename=None, show=True):
    """
    Create comprehensive temperature sweep visualization
    
    Parameters
    ----------
    sweep_results : dict
        Results from TemperatureSweep.run_sweep()
    filename : str or None, optional
        If provided, save figure to this file
    show : bool, optional
        Whether to display the plot (default: True)
    """
    T = sweep_results['temperatures']
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. Bond-orientational order parameter ψ6 vs Temperature
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.errorbar(T, sweep_results['psi6'], yerr=sweep_results['psi6_std'],
                fmt='o-', color='blue', linewidth=2, markersize=6, capsize=3)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='ψ₆ = 0.5')
    ax1.set_xlabel(r'Temperature $T$ ($\epsilon/k_B$)', fontsize=12)
    ax1.set_ylabel(r'$\psi_6$', fontsize=12)
    ax1.set_title('Hexagonal Order Parameter', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Shade different phases
    if len(T) > 5:
        # Simple heuristic: solid if ψ6 > 0.5
        solid_region = sweep_results['psi6'] > 0.5
        if np.any(solid_region):
            ax1.axvspan(T[0], T[solid_region][-1] if np.any(solid_region) else T[0], 
                       alpha=0.1, color='blue', label='Solid-like')
    
    # 2. Potential Energy vs Temperature
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.errorbar(T, sweep_results['potential_energy'], 
                yerr=sweep_results['potential_energy_std'],
                fmt='s-', color='red', linewidth=2, markersize=6, capsize=3)
    ax2.set_xlabel(r'Temperature $T$ ($\epsilon/k_B$)', fontsize=12)
    ax2.set_ylabel(r'Potential Energy per Particle ($\epsilon$)', fontsize=12)
    ax2.set_title('Thermal Energy', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. g(r) First Peak Height vs Temperature
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(T, sweep_results['g_r_peak_height'], 'o-', 
            color='green', linewidth=2, markersize=6)
    ax3.set_xlabel(r'Temperature $T$ ($\epsilon/k_B$)', fontsize=12)
    ax3.set_ylabel(r'Peak Height of $g(r)$', fontsize=12)
    ax3.set_title('First Shell Correlation Strength', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. S(q) Peak Height vs Temperature
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(T, sweep_results['S_q_peak_height'], '^-', 
            color='purple', linewidth=2, markersize=6)
    ax4.set_xlabel(r'Temperature $T$ ($\epsilon/k_B$)', fontsize=12)
    ax4.set_ylabel(r'Peak Height of $S(q)$', fontsize=12)
    ax4.set_title('Structure Factor Peak Intensity', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle('Temperature-Dependent Phase Behavior: 2D Lennard-Jones System', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_psi6_evolution(times, psi6_values, title=None, filename=None, show=True):
    """
    Plot time evolution of ψ6 order parameter
    
    Parameters
    ----------
    times : array-like
        Time steps or simulation time
    psi6_values : array-like
        ψ6 values over time
    title : str or None, optional
        Plot title
    filename : str or None, optional
        If provided, save figure to this file
    show : bool, optional
        Whether to display the plot (default: True)
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    ax.plot(times, psi6_values, 'b-', linewidth=1.5)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, 
              alpha=0.7, label='Transition threshold')
    
    ax.set_xlabel('Simulation Step', fontsize=12)
    ax.set_ylabel(r'$\psi_6$', fontsize=12)
    ax.set_title(title if title else 'Time Evolution of Hexagonal Order', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, ax


def plot_configuration_snapshot(positions, box_size, psi6_local=None, 
                                title=None, filename=None, show=True):
    """
    Plot particle configuration with optional local order coloring
    
    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Particle positions
    box_size : float
        Box size
    psi6_local : array-like or None, optional
        Local ψ6 values for color mapping
    title : str or None, optional
        Plot title
    filename : str or None, optional
        If provided, save figure to this file
    show : bool, optional
        Whether to display the plot (default: True)
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    
    if psi6_local is not None:
        # Color by local order
        scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                           c=np.abs(psi6_local), s=50, cmap='viridis',
                           vmin=0, vmax=1, edgecolors='black', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(r'Local $|\psi_6|$', fontsize=12)
    else:
        # Uniform color
        ax.scatter(positions[:, 0], positions[:, 1], 
                  s=50, c='blue', edgecolors='black', linewidth=0.5)
    
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$ ($\sigma$)', fontsize=12)
    ax.set_ylabel(r'$y$ ($\sigma$)', fontsize=12)
    ax.set_title(title if title else 'Particle Configuration', fontsize=13)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, ax


def create_phase_diagram_plot(density_values, temperature_values, order_parameter_matrix,
                              xlabel='Density', ylabel='Temperature', 
                              title='Phase Diagram', filename=None, show=True):
    """
    Create a 2D phase diagram heatmap
    
    Parameters
    ----------
    density_values : array-like
        Density values (x-axis)
    temperature_values : array-like
        Temperature values (y-axis)
    order_parameter_matrix : ndarray, shape (len(T), len(rho))
        Order parameter values (e.g., ψ6) on grid
    xlabel, ylabel, title : str, optional
        Axis labels and title
    filename : str or None, optional
        If provided, save figure to this file
    show : bool, optional
        Whether to display the plot (default: True)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.contourf(density_values, temperature_values, order_parameter_matrix,
                    levels=20, cmap='RdYlBu_r')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\psi_6$', fontsize=12)
    
    # Add contour lines
    contours = ax.contour(density_values, temperature_values, order_parameter_matrix,
                         levels=[0.3, 0.5, 0.7], colors='black', linewidths=1.5)
    ax.clabel(contours, inline=True, fontsize=10)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, ax
