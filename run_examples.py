#!/usr/bin/env python3
"""
Simple Demo: All Features in One Script
========================================

This script demonstrates the complete research-grade functionality
in a single, compact example.

Run this to verify installation and see all features working together.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import our modules
from simulation import MDSimulation
from structure import StructureAnalyzer
from analysis import TemperatureSweep
from plotting import plot_temperature_sweep


def main():
    """Run a complete demonstration"""
    
    print("="*70)
    print("2D LENNARD-JONES SIMULATOR: COMPLETE FEATURE DEMONSTRATION")
    print("="*70)
    print()
    
    # =========================================================================
    # PART 1: Single Temperature Analysis
    # =========================================================================
    print("PART 1: Single Temperature Structural Analysis")
    print("-"*70)
    
    # System parameters
    num_particles = 250
    box_size = 22.0
    temperature = 0.8  # Moderate temperature
    
    print(f"System: {num_particles} particles, box = {box_size}")
    print(f"Temperature: T = {temperature}")
    print(f"Density: ρ = {num_particles/box_size**2:.4f}")
    print()
    
    # Initialize simulation
    print("Initializing MD simulation...")
    sim = MDSimulation(
        num_particles=num_particles,
        box_size=box_size,
        temperature=temperature,
        epsilon=1.0,
        seed=42
    )
    
    # Equilibration
    print("Equilibrating (1000 steps)...")
    for i in range(1000):
        sim.run_step()
        if i % 200 == 0:
            E_pot = sim.potential_energy / num_particles
            T_inst = sim.get_instantaneous_temperature()
            print(f"  Step {i:4d}: E_pot = {E_pot:7.4f}, T_inst = {T_inst:.4f}")
    
    print()
    print("Computing structural properties...")
    
    # Structural analysis
    analyzer = StructureAnalyzer(sim.positions, sim.box_size)
    
    # 1. Radial distribution function
    r, g_r = analyzer.radial_distribution_function(num_bins=100, r_max=5.0)
    
    # 2. Static structure factor
    q, S_q = analyzer.static_structure_factor(q_max=15.0, num_q=60)
    
    # 3. Bond-orientational order
    psi6_global, psi6_local = analyzer.bond_orientational_order_psi6(cutoff=1.5)
    
    # Report results
    print()
    print("Results:")
    print(f"  g(r) first peak height: {np.max(g_r):.3f}")
    print(f"  g(r) first peak at r = {r[np.argmax(g_r)]:.3f}")
    print(f"  S(q) peak height: {np.max(S_q):.3f}")
    print(f"  S(q) peak at q = {q[np.argmax(S_q)]:.3f}")
    print(f"  Bond-orientational order: ψ6 = {psi6_global:.4f}")
    print()
    
    # Determine phase
    if psi6_global > 0.7:
        phase = "Crystalline solid"
    elif psi6_global > 0.3:
        phase = "Hexatic (intermediate order)"
    else:
        phase = "Fluid (liquid/gas)"
    print(f"Phase classification: {phase}")
    print()
    
    # =========================================================================
    # PART 2: Visualize Single-Temperature Results
    # =========================================================================
    print("-"*70)
    print("PART 2: Creating Visualization")
    print("-"*70)
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Radial distribution function
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(r, g_r, 'b-', linewidth=2.5)
    ax1.axhline(y=1.0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel(r'Distance $r$ ($\sigma$)', fontsize=12)
    ax1.set_ylabel(r'$g(r)$', fontsize=12)
    ax1.set_title('Radial Distribution Function', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.95, f'Peak: {np.max(g_r):.2f}',
             transform=ax1.transAxes, va='top', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Static structure factor
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(q, S_q, 'r-', linewidth=2.5)
    ax2.set_xlabel(r'Wavevector $q$ ($\sigma^{-1}$)', fontsize=12)
    ax2.set_ylabel(r'$S(q)$', fontsize=12)
    ax2.set_title('Static Structure Factor', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    peak_q = q[np.argmax(S_q)]
    peak_S = np.max(S_q)
    ax2.plot(peak_q, peak_S, 'ro', markersize=10, label=f'Peak at q={peak_q:.2f}')
    ax2.legend()
    
    # Plot 3: Configuration with local order
    ax3 = fig.add_subplot(gs[1, 0])
    scatter = ax3.scatter(sim.positions[:, 0], sim.positions[:, 1],
                         c=np.abs(psi6_local), s=60, cmap='viridis',
                         vmin=0, vmax=1, edgecolors='black', linewidth=0.5)
    ax3.set_xlim(0, box_size)
    ax3.set_ylim(0, box_size)
    ax3.set_aspect('equal')
    ax3.set_xlabel(r'$x$ ($\sigma$)', fontsize=12)
    ax3.set_ylabel(r'$y$ ($\sigma$)', fontsize=12)
    ax3.set_title(f'Configuration (ψ6 = {psi6_global:.3f})', 
                  fontsize=13, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label(r'Local $|\psi_6|$', fontsize=11)
    
    # Plot 4: Summary statistics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    summary_text = f"""
    SYSTEM PARAMETERS
    ─────────────────────────────
    Particles:     {num_particles}
    Box size:      {box_size:.1f} σ
    Density:       {num_particles/box_size**2:.4f} σ⁻²
    Temperature:   {temperature:.2f} (ε/k_B)
    
    STRUCTURAL ANALYSIS
    ─────────────────────────────
    g(r) peak:     {np.max(g_r):.3f} at r = {r[np.argmax(g_r)]:.3f}
    S(q) peak:     {np.max(S_q):.3f} at q = {q[np.argmax(S_q)]:.3f}
    
    ORDER PARAMETERS
    ─────────────────────────────
    ψ₆:            {psi6_global:.4f}
    Phase:         {phase}
    
    ENERGETICS
    ─────────────────────────────
    E_pot/N:       {sim.potential_energy/num_particles:.4f} ε
    E_kin/N:       {sim.kinetic_energy/num_particles:.4f} ε
    E_tot/N:       {sim.get_total_energy()/num_particles:.4f} ε
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    fig.suptitle('2D Lennard-Jones System: Complete Structural Analysis',
                fontsize=15, fontweight='bold')
    
    plt.savefig('single_temperature_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: single_temperature_analysis.png")
    print()
    
    # =========================================================================
    # PART 3: Temperature Sweep (Phase Transition Study)
    # =========================================================================
    print("-"*70)
    print("PART 3: Temperature Sweep Analysis")
    print("-"*70)
    print()
    
    # Define temperature range
    temperatures = np.linspace(0.4, 1.8, 10)
    
    print(f"Temperature range: {temperatures[0]:.2f} - {temperatures[-1]:.2f}")
    print(f"Number of points: {len(temperatures)}")
    print()
    
    # Create sweep
    sweep = TemperatureSweep(
        num_particles=200,
        box_size=20.0,
        temperatures=temperatures,
        epsilon=1.0,
        equilibration_steps=800,
        production_steps=400,
        sample_interval=8
    )
    
    # Run sweep
    print("Running temperature sweep...")
    results = sweep.run_sweep(verbose=True)
    
    # Identify transition
    T_c = sweep.identify_transition_temperature(method='psi6', threshold=0.5)
    if T_c:
        print()
        print(f"Estimated transition temperature: T_c ≈ {T_c:.3f} (ε/k_B)")
        print()
    
    # Create phase behavior plot
    plot_temperature_sweep(
        results,
        filename='temperature_sweep_results.png',
        show=False
    )
    print("Saved: temperature_sweep_results.png")
    print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print()
    print("Generated files:")
    print("  1. single_temperature_analysis.png - Complete structural analysis")
    print("  2. temperature_sweep_results.png  - Phase behavior study")
    print()
    print("This simulation demonstrates:")
    print("  ✓ Radial distribution function g(r)")
    print("  ✓ Static structure factor S(q)")
    print("  ✓ Bond-orientational order parameter ψ6")
    print("  ✓ Temperature-dependent phase behavior")
    print("  ✓ Publication-quality visualization")
    print()
    print("All features working correctly!")
    print("="*70)


if __name__ == "__main__":
    main()
