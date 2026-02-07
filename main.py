"""
Main Execution Script: Research-Grade 2D Molecular Dynamics
===========================================================

This script demonstrates the complete functionality of the upgraded
nanoparticle self-assembly simulator, including:

1. Basic MD simulation
2. Radial distribution function g(r) analysis
3. Static structure factor S(q) computation
4. Bond-orientational order parameter ψ6
5. Temperature sweep for phase behavior studies

Run different examples by uncommenting the desired section.

Author: Demonstration of statistical mechanics analysis tools
"""

import numpy as np
import matplotlib.pyplot as plt
from simulation import MDSimulation
from structure import StructureAnalyzer, time_averaged_structure
from analysis import TemperatureSweep, quick_phase_scan
from plotting import (plot_radial_distribution, plot_structure_factor,
                     plot_combined_structure, plot_temperature_sweep,
                     plot_psi6_evolution, plot_configuration_snapshot)


def example_1_basic_simulation():
    """
    Example 1: Basic MD simulation with structural analysis
    --------------------------------------------------------
    Demonstrates single-temperature simulation with g(r), S(q), and ψ6
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Simulation and Structural Analysis")
    print("="*70 + "\n")
    
    # Initialize simulation
    num_particles = 300
    box_size = 25.0
    temperature = 1.0
    
    print(f"System parameters:")
    print(f"  Number of particles: {num_particles}")
    print(f"  Box size: {box_size}")
    print(f"  Density: {num_particles/box_size**2:.4f}")
    print(f"  Temperature: {temperature}")
    
    sim = MDSimulation(
        num_particles=num_particles,
        box_size=box_size,
        temperature=temperature,
        epsilon=1.0,
        seed=42
    )
    
    # Equilibration
    print("\nEquilibrating system...")
    for i in range(500):
        sim.run_step()
        if i % 100 == 0:
            print(f"  Step {i}: E_pot = {sim.potential_energy/num_particles:.4f}, "
                  f"T_inst = {sim.get_instantaneous_temperature():.4f}")
    
    # Production run with structural analysis
    print("\nProduction run and structural analysis...")
    
    # Compute time-averaged structure
    avg_structure = time_averaged_structure(sim, num_samples=30, interval=10)
    
    print(f"\nResults:")
    print(f"  Average ψ6: {avg_structure['psi6']:.4f} ± {avg_structure['psi6_std']:.4f}")
    print(f"  First peak g(r): {np.max(avg_structure['g_r']):.3f}")
    print(f"  First peak S(q): {np.max(avg_structure['S_q']):.3f}")
    
    # Visualizations
    print("\nGenerating plots...")
    
    # Combined structure plot
    analyzer = StructureAnalyzer(sim.positions, sim.box_size)
    psi6_global, psi6_local = analyzer.bond_orientational_order_psi6()
    
    plot_combined_structure(
        avg_structure['r'], avg_structure['g_r'],
        avg_structure['q'], avg_structure['S_q'],
        psi6=avg_structure['psi6'],
        title=f"Structural Analysis (T = {temperature}, ρ = {num_particles/box_size**2:.3f})",
        filename="example1_structure.png",
        show=True
    )
    
    # Configuration snapshot with local order
    plot_configuration_snapshot(
        sim.positions, sim.box_size, psi6_local=psi6_local,
        title=f"Configuration Snapshot (ψ6 = {psi6_global:.3f})",
        filename="example1_snapshot.png",
        show=True
    )
    
    print("\nExample 1 completed!")
    print("Saved: example1_structure.png, example1_snapshot.png")


def example_2_structure_factor_detailed():
    """
    Example 2: Detailed S(q) analysis at different temperatures
    ----------------------------------------------------------
    Compares structure factor at low (solid-like) and high (fluid-like) T
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Structure Factor S(q) at Different Temperatures")
    print("="*70 + "\n")
    
    num_particles = 250
    box_size = 20.0
    temperatures = [0.5, 1.5]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    for idx, temp in enumerate(temperatures):
        print(f"\nRunning simulation at T = {temp}...")
        
        sim = MDSimulation(
            num_particles=num_particles,
            box_size=box_size,
            temperature=temp,
            seed=42
        )
        
        # Equilibrate
        for _ in range(800):
            sim.run_step()
        
        # Analyze
        analyzer = StructureAnalyzer(sim.positions, sim.box_size)
        q, S_q = analyzer.static_structure_factor(q_max=20.0, num_q=60)
        
        # Plot
        axes[idx].plot(q, S_q, 'b-', linewidth=2)
        axes[idx].set_xlabel(r'Wavevector $q$ ($\sigma^{-1}$)', fontsize=12)
        axes[idx].set_ylabel(r'$S(q)$', fontsize=12)
        axes[idx].set_title(f'T = {temp} ({"Solid-like" if temp < 1.0 else "Fluid-like"})',
                           fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        
        # Highlight peaks
        if temp < 1.0:
            # Find multiple peaks for crystalline phase
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(S_q, height=np.max(S_q)*0.3)
            if len(peaks) > 0:
                axes[idx].plot(q[peaks], S_q[peaks], 'ro', markersize=8, 
                             label=f'{len(peaks)} Bragg peaks')
                axes[idx].legend()
    
    plt.suptitle('Static Structure Factor: Phase Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("example2_structure_factor.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nExample 2 completed!")
    print("Saved: example2_structure_factor.png")


def example_3_psi6_time_evolution():
    """
    Example 3: ψ6 time evolution during crystallization
    --------------------------------------------------
    Shows how hexagonal order develops over time
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Time Evolution of Bond-Orientational Order")
    print("="*70 + "\n")
    
    num_particles = 200
    box_size = 18.0
    temperature = 0.6  # Low temperature for crystallization
    
    print(f"System parameters:")
    print(f"  Particles: {num_particles}, Box: {box_size}")
    print(f"  Temperature: {temperature} (low, favors crystallization)")
    print(f"  Density: {num_particles/box_size**2:.4f}")
    
    sim = MDSimulation(
        num_particles=num_particles,
        box_size=box_size,
        temperature=temperature,
        seed=123
    )
    
    # Track ψ6 over time
    times = []
    psi6_values = []
    
    print("\nSimulating crystallization process...")
    num_steps = 2000
    sample_interval = 20
    
    for step in range(num_steps):
        sim.run_step()
        
        if step % sample_interval == 0:
            analyzer = StructureAnalyzer(sim.positions, sim.box_size)
            psi6, _ = analyzer.bond_orientational_order_psi6()
            times.append(step)
            psi6_values.append(psi6)
            
            if step % 400 == 0:
                print(f"  Step {step}: ψ6 = {psi6:.4f}")
    
    # Plot evolution
    plot_psi6_evolution(
        times, psi6_values,
        title=f"Crystallization Dynamics (T = {temperature})",
        filename="example3_psi6_evolution.png",
        show=True
    )
    
    print(f"\nFinal ψ6: {psi6_values[-1]:.4f}")
    print("Example 3 completed!")
    print("Saved: example3_psi6_evolution.png")


def example_4_temperature_sweep():
    """
    Example 4: Temperature sweep to identify phase transition
    --------------------------------------------------------
    Systematic scan over temperature range
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Temperature Sweep for Phase Behavior")
    print("="*70 + "\n")
    
    # Use quick scan function for demonstration
    sweep = quick_phase_scan(
        num_particles=200,
        box_size=20.0,
        verbose=True
    )
    
    # Identify transition temperature
    T_transition = sweep.identify_transition_temperature(method='psi6', threshold=0.5)
    
    if T_transition:
        print(f"\nEstimated transition temperature: T_c ≈ {T_transition:.3f}")
        print(f"Reduced temperature: T* = T/ε ≈ {T_transition:.3f}")
    
    # Create comprehensive plots
    plot_temperature_sweep(
        sweep.results,
        filename="example4_phase_behavior.png",
        show=True
    )
    
    print("\nExample 4 completed!")
    print("Saved: example4_phase_behavior.png")


def example_5_custom_temperature_sweep():
    """
    Example 5: Custom temperature sweep with detailed control
    --------------------------------------------------------
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Custom Temperature Sweep")
    print("="*70 + "\n")
    
    # Custom temperature range
    temperatures = np.linspace(0.3, 2.0, 12)
    
    sweep = TemperatureSweep(
        num_particles=250,
        box_size=22.0,
        temperatures=temperatures,
        epsilon=1.0,
        equilibration_steps=1000,
        production_steps=600,
        sample_interval=10
    )
    
    results = sweep.run_sweep(verbose=True)
    
    # Analysis
    print("\nPhase characterization:")
    print("-" * 50)
    print(f"{'Temperature':<12} {'ψ6':<10} {'g(r) peak':<12} {'Phase'}")
    print("-" * 50)
    
    for i, T in enumerate(results['temperatures']):
        psi6 = results['psi6'][i]
        g_peak = results['g_r_peak_height'][i]
        phase = "Solid-like" if psi6 > 0.5 else "Fluid-like"
        print(f"{T:<12.3f} {psi6:<10.4f} {g_peak:<12.3f} {phase}")
    
    # Visualization
    plot_temperature_sweep(results, filename="example5_sweep.png", show=True)
    
    print("\nExample 5 completed!")
    print("Saved: example5_sweep.png")


def run_all_examples():
    """Run all example demonstrations"""
    example_1_basic_simulation()
    example_2_structure_factor_detailed()
    example_3_psi6_time_evolution()
    example_4_temperature_sweep()
    example_5_custom_temperature_sweep()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("2D MOLECULAR DYNAMICS: RESEARCH-GRADE STRUCTURAL ANALYSIS")
    print("="*70)
    print("\nSelect an example to run:")
    print("  1 - Basic simulation with g(r), S(q), ψ6")
    print("  2 - Structure factor S(q) comparison")
    print("  3 - Time evolution of ψ6 (crystallization)")
    print("  4 - Quick temperature sweep (phase transition)")
    print("  5 - Custom temperature sweep")
    print("  A - Run all examples")
    print("\nOr edit this file to uncomment specific examples.")
    print("="*70 + "\n")
    
    # UNCOMMENT THE EXAMPLE YOU WANT TO RUN:
    
    example_1_basic_simulation()
    # example_2_structure_factor_detailed()
    # example_3_psi6_time_evolution()
    # example_4_temperature_sweep()
    # example_5_custom_temperature_sweep()
    # run_all_examples()
    
    print("\n" + "="*70)
    print("Execution completed. Check generated PNG files for results.")
    print("="*70 + "\n")
