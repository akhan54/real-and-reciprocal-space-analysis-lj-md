# Usage Guide

## Installation

```bash
pip install numpy scipy matplotlib tqdm
```

## Modules

simulation.py - Molecular dynamics engine  
structure.py - Structural analysis (g(r), S(q), ψ6)  
analysis.py - Temperature sweep  
plotting.py - Plotting utilities  
main.py - Example scripts

## Quick Examples

### 1. Basic Simulation with g(r), S(q), and ψ6

```python
from simulation import MDSimulation
from structure import StructureAnalyzer
from plotting import plot_combined_structure

# Initialize system
sim = MDSimulation(
    num_particles=300,
    box_size=25.0,
    temperature=1.0,
    seed=42
)

# Equilibrate
for _ in range(1000):
    sim.run_step()

# Analyze structure
analyzer = StructureAnalyzer(sim.positions, sim.box_size)
r, g_r = analyzer.radial_distribution_function()
q, S_q = analyzer.static_structure_factor()
psi6_global, psi6_local = analyzer.bond_orientational_order_psi6()

print(f"Bond-orientational order: ψ6 = {psi6_global:.4f}")

# Visualize
plot_combined_structure(r, g_r, q, S_q, psi6=psi6_global,
                       title=f"Structural Analysis (T = 1.0)",
                       filename="structure_analysis.png")
```

### 2. Temperature Sweep (Phase Transition Study)

```python
import numpy as np
from analysis import TemperatureSweep
from plotting import plot_temperature_sweep

# Define temperature range
temperatures = np.linspace(0.3, 2.0, 15)

# Create and run sweep
sweep = TemperatureSweep(
    num_particles=200,
    box_size=20.0,
    temperatures=temperatures,
    equilibration_steps=1000,
    production_steps=500
)

results = sweep.run_sweep(verbose=True)

# Identify transition temperature
T_c = sweep.identify_transition_temperature(method='psi6', threshold=0.5)
print(f"\nEstimated transition temperature: T_c = {T_c:.3f}")

# Visualize phase behavior
plot_temperature_sweep(results, filename="phase_diagram.png")
```

### 3. Time Evolution of Crystallization

```python
from simulation import MDSimulation
from structure import StructureAnalyzer
from plotting import plot_psi6_evolution

# Low temperature favors crystallization
sim = MDSimulation(
    num_particles=200,
    box_size=18.0,
    temperature=0.6,  # Low T
    seed=123
)

# Track ψ6 over time
times = []
psi6_values = []

for step in range(2000):
    sim.run_step()

    if step % 20 == 0:
        analyzer = StructureAnalyzer(sim.positions, sim.box_size)
        psi6, _ = analyzer.bond_orientational_order_psi6()
        times.append(step)
        psi6_values.append(psi6)

# Plot crystallization dynamics
plot_psi6_evolution(times, psi6_values,
                   title="Crystallization Dynamics",
                   filename="psi6_evolution.png")
```

### 4. Complete Structural Analysis

```python
from structure import StructureAnalyzer

# After running simulation...
analyzer = StructureAnalyzer(sim.positions, sim.box_size)

# Compute all properties at once
results = analyzer.compute_all_structure(
    r_max=5.0,
    q_max=15.0,
    psi6_cutoff=1.5
)

print(f"Radial distribution function:")
print(f"  First peak at r = {results['r'][np.argmax(results['g_r'])]:.3f}")
print(f"  Peak height g(r_max) = {np.max(results['g_r']):.3f}")

print(f"\nStructure factor:")
print(f"  First peak at q = {results['q'][np.argmax(results['S_q'])]:.3f}")
print(f"  Peak intensity S(q_max) = {np.max(results['S_q']):.3f}")

print(f"\nBond-orientational order:")
print(f"  Global ψ6 = {results['psi6_global']:.4f}")
```

### 5. Configuration Snapshot with Local Order

```python
from plotting import plot_configuration_snapshot

# Compute local order parameters
analyzer = StructureAnalyzer(sim.positions, sim.box_size)
psi6_global, psi6_local = analyzer.bond_orientational_order_psi6()

# Visualize with color-coded local order
plot_configuration_snapshot(
    positions=sim.positions,
    box_size=sim.box_size,
    psi6_local=psi6_local,
    title=f"Configuration (ψ6 = {psi6_global:.3f})",
    filename="snapshot.png"
)
```

## Running Pre-Built Examples

The `main.py` file contains five complete examples:

```bash
python main.py
```

By default, it runs Example 1. To run different examples, edit the file and uncomment:

- `example_1_basic_simulation()` - Basic g(r), S(q), ψ6 analysis
- `example_2_structure_factor_detailed()` - S(q) at different temperatures
- `example_3_psi6_time_evolution()` - Crystallization dynamics
- `example_4_temperature_sweep()` - Quick phase scan
- `example_5_custom_temperature_sweep()` - Detailed temperature study
