# Project Overview: 2D Lennard-Jones Molecular Dynamics

## Overview

This project implements a 2D Lennard-Jones molecular dynamics simulation with structural analysis tools including the radial distribution function, static structure factor, and bond-orientational order parameter.

## Project Structure

### Core Modules

```
simulation.py        - Molecular dynamics engine
structure.py         - Structural analysis tools
analysis.py          - Temperature-dependent analysis
plotting.py          - Visualization utilities
main.py              - Example runs

```

### Structural Analysis Methods

#### **Static Structure Factor S(q)**

- **Location**: `structure.py` → `StructureAnalyzer.static_structure_factor()`
- **Computation method**: Direct summation S(q) = (1/N)|Σⱼ exp(iq·rⱼ)|²
- **Features**:
  - Vectorized NumPy implementation for efficiency
  - Isotropic averaging over wavevector magnitudes
  - Configurable q-space resolution
  - Returns q vs S(q) ready for plotting

Computes the static structure factor S(q) from particle positions using direct summation.

**Usage**:

```python
analyzer = StructureAnalyzer(positions, box_size)
q, S_q = analyzer.static_structure_factor(q_max=15.0, num_q=50)
```

#### **Bond-Orientational Order Parameter ψ₆**

- **Location**: `structure.py` → `StructureAnalyzer.bond_orientational_order_psi6()`
- **Computation method**: ψ₆ = (1/N) Σⱼ |(1/nⱼ) Σₖ exp(i 6θⱼₖ)|
- **Features**:
  - KDTree-based efficient neighbor search
  - Handles periodic boundaries correctly
  - Returns both global and local ψ₆ values
  - Configurable cutoff distance

**Usage**:

```python
psi6_global, psi6_local = analyzer.bond_orientational_order_psi6(cutoff=1.5)
```

#### **Temperature Sweep Analysis**

- **Location**: `analysis.py` → `TemperatureSweep` class
- **Features**:
  - Systematic temperature scanning
  - Automatic equilibration at each T
  - Statistical sampling and error estimation
  - Computes: energy, ψ₆, g(r) peaks, S(q) peaks
  - Computes temperature-dependent structural metrics.

**Usage**:

```python
sweep = TemperatureSweep(
    num_particles=200,
    box_size=20.0,
    temperatures=np.linspace(0.3, 2.0, 15)
)
results = sweep.run_sweep()
```

### Enhanced Radial Distribution Function

- **Location**: `structure.py` → `StructureAnalyzer.radial_distribution_function()`
- **Improvements over original**:
  - Correct 2D normalization (2πr instead of 4πr²)
  - Clean, documented implementation
  - Configurable binning and range
  - Returns data arrays for further analysis

### Plotting

- **Location**: `plotting.py`
- **Functions**:
  - `plot_radial_distribution()` - g(r) with proper labels
  - `plot_structure_factor()` - S(q) with peak highlighting
  - `plot_combined_structure()` - Side-by-side g(r) and S(q)
  - `plot_temperature_sweep()` - Comprehensive 4-panel phase plot
  - `plot_psi6_evolution()` - Time series of order parameter
  - `plot_configuration_snapshot()` - Spatial distribution with color maps

## Quick Start Guide

### Installation

```bash
# Install dependencies
pip install numpy scipy matplotlib

# Optional (for progress bars)
pip install tqdm
```

### Running Examples

The `main.py` file contains 5 complete examples:

```bash
# Test installation first
python test_installation.py

# Run examples (edit main.py to select which one)
python main.py
```

**Example 1**: Basic simulation with all structural analysis  
**Example 2**: Structure factor S(q) at different temperatures  
**Example 3**: Time evolution of ψ₆ (crystallization dynamics)  
**Example 4**: Quick temperature sweep (phase transition)  
**Example 5**: Custom temperature sweep with detailed control

### Custom Usage

```python
# Import modules
from simulation import MDSimulation
from structure import StructureAnalyzer
from plotting import plot_combined_structure

# Create simulation
sim = MDSimulation(
    num_particles=300,
    box_size=25.0,
    temperature=1.0,
    epsilon=1.0,
    seed=42
)

# Equilibrate
for _ in range(1000):
    sim.run_step()

# Analyze structure
analyzer = StructureAnalyzer(sim.positions, sim.box_size)
results = analyzer.compute_all_structure()

# results contains: 'r', 'g_r', 'q', 'S_q', 'psi6_global', 'psi6_local'

# Visualize
plot_combined_structure(
    results['r'], results['g_r'],
    results['q'], results['S_q'],
    psi6=results['psi6_global'],
    filename='my_analysis.png'
)
```

## Connection to Experiments

### X-ray/Neutron Scattering

Your computed S(q) can be directly compared to:

- Small-angle X-ray scattering (SAXS) on nanoparticle films
- Grazing-incidence diffraction
- Neutron scattering on soft matter

**Procedure**:

1. Compute S(q) from simulation
2. Convert q to 2θ scattering angle (q = 4π sin(θ)/λ)
3. Compare peak positions and intensities

### Particle Tracking

Your ψ₆ analysis matches:

- Video microscopy of colloidal monolayers
- Optical tracking of confined particles
- Electron microscopy of nanoparticle arrays

**Procedure**:

1. Extract particle positions from images
2. Compute ψ₆ using same algorithm
3. Compare order parameter distributions

## Key Equations Implemented

### Lennard-Jones Potential

```
U(r) = 4ε[(σ/r)¹² - (σ/r)⁶]
```

### Radial Distribution Function (2D)

```
g(r) = (1/ρ) ⟨ΣᵢΣⱼ≠ᵢ δ(r - rᵢⱼ)⟩ / (2πr dr)
```

### Static Structure Factor

```
S(q) = (1/N) |Σⱼ exp(iq·rⱼ)|²
```

### Bond-Orientational Order

```
ψ₆ = (1/N) Σⱼ |(1/nⱼ) Σₖ exp(i 6θⱼₖ)|
```
