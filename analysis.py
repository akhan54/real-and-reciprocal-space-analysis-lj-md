"""
Temperature Sweep Analysis for Phase Behavior Studies
======================================================

This module performs systematic temperature scans to characterize phase
transitions and thermal properties of 2D Lennard-Jones systems.

The analysis identifies:
- Fluid-solid transitions via ψ6 order parameter
- Thermal energy evolution
- Structural correlation changes

This type of study is fundamental in soft matter physics for mapping
phase diagrams and understanding self-assembly processes.

Author: Phase transition studies for materials science applications
"""

import numpy as np
import warnings
from simulation import MDSimulation
from structure import StructureAnalyzer

# Try to import tqdm for progress bars, fall back to basic iteration if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        """Fallback tqdm that just returns the iterable"""
        return iterable


class TemperatureSweep:
    """
    Perform temperature sweep to study phase behavior
    
    Parameters
    ----------
    num_particles : int
        Number of particles
    box_size : float
        Simulation box size
    temperatures : array-like
        Array of temperatures to scan
    epsilon : float, optional
        LJ energy parameter (default: 1.0)
    equilibration_steps : int, optional
        Steps to equilibrate at each temperature (default: 1000)
    production_steps : int, optional
        Steps for production/sampling (default: 500)
    sample_interval : int, optional
        Steps between samples (default: 10)
    """
    
    def __init__(self, num_particles, box_size, temperatures, epsilon=1.0,
                 equilibration_steps=1000, production_steps=500, sample_interval=10):
        self.num_particles = num_particles
        self.box_size = box_size
        self.temperatures = np.array(temperatures)
        self.epsilon = epsilon
        self.equilibration_steps = equilibration_steps
        self.production_steps = production_steps
        self.sample_interval = sample_interval
        
        # Storage for results
        self.results = {
            'temperatures': self.temperatures,
            'potential_energy': [],
            'potential_energy_std': [],
            'kinetic_energy': [],
            'total_energy': [],
            'psi6': [],
            'psi6_std': [],
            'g_r_peak_height': [],
            'g_r_peak_position': [],
            'S_q_peak_height': [],
            'S_q_peak_position': []
        }
    
    def run_sweep(self, verbose=True):
        """
        Execute the temperature sweep
        
        For each temperature:
        1. Create/reinitialize simulation at target temperature
        2. Equilibrate the system
        3. Sample structural properties during production run
        4. Compute time-averaged observables
        
        Parameters
        ----------
        verbose : bool, optional
            Print progress information (default: True)
        
        Returns
        -------
        results : dict
            Temperature-dependent observables
        """
        if verbose:
            print(f"Temperature Sweep Analysis")
            print(f"{'='*60}")
            print(f"Number of particles: {self.num_particles}")
            print(f"Box size: {self.box_size:.2f}")
            print(f"Density: {self.num_particles/self.box_size**2:.4f}")
            print(f"Temperature range: {self.temperatures[0]:.3f} - {self.temperatures[-1]:.3f}")
            print(f"Number of temperatures: {len(self.temperatures)}")
            print(f"{'='*60}\n")
        
        # Use tqdm for progress bar if available
        if verbose and HAS_TQDM:
            iterator = tqdm(self.temperatures, desc="Temperature sweep")
        else:
            iterator = self.temperatures
        
        for T in iterator:
            if verbose and not HAS_TQDM:
                print(f"T = {T:.3f}...", end=" ")
            
            # Run simulation at this temperature
            T_results = self._run_single_temperature(T)
            
            # Store results
            self.results['potential_energy'].append(T_results['potential_energy'])
            self.results['potential_energy_std'].append(T_results['potential_energy_std'])
            self.results['kinetic_energy'].append(T_results['kinetic_energy'])
            self.results['total_energy'].append(T_results['total_energy'])
            self.results['psi6'].append(T_results['psi6'])
            self.results['psi6_std'].append(T_results['psi6_std'])
            self.results['g_r_peak_height'].append(T_results['g_r_peak_height'])
            self.results['g_r_peak_position'].append(T_results['g_r_peak_position'])
            self.results['S_q_peak_height'].append(T_results['S_q_peak_height'])
            self.results['S_q_peak_position'].append(T_results['S_q_peak_position'])
            
            if verbose and not HAS_TQDM:
                print(f"ψ6 = {T_results['psi6']:.4f}, E_pot = {T_results['potential_energy']:.4f}")
        
        # Convert lists to arrays
        for key in self.results:
            if key != 'temperatures':
                self.results[key] = np.array(self.results[key])
        
        if verbose:
            print(f"\n{'='*60}")
            print("Temperature sweep completed!")
            print(f"{'='*60}\n")
        
        return self.results
    
    def _run_single_temperature(self, T):
        """
        Run simulation at a single temperature and collect statistics
        
        Parameters
        ----------
        T : float
            Temperature
        
        Returns
        -------
        T_results : dict
            Observables at this temperature
        """
        # Initialize simulation
        sim = MDSimulation(
            num_particles=self.num_particles,
            box_size=self.box_size,
            temperature=T,
            epsilon=self.epsilon,
            seed=None  # Different initialization each time for better sampling
        )
        
        # Equilibration phase
        for _ in range(self.equilibration_steps):
            sim.run_step()
        
        # Production phase: sample observables
        potential_energies = []
        kinetic_energies = []
        psi6_values = []
        g_r_samples = []
        S_q_samples = []
        
        num_samples = self.production_steps // self.sample_interval
        
        for sample_idx in range(num_samples):
            # Run between samples
            for _ in range(self.sample_interval):
                sim.run_step()
            
            # Collect energy data
            potential_energies.append(sim.potential_energy / self.num_particles)
            kinetic_energies.append(sim.kinetic_energy / self.num_particles)
            
            # Structural analysis
            analyzer = StructureAnalyzer(sim.positions, sim.box_size)
            
            # Compute ψ6
            psi6, _ = analyzer.bond_orientational_order_psi6(cutoff=1.5)
            psi6_values.append(psi6)
            
            # Compute g(r) and S(q) (less frequently to save time)
            if sample_idx % 5 == 0:  # Every 5th sample
                r, g_r = analyzer.radial_distribution_function(num_bins=100, r_max=5.0)
                q, S_q = analyzer.static_structure_factor(q_max=15.0, num_q=50)
                g_r_samples.append(g_r)
                S_q_samples.append(S_q)
        
        # Compute statistics
        potential_energies = np.array(potential_energies)
        kinetic_energies = np.array(kinetic_energies)
        psi6_values = np.array(psi6_values)
        
        # Time-averaged g(r) and S(q)
        if len(g_r_samples) > 0:
            g_r_avg = np.mean(g_r_samples, axis=0)
            S_q_avg = np.mean(S_q_samples, axis=0)
            
            # Find peaks in g(r) - first peak after r > 0.5
            peak_region = r > 0.5
            if np.any(peak_region):
                peak_idx = np.argmax(g_r_avg[peak_region])
                g_r_peak_height = g_r_avg[peak_region][peak_idx]
                g_r_peak_position = r[peak_region][peak_idx]
            else:
                g_r_peak_height = 0.0
                g_r_peak_position = 0.0
            
            # Find peak in S(q) - skip q=0 region
            peak_region = q > 1.0
            if np.any(peak_region):
                peak_idx = np.argmax(S_q_avg[peak_region])
                S_q_peak_height = S_q_avg[peak_region][peak_idx]
                S_q_peak_position = q[peak_region][peak_idx]
            else:
                S_q_peak_height = 0.0
                S_q_peak_position = 0.0
        else:
            g_r_peak_height = 0.0
            g_r_peak_position = 0.0
            S_q_peak_height = 0.0
            S_q_peak_position = 0.0
        
        return {
            'potential_energy': np.mean(potential_energies),
            'potential_energy_std': np.std(potential_energies),
            'kinetic_energy': np.mean(kinetic_energies),
            'total_energy': np.mean(potential_energies + kinetic_energies),
            'psi6': np.mean(psi6_values),
            'psi6_std': np.std(psi6_values),
            'g_r_peak_height': g_r_peak_height,
            'g_r_peak_position': g_r_peak_position,
            'S_q_peak_height': S_q_peak_height,
            'S_q_peak_position': S_q_peak_position
        }
    
    def identify_transition_temperature(self, method='psi6', threshold=0.5):
        """
        Estimate phase transition temperature
        
        Parameters
        ----------
        method : str, optional
            Method to identify transition: 'psi6', 'energy', or 'peak_height'
            (default: 'psi6')
        threshold : float, optional
            Threshold value for transition (default: 0.5 for ψ6)
        
        Returns
        -------
        T_transition : float
            Estimated transition temperature
        """
        if method == 'psi6':
            # Find where ψ6 crosses threshold
            psi6 = self.results['psi6']
            crossings = np.where(np.diff(np.sign(psi6 - threshold)))[0]
            if len(crossings) > 0:
                idx = crossings[0]
                # Linear interpolation
                T_low, T_high = self.temperatures[idx], self.temperatures[idx + 1]
                psi6_low, psi6_high = psi6[idx], psi6[idx + 1]
                T_transition = T_low + (threshold - psi6_low) * (T_high - T_low) / (psi6_high - psi6_low)
                return T_transition
        
        elif method == 'energy':
            # Find maximum in heat capacity (dE/dT)
            energy = self.results['potential_energy']
            dE_dT = np.gradient(energy, self.temperatures)
            idx = np.argmax(dE_dT)
            return self.temperatures[idx]
        
        elif method == 'peak_height':
            # Find maximum in g(r) first peak
            peak_heights = self.results['g_r_peak_height']
            dPeak_dT = np.gradient(peak_heights, self.temperatures)
            idx = np.argmax(np.abs(dPeak_dT))
            return self.temperatures[idx]
        
        return None
    
    def get_reduced_temperatures(self):
        """
        Return reduced temperatures T* = k_B T / ε
        
        Returns
        -------
        T_star : ndarray
            Reduced temperatures
        """
        return self.results['temperatures'] / self.epsilon


def quick_phase_scan(num_particles=400, box_size=20.0, verbose=True):
    """
    Perform a quick phase scan to demonstrate phase transition
    
    Parameters
    ----------
    num_particles : int, optional
        Number of particles (default: 200)
    box_size : float, optional
        Box size (default: 20.0)
    verbose : bool, optional
        Print progress (default: True)
    
    Returns
    -------
    sweep : TemperatureSweep
        Completed temperature sweep object
    """
    # Temperature range covering fluid and solid phases
    temperatures = np.array([0.25, 0.35, 0.5, 0.8, 1.2, 1.8, 2.5, 3.0])
    
    sweep = TemperatureSweep(
        num_particles=num_particles,
        box_size=box_size,
        temperatures=temperatures,
        epsilon=1.0,
        equilibration_steps=2000,
        production_steps=1200,
        sample_interval=5
    )
    
    results = sweep.run_sweep(verbose=verbose)
    
    return sweep
