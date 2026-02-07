"""
Structural Analysis Tools for 2D Soft Matter Systems
====================================================

This module implements advanced structure characterization methods:
1. Radial Distribution Function g(r) - pair correlation function
2. Static Structure Factor S(q) - Fourier transform of spatial correlations
3. Bond-Orientational Order Parameter ψ6 - hexagonal crystalline order

These tools are essential for characterizing phase behavior, crystallization,
and structural ordering in soft matter systems such as colloidal suspensions,
nanoparticle assemblies, and 2D materials.

Author: Statistical mechanics analysis for condensed matter research
"""

import numpy as np
from scipy.spatial import cKDTree


class StructureAnalyzer:
    """
    Analyzes structural properties of 2D particle configurations
    
    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Particle positions
    box_size : float
        Periodic box side length
    """
    
    def __init__(self, positions, box_size):
        self.positions = positions
        self.box_size = box_size
        self.num_particles = len(positions)
    
    def radial_distribution_function(self, num_bins=100, r_max=None):
        """
        Compute the radial distribution function g(r)
        
        The RDF measures the probability of finding a particle at distance r
        from a reference particle, normalized by the ideal gas distribution.
        
        Physical interpretation:
        - g(r) → 0 for r < σ (hard-core exclusion)
        - g(r) → 1 for r → ∞ (uncorrelated)
        - Peaks indicate preferred neighbor distances (shell structure)
        - Oscillations decay in liquids, persist in crystals
        
        Parameters
        ----------
        num_bins : int, optional
            Number of bins for histogram (default: 100)
        r_max : float or None, optional
            Maximum distance to consider (default: box_size/2)
        
        Returns
        -------
        r : ndarray
            Radial distances (bin centers)
        g_r : ndarray
            Radial distribution function values
        """
        if r_max is None:
            r_max = self.box_size / 2.0
        
        # Collect all pair distances with periodic boundary conditions
        distances = []
        for i in range(self.num_particles):
            for j in range(i + 1, self.num_particles):
                # Minimum image convention
                dr = self.positions[i] - self.positions[j]
                dr -= self.box_size * np.rint(dr / self.box_size)
                r = np.linalg.norm(dr)
                distances.append(r)
        
        # Histogram of distances
        hist, bin_edges = np.histogram(distances, bins=num_bins, range=(0, r_max))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        dr = bin_edges[1] - bin_edges[0]
        
        # Number density
        rho = self.num_particles / self.box_size**2
        
        # Normalization for 2D: number of ideal gas particles in annulus
        # For unique pairs (i<j): norm = N * ρ * 2πr * dr
        norm = 2.0 * np.pi * bin_centers * dr * self.num_particles * rho
        norm[norm == 0] = 1.0  # Avoid division by zero
        
        g_r = hist / norm
        
        return bin_centers, g_r
    
    def static_structure_factor(self, q_max=None, num_q=50):
        """
        Compute the static structure factor S(q) via direct summation
        
        The structure factor is the Fourier transform of the pair correlation:
            S(q) = (1/N) |Σ_j exp(i q·r_j)|²
        
        Physical interpretation:
        - S(q) is measured in X-ray and neutron scattering experiments
        - Peaks in S(q) indicate characteristic length scales (2π/q)
        - Bragg peaks appear for crystalline order
        - S(q→0) relates to isothermal compressibility
        
        For a 2D system, we compute S(q) for a grid of wavevectors and
        then perform isotropic averaging to get S(|q|).
        
        Parameters
        ----------
        q_max : float or None, optional
            Maximum wavevector magnitude (default: 2π/box_size * 20)
        num_q : int, optional
            Number of q-magnitude bins (default: 50)
        
        Returns
        -------
        q_values : ndarray
            Wavevector magnitudes
        S_q : ndarray
            Structure factor values (isotropically averaged)
        """
        if q_max is None:
            q_max = 2.0 * np.pi / self.box_size * 20.0
        
        # Generate grid of q-vectors in 2D
        # Use a square grid in reciprocal space
        n_grid = 30  # Grid points per dimension
        q_min = 2.0 * np.pi / self.box_size
        
        qx = np.linspace(-q_max, q_max, n_grid)
        qy = np.linspace(-q_max, q_max, n_grid)
        QX, QY = np.meshgrid(qx, qy)
        
        # Flatten for vectorized computation
        qx_flat = QX.flatten()
        qy_flat = QY.flatten()
        q_mag = np.sqrt(qx_flat**2 + qy_flat**2)
        
        # Compute structure factor for each q-vector
        # S(q) = (1/N) |Σ_j exp(i q·r_j)|²
        positions_x = self.positions[:, 0]
        positions_y = self.positions[:, 1]
        
        # Vectorized computation: shape (num_q_vectors, num_particles)
        phase_x = np.outer(qx_flat, positions_x)  # q_x * x_j
        phase_y = np.outer(qy_flat, positions_y)  # q_y * y_j
        phase_total = phase_x + phase_y
        
        # Sum over particles: Σ_j exp(i q·r_j)
        rho_q = np.sum(np.exp(1j * phase_total), axis=1)
        
        # Structure factor: |ρ(q)|² / N
        S_q_raw = np.abs(rho_q)**2 / self.num_particles
        
        # Isotropic averaging: bin by |q|
        q_bins = np.linspace(0, q_max, num_q + 1)
        q_values = 0.5 * (q_bins[:-1] + q_bins[1:])
        S_q = np.zeros(num_q)
        counts = np.zeros(num_q)
        
        for i in range(len(q_mag)):
            if q_mag[i] < q_max:
                bin_idx = np.searchsorted(q_bins, q_mag[i]) - 1
                if 0 <= bin_idx < num_q:
                    S_q[bin_idx] += S_q_raw[i]
                    counts[bin_idx] += 1
        
        # Average within each bin
        mask = counts > 0
        S_q[mask] /= counts[mask]
        
        return q_values, S_q
    
    def bond_orientational_order_psi6(self, cutoff=1.5):
        """
        Compute the hexagonal bond-orientational order parameter ψ6
        
        The ψ6 parameter quantifies hexagonal symmetry in 2D systems:
            ψ6 = (1/N) Σ_j |ψ6_j|
        where for particle j:
            ψ6_j = (1/n_j) Σ_k exp(i 6 θ_jk)
        
        θ_jk is the angle of the bond between particle j and neighbor k.
        
        Physical interpretation:
        - ψ6 ≈ 0: Disordered (liquid/gas phase)
        - ψ6 ≈ 1: Perfect hexagonal crystalline order (2D solid)
        - Intermediate values: Defective crystal or hexatic phase
        
        This parameter is crucial for detecting the Kosterlitz-Thouless-Halperin-
        Nelson-Young (KTHNY) melting transition in 2D systems.
        
        Parameters
        ----------
        cutoff : float, optional
            Neighbor cutoff distance (default: 1.5σ, first coordination shell)
        
        Returns
        -------
        psi6_global : float
            Global ψ6 order parameter (magnitude)
        psi6_local : ndarray
            Local ψ6 for each particle
        """
        # Build KDTree for efficient neighbor searching
        # Handle periodic boundaries by creating image particles
        positions_extended = []
        particle_indices = []
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                offset = np.array([dx * self.box_size, dy * self.box_size])
                positions_extended.append(self.positions + offset)
                particle_indices.extend(range(self.num_particles))
        
        positions_extended = np.vstack(positions_extended)
        tree = cKDTree(positions_extended)
        
        # Compute local ψ6 for each particle
        psi6_local = np.zeros(self.num_particles, dtype=complex)
        
        for i in range(self.num_particles):
            # Find neighbors within cutoff (excluding self)
            neighbors = tree.query_ball_point(self.positions[i], r=cutoff)
            
            # Filter to get unique particle neighbors (accounting for periodic images)
            neighbor_particles = [particle_indices[n] for n in neighbors]
            neighbor_positions = [positions_extended[n] for n in neighbors]
            
            # Remove self and duplicate periodic images
            unique_neighbors = []
            unique_positions = []
            seen_particles = set()
            
            for np_id, np_pos in zip(neighbor_particles, neighbor_positions):
                if np_id != i and np_id not in seen_particles:
                    seen_particles.add(np_id)
                    unique_neighbors.append(np_id)
                    unique_positions.append(np_pos)
            
            if len(unique_neighbors) == 0:
                continue
            
            # Compute angles θ_jk for all neighbors k of particle j
            neighbor_positions = np.array(unique_positions)
            dr = neighbor_positions - self.positions[i]
            angles = np.arctan2(dr[:, 1], dr[:, 0])
            
            # ψ6_j = (1/n_j) Σ_k exp(i 6 θ_jk)
            psi6_local[i] = np.mean(np.exp(1j * 6.0 * angles))
        
        # Global order parameter: average magnitude
        psi6_global = np.mean(np.abs(psi6_local))
        
        return psi6_global, psi6_local
    
    def compute_all_structure(self, r_max=None, q_max=None, psi6_cutoff=1.5):
        """
        Compute all structural properties in one call
        
        Returns
        -------
        results : dict
            Dictionary containing:
            - 'r', 'g_r': radial distribution function
            - 'q', 'S_q': static structure factor
            - 'psi6_global': global hexagonal order parameter
            - 'psi6_local': local order parameters
        """
        r, g_r = self.radial_distribution_function(r_max=r_max)
        q, S_q = self.static_structure_factor(q_max=q_max)
        psi6_global, psi6_local = self.bond_orientational_order_psi6(cutoff=psi6_cutoff)
        
        return {
            'r': r,
            'g_r': g_r,
            'q': q,
            'S_q': S_q,
            'psi6_global': psi6_global,
            'psi6_local': psi6_local
        }


def time_averaged_structure(simulation, num_samples=50, interval=10):
    """
    Compute time-averaged structural properties from a running simulation
    
    Parameters
    ----------
    simulation : MDSimulation
        Running MD simulation object
    num_samples : int, optional
        Number of configurations to sample (default: 50)
    interval : int, optional
        MD steps between samples (default: 10)
    
    Returns
    -------
    avg_results : dict
        Time-averaged structural properties
    """
    all_g_r = []
    all_S_q = []
    all_psi6 = []
    
    for _ in range(num_samples):
        # Run simulation
        for _ in range(interval):
            simulation.run_step()
        
        # Analyze structure
        analyzer = StructureAnalyzer(simulation.positions, simulation.box_size)
        results = analyzer.compute_all_structure()
        
        all_g_r.append(results['g_r'])
        all_S_q.append(results['S_q'])
        all_psi6.append(results['psi6_global'])
    
    # Time average
    avg_results = {
        'r': results['r'],  # Same for all samples
        'q': results['q'],  # Same for all samples
        'g_r': np.mean(all_g_r, axis=0),
        'g_r_std': np.std(all_g_r, axis=0),
        'S_q': np.mean(all_S_q, axis=0),
        'S_q_std': np.std(all_S_q, axis=0),
        'psi6': np.mean(all_psi6),
        'psi6_std': np.std(all_psi6)
    }
    
    return avg_results
