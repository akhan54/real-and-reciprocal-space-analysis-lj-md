"""
Core Molecular Dynamics Simulation Engine
==========================================

This module implements a 2D Lennard-Jones molecular dynamics simulator
with periodic boundary conditions, Verlet integration, and cell-list
optimization for efficient force calculations.

Physical System:
- Lennard-Jones potential: U(r) = 4ε[(σ/r)^12 - (σ/r)^6]
- Reduced units: σ = 1.0, mass m = 1.0, time unit τ = √(mσ²/ε)
- Temperature control via velocity rescaling (Berendsen-like thermostat)

Author: Research-grade MD implementation for soft matter studies
"""

import numpy as np


class MDSimulation:
    """
    2D Lennard-Jones Molecular Dynamics Simulation Engine
    
    Parameters
    ----------
    num_particles : int
        Number of particles in the system
    box_size : float
        Side length of the square periodic box
    temperature : float
        Target temperature in reduced units (k_B T / ε)
    epsilon : float, optional
        LJ energy parameter (default: 1.0)
    sigma : float, optional
        LJ length parameter (default: 1.0)
    dt : float, optional
        Time step for integration (default: 0.005)
    seed : int or None, optional
        Random seed for reproducibility
    """
    
    def __init__(self, num_particles, box_size, temperature, 
                 epsilon=1.0, sigma=1.0, dt=0.005, seed=None):
        self.num_particles = num_particles
        self.box_size = box_size
        self.temperature = temperature
        self.epsilon = epsilon
        self.sigma = sigma
        self.dt = dt
        
        # Random number generator for reproducibility
        self.rng = np.random.default_rng(seed)
        
        # Initialize positions on a grid to avoid overlaps
        self._initialize_positions()
        
        # Maxwell-Boltzmann velocity distribution
        self.velocities = self.rng.normal(0, np.sqrt(temperature), (num_particles, 2))
        
        # Remove center-of-mass motion
        self.velocities -= np.mean(self.velocities, axis=0)
        
        # Force array
        self.forces = np.zeros((num_particles, 2))
        
        # Energy tracking
        self.potential_energy = 0.0
        self.kinetic_energy = 0.0
        
        # Cell list parameters for efficient neighbor search
        self.cutoff = 3.0 * sigma  # Standard LJ cutoff
        self.cell_size = self.cutoff
        self.num_cells_x = max(1, int(self.box_size / self.cell_size))
        self.num_cells_y = max(1, int(self.box_size / self.cell_size))
        self.cell_list = [[] for _ in range(self.num_cells_x * self.num_cells_y)]
        
        # Simulation step counter
        self.step = 0
        
    def _initialize_positions(self):
        """Initialize particle positions on a square lattice"""
        # Determine grid size
        n_side = int(np.ceil(np.sqrt(self.num_particles)))
        lattice_spacing = self.box_size / n_side
        
        positions = []
        for i in range(n_side):
            for j in range(n_side):
                if len(positions) < self.num_particles:
                    x = (i + 0.5) * lattice_spacing
                    y = (j + 0.5) * lattice_spacing
                    # Add small random displacement to break symmetry
                    x += self.rng.uniform(-0.1, 0.1) * lattice_spacing
                    y += self.rng.uniform(-0.1, 0.1) * lattice_spacing
                    positions.append([x % self.box_size, y % self.box_size])
        
        self.positions = np.array(positions)
    
    def _build_cell_list(self):
        """Build cell list for efficient neighbor searching"""
        # Clear all cells
        for cell in self.cell_list:
            cell.clear()
        
        # Assign particles to cells
        for i in range(self.num_particles):
            ix = int(self.positions[i, 0] / self.cell_size) % self.num_cells_x
            iy = int(self.positions[i, 1] / self.cell_size) % self.num_cells_y
            cell_index = iy * self.num_cells_x + ix
            self.cell_list[cell_index].append(i)
    
    def calculate_forces(self):
        """
        Calculate Lennard-Jones forces using cell list method
        
        The LJ potential is:
            U(r) = 4ε[(σ/r)^12 - (σ/r)^6]
        
        The force is:
            F(r) = 24ε/r² [2(σ/r)^12 - (σ/r)^6] * r_vec
        
        Returns
        -------
        potential_energy : float
            Total potential energy of the system
        """
        self.forces.fill(0.0)
        self.potential_energy = 0.0
        self._build_cell_list()
        
        min_dist = 0.1 * self.sigma  # Prevent numerical instabilities
        
        for i in range(self.num_particles):
            # Find cell of particle i
            ix = int(self.positions[i, 0] / self.cell_size) % self.num_cells_x
            iy = int(self.positions[i, 1] / self.cell_size) % self.num_cells_y
            
            # Check neighboring cells (including self)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx = (ix + dx + self.num_cells_x) % self.num_cells_x
                    ny = (iy + dy + self.num_cells_y) % self.num_cells_y
                    neighbor_cell_index = ny * self.num_cells_x + nx
                    
                    for j in self.cell_list[neighbor_cell_index]:
                        if i < j:  # Count each pair only once
                            # Minimum image convention for periodic boundaries
                            r_vec = self.positions[i] - self.positions[j]
                            r_vec -= self.box_size * np.rint(r_vec / self.box_size)
                            
                            r = np.linalg.norm(r_vec)
                            
                            if r < min_dist:
                                r = min_dist
                            
                            if r < self.cutoff:
                                # LJ potential calculation
                                r_inv = self.sigma / r
                                r6_inv = r_inv**6
                                r12_inv = r6_inv**2
                                
                                # Potential energy
                                u_ij = 4.0 * self.epsilon * (r12_inv - r6_inv)
                                self.potential_energy += u_ij
                                
                                # Force magnitude
                                f_mag = 24.0 * self.epsilon * (2.0 * r12_inv - r6_inv) / r**2
                                f_vec = f_mag * r_vec
                                
                                # Newton's third law
                                self.forces[i] += f_vec
                                self.forces[j] -= f_vec
        
        return self.potential_energy
    
    def integrate_step(self):
        """
        Perform one MD step using velocity Verlet integration
        
        Velocity Verlet algorithm:
        1. x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
        2. Calculate a(t+dt) from new positions
        3. v(t+dt) = v(t) + 0.5*[a(t) + a(t+dt)]*dt
        """
        # Update positions
        self.positions += self.velocities * self.dt + 0.5 * self.forces * self.dt**2
        
        # Apply periodic boundary conditions
        self.positions %= self.box_size
        
        # Store old forces
        old_forces = self.forces.copy()
        
        # Calculate new forces at t+dt
        self.calculate_forces()
        
        # Update velocities
        self.velocities += 0.5 * (old_forces + self.forces) * self.dt
        
        # Calculate kinetic energy
        self.kinetic_energy = 0.5 * np.sum(self.velocities**2)
        
        self.step += 1
    
    def rescale_velocities(self):
        """
        Rescale velocities to maintain target temperature (Berendsen thermostat)
        
        In 2D, the equipartition theorem gives:
            <K.E.> = N * k_B * T
        where N is the number of particles and we have 2 degrees of freedom per particle.
        """
        current_ke = 0.5 * np.sum(self.velocities**2)
        target_ke = self.num_particles * self.temperature
        
        if current_ke > 0:
            scale_factor = np.sqrt(target_ke / current_ke)
            self.velocities *= scale_factor
            self.kinetic_energy = target_ke
    
    def run_step(self):
        """Perform one complete MD step with thermostat"""
        self.integrate_step()
        self.rescale_velocities()
    
    def get_total_energy(self):
        """Return total energy (kinetic + potential)"""
        return self.kinetic_energy + self.potential_energy
    
    def get_instantaneous_temperature(self):
        """
        Calculate instantaneous temperature from kinetic energy
        
        In 2D: T = K.E. / (N * k_B) where k_B = 1 in reduced units
        """
        return self.kinetic_energy / self.num_particles
    
    def get_density(self):
        """Return number density (particles per unit area)"""
        return self.num_particles / self.box_size**2
