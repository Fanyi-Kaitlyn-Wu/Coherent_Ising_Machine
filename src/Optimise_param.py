"""!@file cim_optimization.py
@brief Module for optimizing Coherent Ising Machine (CIM) parameters.

@details This module contains functions and classes for optimizing the parameters of a Coherent Ising Machine (CIM) 
with Adaptive Heuristic Correction (AHC). It includes:

- A grid search optimization function that explores different combinations of epsilon_0 and r_0 parameters.
- A visualization of the optimization results using contour plots.
- An implementation of the M-LOOP (Machine Learning Online Optimization) interface for CIM parameter optimization.
- A function to run M-LOOP optimization on CIM parameters.

The module uses numpy for numerical computations, matplotlib for visualization, tqdm for progress bars, 
and the M-LOOP library for advanced optimization techniques.

@note This module assumes the existence of a separate 'AHC' module containing the CIM_AHC_GPU function 
and related utilities.

@author F.Wu
@date 30/June/2024
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mloop.interfaces as mli
import mloop.controllers as mlc
from src.AHC import *

def optimize_parameters(J, eps_0_range, r_0_range):
    """
    @brief Perform grid search optimization for CIM parameters.

    @param J The Ising model coupling matrix.
    @param eps_0_range Range of epsilon_0 values to search.
    @param r_0_range Range of r_0 values to search.
    @return None, but prints best parameters and energy, and displays a contour plot.
    """
    # Parameter ranges are now passed as arguments
    best_energy = float('inf')
    best_params = {}
    energy_records = []

    for eps_0 in tqdm(eps_0_range, desc='Optimizing progress'):
        for r_0 in tqdm(r_0_range, desc='r₀ progress', leave=False):
            result = CIM_AHC_GPU(
                T_time=40,
                J=J,
                batch_size=1,
                time_step=0.01,
                custom_fb_schedule=lambda ticks, time_step: custom_fb_schedule(ticks, time_step, eps_0),
                custom_pump_schedule=lambda ticks, time_step: custom_pump_schedule(ticks, time_step, r_0)
            )
            
            final_energy = result[3].min() 
            energy_records.append((eps_0, r_0, final_energy))
            
            if final_energy < best_energy:
                best_energy = final_energy
                best_params = {'eps_0': eps_0, 'r_0': r_0}

    print("Best parameters found:", best_params)
    print("Lowest energy recorded:", best_energy)

    # Plotting results
    eps_0_vals, r_0_vals, energies = zip(*energy_records)
    eps_0_grid, r_0_grid = np.meshgrid(np.unique(eps_0_vals), np.unique(r_0_vals))
    energy_grid = np.array(energies).reshape(len(np.unique(r_0_vals)), len(np.unique(eps_0_vals)))

    plt.figure(figsize=(5, 4))
    contour = plt.contourf(eps_0_grid, r_0_grid, energy_grid, levels=50, cmap='viridis')
    plt.colorbar(contour)
    plt.xlabel('$\epsilon_0$')
    plt.ylabel('$r_0$')
    plt.scatter(best_params['eps_0'], best_params['r_0'], color='red') 
    plt.show()

# This makes the function available as a module.
if __name__ == '__main__':
    eps_0_range = np.linspace(0.05, 0.2, 16)
    r_0_range = np.linspace(0.05, 0.2, 16)
    optimize_parameters(eps_0_range, r_0_range)


class CIMInterface(mli.Interface):
    """
    @brief M-LOOP interface for CIM optimization.
    """
    def __init__(self, J_matrix):
        """
        @brief Initialize the CIMInterface.

        @param J_matrix The Ising model coupling matrix.
        """
        super(CIMInterface, self).__init__()
        self.J_matrix = J_matrix

    def get_next_cost_dict(self, params_dict):
        """
        @brief Compute the cost for given parameters.

        @param params_dict Dictionary containing the parameters to evaluate.
        @return Dictionary with the computed cost and uncertainty.
        """
        eps_0 = params_dict['params'][0]
        r_0 = params_dict['params'][1]
        result = CIM_AHC_GPU(
            T_time=40,
            J=self.J_matrix,
            batch_size=1,
            time_step=0.01,
            custom_fb_schedule=lambda ticks, time_step: custom_fb_schedule(ticks, time_step, eps_0),
            custom_pump_schedule=lambda ticks, time_step: custom_pump_schedule(ticks, time_step, r_0)
        )
        final_energy = result[3].min() 
        return {'cost': final_energy, 'uncertainty': 0.1}
    
def mloop_optimize(J_matrix, min_boundary, max_boundary,max_num_runs=100):
    """
    @brief Perform M-LOOP optimization for CIM parameters.

    @param J_matrix The Ising model coupling matrix.
    @param min_boundary List of minimum values for parameters.
    @param max_boundary List of maximum values for parameters.
    @param max_num_runs Maximum number of optimization runs.
    @return The M-LOOP controller object after optimization.
    """
    interface = CIMInterface(J_matrix)
    controller = mlc.create_controller(
        interface=interface,
        controller_type='gaussian_process',
        max_num_runs=max_num_runs,  
        num_params=2,
        param_names=['eps_0', 'r_0'],
        min_boundary=min_boundary,
        max_boundary=max_boundary
    )
    controller.optimize()
    return controller
