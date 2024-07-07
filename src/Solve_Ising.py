## @file
#  This script performs optimization using the CIM_AHC_GPU function.
#  It reads a matrix from a file and uses it as an input for the optimization algorithm.
#
#  @brief Optimization using CIM_AHC_GPU with a provided J matrix.

import numpy as np                #< Importing the numpy library for numerical operations.
from AHC import *                 #< Importing all functions and classes from the AHC module.
from Optimise_param import *      #< Importing all functions and classes from the Optimise_param module.
import sys                        #< Importing the sys module for accessing system-specific parameters.

## Read the input file.
#  The path of the file is taken from the command line argument. The file is expected to contain
#  a numpy array which represents the J matrix for the optimization process.
#
#  @param file_path Command line argument that specifies the input file path.
file_path = sys.argv[1]
J = -np.load(file_path)           #< Load J matrix from the file and negate it.

eps_0 = 0.2                       #< Set the initial epsilon value.
r_0 = 0.07                        #< Set the initial r value.

## Perform optimization using CIM_AHC_GPU.
#  The function CIM_AHC_GPU is called with specified parameters. The results of the optimization
#  are stored in the variable 'results'.
#
#  @param T_time Total time for the CIM optimization process.
#  @param J J matrix used in the optimization.
#  @param batch_size The size of batches used in the optimization.
#  @param time_step Time step size for the optimization.
#  @param custom_fb_schedule Custom feedback schedule (if any).
#  @param custom_pump_schedule Custom pump schedule (if any).
results = CIM_AHC_GPU(T_time=20, 
                      J=J, 
                      batch_size=1, 
                      time_step=0.01, 
                      custom_fb_schedule=None, 
                      custom_pump_schedule=None)
spin_config, x_trajectory, t, energy_plot_data, error_var_data, divg, kappa = results

energy_trace = energy_plot_data[0]
min_energy = np.min(energy_trace)
# Convert the energy to the max-cut energy.
maxcut_energy = calculate_maxcut_energy(min_energy, J)
print(f"Ising ground state energy: {min_energy}")
print(f"Max-Cut Energy: {maxcut_energy}")
## Plot the results.
#  Calls the function plot_results to visualize the outcomes of the optimization process.
#
#  @param results The results obtained from the CIM_AHC_GPU function.
plot_results(results)
