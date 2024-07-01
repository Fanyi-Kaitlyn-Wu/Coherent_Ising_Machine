import torch
import numpy as np

def linear_time_schedule(initial_value, ticks, time_step):
    return initial_value * torch.arange(0, ticks * time_step, time_step)

def custom_fb_schedule(ticks, time_step, eps_0=0.07):
    return linear_time_schedule(eps_0, ticks, time_step)

def custom_pump_schedule(ticks, time_step, r_0=0.2):
    return linear_time_schedule(r_0, ticks, time_step)

def e_0(t, beta, p, a):
    integral = torch.cumsum((-1 + p - a) * t, dim=0)
    return torch.exp(-beta * integral)

def compute_divg(N, beta, p, a, eps, H_t, e_0_t):
    divg = beta * (N * (1 - p + a) + 2 * eps * e_0_t * H_t )#+ (e_0_t**2 * eps**2) / (p - 1))
    return divg

def calculate_maxcut_energy(E, J):
    """
    Calculate the Max-Cut value from the Ising model energy and interaction matrix.

    Parameters:
    E (float): Ising model energy.
    J (numpy.ndarray): Interaction matrix where J[i][j] represents the weight of the edge between nodes i and j.

    Returns:
    float: Max-Cut value.
    """
    # Ensure J is a numpy array
    J = np.array(J)
    
    # Calculate the sum of the interaction strengths
    total_interaction_strength = np.sum(J) / 2  # Sum of upper triangle (excluding diagonal)

    # Calculate the Max-Cut value
    maxcut_value = 0.5 * (total_interaction_strength + E)
    
    return maxcut_value

torch.backends.cudnn.benchmark = True

def CIM_AHC_GPU(T_time, J, batch_size=1, time_step=0.05, beta=0.05, mu=0.5, noise=0, custom_fb_schedule=None, custom_pump_schedule=None, random_number_function=None, ahc_nonlinearity=None, device=torch.device('cpu')):
    # Compute instance sizes, cast Ising problem matrix to torch tensor.
    J = torch.from_numpy(J).float().to(device)
    N = J.size(1)

    # Initialize plot arrays and runtime variables.
    end_ising_energy = (1e20 * torch.ones(batch_size)).to(device)
    target_a_baseline = 0.2
    target_a = (target_a_baseline * torch.ones(batch_size)).to(device)

    ticks = int(T_time / time_step)
    spin_amplitude_trajectory = torch.zeros(batch_size, N, ticks).to(device)
    error_var_data = torch.zeros(batch_size, N, ticks).to(device)
    energy_plot_data = torch.zeros(batch_size, ticks).to(device)
    divg_values = torch.zeros(batch_size, ticks).to(device)
    t_opt = torch.zeros(batch_size).to(device)
    EMAX_FLOAT = 32.0

    # Initialize Spin-Amplitude Vectors and Auxiliary Variables
    x = 0.001 * torch.rand(batch_size, N).to(device) - 0.0005
    error_var = torch.ones(batch_size, N).to(device).float()
    etc_flag = torch.ones(batch_size, N).to(device)
    sig_ = ((2 * (x > 0) - 1).float()).to(device)
    sig_opt = sig_

    # Configure ramp schedules, random number function.
    if random_number_function is None:
        random_number_function = lambda c: torch.rand(c, 3)

    if custom_fb_schedule is None:
        eps_schedule = torch.ones(ticks).to(device)
    else:
        eps_schedule = custom_fb_schedule(ticks, time_step).to(device)
    
    if custom_pump_schedule is None:
        r_schedule = torch.ones(ticks).to(device)
    else:
        r_schedule = custom_pump_schedule(ticks, time_step).to(device)

    if ahc_nonlinearity is None:
        ahc_nonlinearity = lambda c: torch.pow(c, 3)

    # Compute e_0 over time
    time_array = torch.arange(0, T_time, time_step).to(device)
    e_0_values = e_0(time_array, beta, r_schedule, target_a_baseline).to(device)

    # Spin evolution euler-step iteration loop.
    for t in range(ticks):
        # Update spin states and Ising energy.
        sig = ((2 * (x > 0) - 1).float())
        sig_ = sig

        # Save current Ising energy.
        curr_ising_energy = (-1 / 2 * (torch.bmm(sig.view(batch_size, 1, N), (sig @ J).view(batch_size, N, 1)))[:, :, 0]).view(batch_size)
        energy_plot_data[:, t] = curr_ising_energy
        H_t = curr_ising_energy

        # Simulate Time-Evolution of Spin Amplitudes
        spin_amplitude_trajectory[:, :, t] = x
        error_var_data[:, :, t] = error_var
        x_squared = x ** 2
        MVM = x @ J
        x += time_step * (x * ((r_schedule[t] - 1) - mu * x_squared))
        x += time_step * eps_schedule[t] * (MVM * error_var)
        x += eps_schedule[t] * noise * (torch.rand(N, device=device) - 0.5)

        # Modulate target amplitude, error variable rate of change parameters depending on Ising energy.
        delta_a = eps_schedule[t] * torch.mean((sig @ J) * sig * etc_flag, 1)
        target_a = target_a_baseline + delta_a
        x_squared = x ** 2

        # Euler step for equations of motion of error variables.
        error_var += time_step * (-beta * ((x_squared) - target_a[:, None]) * error_var)

        # Normalize auxiliary error variables.
        error_var[error_var > EMAX_FLOAT] = EMAX_FLOAT

        # Update divg using integral for e_0
        e0 = e_0_values[t]
        divg = compute_divg(N, beta, r_schedule[t], target_a, eps_schedule[t], H_t, e0)
        divg_values[:, t] = divg

        # Ensure divg is maintained above kappa
        kappa = N * beta * (1 - r_schedule[t] + target_a_baseline)

        # Use boolean-array indexing to update ramp schedules and minimum Ising energy.
        comparison = torch.any(sig_ != sig, 1)
        etc_flag[comparison, :] = error_var[comparison, :]
        t_opt[curr_ising_energy < end_ising_energy] = t
        end_ising_energy = torch.minimum(end_ising_energy, curr_ising_energy)

    # Parse and Return Solutions
    sig = ((2 * (x > 0) - 1).float())
    spin_amplitude_trajectory = spin_amplitude_trajectory.cpu()
    spin_plot_data = 2 * (spin_amplitude_trajectory > 0) - 1
    energy_plot_data = energy_plot_data.cpu()
    error_var_data = error_var_data.cpu()
    divg_plot_data = divg_values.cpu()
    for k in range(batch_size):
        sig_opt[k, :] = spin_plot_data[k, :, t_opt.long()[k]]
    sig_opt = sig_opt.cpu()

    return (sig_opt.numpy(), spin_amplitude_trajectory.numpy(), t, energy_plot_data.numpy(), error_var_data.numpy(), divg_plot_data.numpy(), kappa)