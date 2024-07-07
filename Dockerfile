# Use the base image with Miniconda
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /usr/IsingSolver

# Copy the current directory contents into the container
COPY . .

# Create the environment using the environment.yml file
RUN conda env create -n ising-solver -f environment.yml

# Make RUN commands use the ising-solver environment
SHELL ["conda", "run", "-n", "ising-solver", "/bin/bash", "-c"]

# The code to run when the container is started
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ising-solver", "python", "src/Solve_Ising.py", "MC_Instances/MC50_N=50_1.npz"]
