# Advanced Collision Avoidance using Model Predictive Control (MPC)

## Overview

This project implements an advanced collision avoidance system utilizing Model Predictive Control (MPC). The system is designed to enable autonomous vehicles to navigate safely by predicting and avoiding potential obstacles in real-time.

## Features

- **Model Predictive Control (MPC):** Employs real-time optimization for effective collision avoidance.
- **Simulation Environment:** Provides a framework for testing and visualizing the vehicle's behavior.
- **Modular Architecture:** Designed for easy integration and extension.

## Requirements

Ensure the following dependencies are installed:

- Python 3.7
- Required Python libraries (install using `pip install -r requirements.txt`):
- Carla 0.9.15

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/IslamEssam01/Advanced-Collision-Avoidance-using-MPC.git
   cd Advanced-Collision-Avoidance-using-MPC
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main script:**

   ```bash
   python Main.py
   ```

## Usage

1. **Configure Parameters:**
   - Modify parameters in the `MPC.py` and `TrajectoryTracker.py` files to adjust vehicle dynamics and control settings.

2. **Execute Simulation:**
   - Run `Main.py` to start the simulation.

3. **Visualize Results:**
   - Utilize `Plotting.py` to generate visual representations of the vehicle's trajectory and performance.

## Project Structure

```
Advanced-Collision-Avoidance-using-MPC/
├── MPC.py                 # MPC controller implementation
├── Main.py                # Main script to execute the simulation
├── Plotting.py            # Visualization tools for simulation results
├── Recording.py           # Utilities for recording simulation data
├── TrajectoryTracker.py   # Trajectory tracking algorithms
└── requirements.txt       # List of project dependencies
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
