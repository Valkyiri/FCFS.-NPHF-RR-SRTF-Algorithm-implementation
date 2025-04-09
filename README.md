# OS Scheduler Simulator

This project implements various CPU scheduling algorithms and visualizes their performance through an interactive web interface.

## Features

- Process Generation with configurable parameters
- Implementation of 4 scheduling algorithms:
  1. Non-preemptive Highest Priority First (17.5%)
  2. First Come First Serve (17.5%)
  3. Round Robin (22.5%)
  4. Preemptive Shortest Remaining Time First (22.5%)
- Interactive web interface with Gantt charts
- Performance metrics comparison
- Real-time visualization

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your web browser and navigate to: http://localhost:5000

## Usage

1. Enter the simulation parameters:
   - Number of processes
   - Arrival time distribution parameters (mean and standard deviation)
   - Burst time distribution parameters (mean and standard deviation)
   - Priority distribution parameter (lambda for Poisson distribution)

2. Click "Generate and Schedule" to run the simulation

3. View the results:
   - Gantt charts for each scheduling algorithm
   - Performance metrics (average turnaround time and waiting time)

## Assumptions

- Higher priority number indicates higher priority
- In case of ties, processes are selected by their ID (lower ID first)
- All processes are CPU-bound (no I/O operations)
- Round Robin quantum is set to 2 time units
