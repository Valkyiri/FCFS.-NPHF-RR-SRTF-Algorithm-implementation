from flask import Flask, render_template, request, jsonify
import numpy as np
import plotly.express as px
import plotly.utils
import json
from dataclasses import dataclass
from typing import List
import heapq

# Creates a Flask application instance
app = Flask(__name__)

# Process class to represent a single process in the scheduling system
@dataclass
class Process:
    # Unique identifier for the process
    id: int
    # Time at which the process arrives in the system
    arrival_time: float
    # Total CPU time required by the process
    burst_time: float
    # Priority level of the process (higher number = higher priority)
    priority: int
    # Time remaining until process completion (initialized to burst_time)
    remaining_time: float = None
    # Time when process first starts executing
    start_time: float = None
    # Time when process completes execution
    completion_time: float = None
    
    def __post_init__(self):
        # Initialize remaining_time to burst_time if not explicitly set
        if self.remaining_time is None:
            self.remaining_time = self.burst_time

# Class responsible for generating process instances with random attributes
class ProcessGenerator:
    @staticmethod
    def generate_processes(num_processes, arrival_params, burst_params, priority_lambda):
        # Initialize empty list to store generated processes
        processes = []
        # Generate random values using normal distribution for arrival and burst times
        # arrival_params and burst_params contain (mean, standard_deviation)
        arrival_times = np.random.normal(*arrival_params, num_processes)
        burst_times = np.random.normal(*burst_params, num_processes)
        # Generate random priorities using Poisson distribution
        priorities = np.random.poisson(priority_lambda, num_processes)
        
        # Create Process instances with the generated random values
        for i in range(num_processes):
            # Ensure arrival time is not negative and burst time is at least 0.1
            processes.append(Process(
                id=i+1,  # Process IDs start from 1
                arrival_time=max(0, arrival_times[i]),  
                burst_time=max(0.1, burst_times[i]),    
                priority=max(1, priorities[i])          # Ensure priority is at least 1
            ))
        
        # Return the list of generated processes
        return processes

# Class responsible for scheduling processes using various algorithms
class Scheduler:
    @staticmethod
    # Non-Preemptive Highest Priority First (HPF) scheduling algorithm
    def non_preemptive_highest_priority_first(processes: List[Process]):
        # Initialize the current time to 0
        time = 0
        # Initialize empty list to store completed processes
        completed = []
        # Initialize empty list to store processes in the ready queue
        ready_queue = []
        # Sort all processes by their arrival time
        processes = sorted(processes, key=lambda x: x.arrival_time)
        
        # Continue scheduling until all processes have completed (Like a while-condition to check if everything is completed)
        while processes or ready_queue:
            # Add processes that have arrived to the ready queue (Now starts by adding processes to ready queue)
            while processes and processes[0].arrival_time <= time:
                # Use a heap to store processes in the ready queue, sorted by priority (Heap like stack in datastructure in c++)
                heapq.heappush(ready_queue, (-processes[0].priority, processes[0].id, processes[0]))
                # Remove the first process from the list since it's now in the ready queue (Pops processes that are now in ready queue)
                processes.pop(0)
                
            # If there are no processes in the ready queue, we need to jump time forward (Condition to check for processes in ready queue, if not, continue)
            if not ready_queue:
                # Jump time forward to when the next process arrives
                time = processes[0].arrival_time
                # Skip the rest of this loop iteration
                continue
                
            # Get the highest priority process from the ready queue
            # The _ variables are used to ignore the priority and id values from the heap (Used extra knowledge because we never took heap)
            _, _, current_process = heapq.heappop(ready_queue)
            # Record when this process starts executing
            current_process.start_time = time
            # Calculate when this process will finish (current time + how long it needs to run)
            current_process.completion_time = time + current_process.burst_time
            # Move time forward to when this process completes
            time = current_process.completion_time
            # Add this process to our list of completed processes
            completed.append(current_process)
            
        # Return all processes that have been completed
        return completed
        '''
        TEST CODE FOR NPHF
        processes = [("P1", 1, 2, 5), ("P2", 2, 3, 4), ("P3", 3, 4, 3)]
        processes.sort(key=lambda x: x[1])
        time = 0
        ready_queue = []
        completed = []
        while processes or ready_queue:
            while processes and processes[0][1] <= time:
                ready_queue.append(processes.pop(0))
            if ready_queue:
                ready_queue.sort(key=lambda x: x[3])
                current_process = ready_queue.pop(0)
                pid, arrival, burst, priority = current_process
                start_time = time
                end_time = start_time + burst
                completed.append((pid, start_time, end_time))
                time = end_time
            else:
                time+=1
        for p in completed:
            print(f"{p[0]}: start={p[1]}, end={p[2]}")
        '''

    @staticmethod
    # First-Come-First-Served (FCFS) scheduling algorithm
    def fcfs(processes: List[Process]):
        # Initialize the current time to 0
        time = 0
        # Initialize empty list to store completed processes
        completed = []
        # Sort all processes by their arrival time to ensure FCFS order
        processes = sorted(processes, key=lambda x: x.arrival_time)
        
        # Process each task in arrival order
        for process in processes:
            # Set time to either current time or when the process arrives, whichever is later
            time = max(time, process.arrival_time)
            # Record when this process starts executing
            process.start_time = time
            # Calculate when this process will finish
            process.completion_time = time + process.burst_time
            # Move time forward to when this process completes
            time = process.completion_time
            # Add this process to our list of completed processes
            completed.append(process)
            
        # Return all processes that have been completed
        return completed
        '''
        TEST FCFS code:
        processes = [("P1", 0, 5), ("P2", 1, 3), ("P3", 2, 4)]
        processes.sort(key=lambda x: x[1])
        start_time = 0
        for PID, arrival_time, burst_time in processes:
            if start_time < arrival_time:
                start_time = arrival_time
            finish_time = start_time + burst_time
            print(f"{PID}: start={start_time}, end={finish_time}")
            start_time = finish_time
        '''

    @staticmethod
    # Round Robin (RR) scheduling algorithm
    def round_robin(processes: List[Process], quantum=2):
        # Initialize the current time to 0
        time = 0
        # Initialize empty list to store completed processes
        completed = []
        # Initialize empty list to store processes in the ready queue
        ready_queue = []
        # Sort all processes by their arrival time
        processes = sorted(processes, key=lambda x: x.arrival_time)
        # Create a copy of the processes list to avoid modifying the original list
        current_processes = [p for p in processes]
        
        # Continue scheduling until all processes have completed
        while current_processes or ready_queue:
            # Add processes that have arrived to the ready queue
            while current_processes and current_processes[0].arrival_time <= time:
                ready_queue.append(current_processes.pop(0))
                
            # If the ready queue is empty, move to the next time slot
            if not ready_queue:
                time = current_processes[0].arrival_time
                continue
                
            # Select the next process from the ready queue
            current_process = ready_queue.pop(0)
            
            # If this is the first time the process is executed, set its start time
            if current_process.start_time is None:
                current_process.start_time = time
                
            # Execute the process for the given quantum
            execution_time = min(quantum, current_process.remaining_time)
            current_process.remaining_time -= execution_time
            time += execution_time
            
            # If the process has not completed, add it back to the ready queue
            if current_process.remaining_time > 0:
                while current_processes and current_processes[0].arrival_time <= time:
                    ready_queue.append(current_processes.pop(0))
                ready_queue.append(current_process)
            else:
                # If the process has completed, set its completion time and add it to the completed list
                current_process.completion_time = time
                completed.append(current_process)
                
        # Return all processes that have been completed
        return completed
        '''
        TEST FOR RR CODE:
        processes = [("P1", 1, 2, 5), ("P2", 2, 3, 4), ("P3", 3, 4, 3)]
        processes.sort(key=lambda x: x[1])
        time = 0
        ready_queue = []
        completed = []
        time_quantam = 2
        start_time = 0
        while processes or ready_queue:
            while processes and processes[0][1] <= time:
                ready_queue.append(processes.pop(0))
                if ready_queue:
                    current_process = ready_queue.pop(0)
                    PID, arrival_time, burst_time, priority = current_process
                    if burst_time > time_quantam:
                        remaining_burst_time = burst_time - time_quantam
                        ready_queue.append((PID, arrival_time, remaining_burst_time, priority))
                        burst_time = time_quantam
                    completed.append((PID, arrival_time, burst_time))
                    time += burst_time
                else:
                    time += 1
        for p in completed:
            print(f"{p[0]} runs from {p[1]} to {p[2]}")
        '''

    @staticmethod
    # Preemptive Shortest Remaining Time First (SRTF) scheduling algorithm
    def preemptive_srtf(processes: List[Process]):
        # Initialize the current time to 0
        time = 0
        # Initialize empty list to store completed processes
        completed = []
        # Initialize empty list to store processes in the ready queue
        ready_queue = []
        # Sort all processes by their arrival time
        processes = sorted(processes, key=lambda x: x.arrival_time)
        # Create a copy of the processes list to avoid modifying the original list
        current_processes = [p for p in processes]
        
        # Continue scheduling until all processes have completed
        while current_processes or ready_queue:
            # Add processes that have arrived to the ready queue
            while current_processes and current_processes[0].arrival_time <= time:
                # Use a heap to store processes in the ready queue, sorted by remaining time
                heapq.heappush(ready_queue, (current_processes[0].remaining_time, current_processes[0].id, current_processes[0]))
                current_processes.pop(0)
                
            # If the ready queue is empty, move to the next time slot
            if not ready_queue:
                time = current_processes[0].arrival_time
                continue
                
            # Select the process with the shortest remaining time from the ready queue
            remaining_time, _, current_process = heapq.heappop(ready_queue)
            
            # If this is the first time the process is executed, set its start time
            if current_process.start_time is None:
                current_process.start_time = time
                
            # Calculate the time until the next process arrives
            next_arrival = float('inf')
            if current_processes:
                next_arrival = current_processes[0].arrival_time
                
            # Execute the process until it completes or the next process arrives
            execution_time = min(current_process.remaining_time, next_arrival - time)
            current_process.remaining_time -= execution_time
            time += execution_time
            
            # If the process has not completed, add it back to the ready queue
            if current_process.remaining_time > 0:
                heapq.heappush(ready_queue, (current_process.remaining_time, current_process.id, current_process))
            else:
                # If the process has completed, set its completion time and add it to the completed list
                current_process.completion_time = time
                completed.append(current_process)
                
        # Return all processes that have been completed
        return completed
        '''
        TEST FOR SRTF CODE:
processes = [("P1", 1, 2, 5), ("P2", 2, 3, 4), ("P3", 3, 4, 3)]
processes.sort(key=lambda x: x[1])
time = 0
ready_queue = []
completed = []
while processes or ready_queue:
    while processes and processes[0][1] <= time:
        ready_queue.append(processes.pop(0))
    if ready_queue:
        ready_queue.sort(key=lambda x: x[2])
        current_process = ready_queue.pop(0)
        pid, arrival_time, burst_time, priority = current_process
        burst_time -= 1
        time += 1
        if burst_time == 0:
            completed.append((pid, arrival_time, time))
        else:
            ready_queue.append((pid, arrival_time, burst_time, priority))
    else:
        time += 1
for p in completed:
    print(f"{p[0]} runs from {p[1]} to {p[2]}")
        '''

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for generating process instances and scheduling them using various algorithms
@app.route('/generate', methods=['POST'])
def generate():
    # Get the JSON data from the request
    data = request.json
    # Extract the number of processes, arrival parameters, burst parameters, and priority lambda from the data
    num_processes = int(data['num_processes'])
    arrival_mean, arrival_std = map(float, data['arrival_params'].split())
    burst_mean, burst_std = map(float, data['burst_params'].split())
    priority_lambda = float(data['priority_lambda'])
    
    # Generate process instances with random attributes
    generator = ProcessGenerator()
    processes = generator.generate_processes(
        num_processes,
        (arrival_mean, arrival_std),
        (burst_mean, burst_std),
        priority_lambda
    )
    
    # Schedule the processes using various algorithms
    scheduler = Scheduler()
    results = {}
    metrics = {}
    
    # Iterate over the scheduling algorithms
    for algo in ['HPF', 'FCFS', 'RR', 'SRTF']:
        # Create a copy of the processes list to avoid modifying the original list
        processes_copy = [Process(p.id, p.arrival_time, p.burst_time, p.priority) for p in processes]
        
        # Schedule the processes using the current algorithm
        if algo == 'HPF':
            completed = scheduler.non_preemptive_highest_priority_first(processes_copy)
        elif algo == 'FCFS':
            completed = scheduler.fcfs(processes_copy)
        elif algo == 'RR':
            completed = scheduler.round_robin(processes_copy)
        else:  # SRTF
            completed = scheduler.preemptive_srtf(processes_copy)
            
        # Calculate metrics for the scheduled processes
        turnaround_times = []
        waiting_times = []
        
        # Iterate over the completed processes
        for p in completed:
            # Calculate the turnaround time and waiting time for the process
            turnaround_time = p.completion_time - p.arrival_time
            waiting_time = turnaround_time - p.burst_time
            turnaround_times.append(turnaround_time)
            waiting_times.append(waiting_time)
            
        # Calculate the average turnaround time and waiting time
        metrics[algo] = {
            'avg_turnaround': sum(turnaround_times) / len(turnaround_times),
            'avg_waiting': sum(waiting_times) / len(waiting_times)
        }
        
        # Prepare Gantt chart data for the scheduled processes
        gantt_data = []
        # Iterate over the completed processes
        for p in completed:
            # Create a dictionary to represent the process in the Gantt chart
            gantt_data.append({
                'Process': f'P{p.id}',
                'Start': p.start_time,
                'Finish': p.completion_time,
                'Duration': p.completion_time - p.start_time
            })
            
        # Store the Gantt chart data in the results dictionary
        results[algo] = gantt_data
    
    # Return the results and metrics as JSON
    return jsonify({'results': results, 'metrics': metrics})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
