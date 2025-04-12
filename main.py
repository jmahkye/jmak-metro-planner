import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random


class Station:
    def __init__(self, name, capacity, direction_bias, position):
        self.name = name
        self.capacity = capacity  # Maximum number of passengers the station can hold
        self.direction_bias = direction_bias  # Between -1 and +1
        self.waiting_passengers = 0  # Current number of waiting passengers
        self.position = position  # Position on the line (x, y coordinates)

    def generate_passengers(self):
        # Generate random number of new passengers based on capacity and some randomness
        new_passengers = np.random.poisson(self.capacity * 0.1)  # Using Poisson distribution
        self.waiting_passengers += new_passengers
        return new_passengers

    def get_boarding_direction(self):
        # Based on direction_bias, determine how many passengers board vs. alight
        # Return a value that tells what percentage of passengers want to board
        return (self.direction_bias + 1) / 2  # Convert from [-1,1] to [0,1]


class Train:
    def __init__(self, capacity, current_station=None, direction=1):
        self.capacity = capacity
        self.passengers = 0
        self.current_station = current_station
        self.direction = direction  # 1 for forward, -1 for backward along the line

    def arrive_at_station(self, station):
        # Handle passengers getting off
        departing = int(self.passengers * (1 - station.get_boarding_direction()))
        self.passengers -= departing

        # Handle passengers getting on
        can_board = min(station.waiting_passengers, self.capacity - self.passengers)
        self.passengers += can_board
        station.waiting_passengers -= can_board

        return departing, can_board


class MetroLine:
    def __init__(self, name):
        self.name = name
        self.stations = []
        self.trains = []

    def add_station(self, station):
        self.stations.append(station)

    def add_train(self, train):
        self.trains.append(train)

    def simulate_step(self):
        # Move trains along the line
        for train in self.trains:
            # Find current station index
            if train.current_station is not None:
                current_idx = self.stations.index(train.current_station)
                next_idx = current_idx + train.direction

                # Check if we reached the end of the line
                if next_idx < 0 or next_idx >= len(self.stations):
                    train.direction *= -1  # Reverse direction
                    next_idx = current_idx + train.direction

                train.current_station = self.stations[next_idx]
                departing, boarding = train.arrive_at_station(train.current_station)
                print(
                    f"Train at {train.current_station.name}: {departing} departed, {boarding} boarded. Now carrying {train.passengers} passengers.")

        # Generate new passengers at each station
        for station in self.stations:
            new_passengers = station.generate_passengers()
            print(f"{new_passengers} new passengers at {station.name}. Now waiting: {station.waiting_passengers}")


class MetroSystem:
    def __init__(self):
        self.lines = {}

    def add_line(self, line):
        self.lines[line.name] = line

    def simulate(self, steps):
        for i in range(steps):
            print(f"\n--- Simulation Step {i + 1} ---")
            for line_name, line in self.lines.items():
                print(f"\nLine: {line_name}")
                line.simulate_step()


# Using machine learning to optimize the system (simplified example)
def optimize_train_schedules(metro_system, days=30):
    """
    Example of how you might use ML to optimize train scheduling
    """
    # In a real system, you would:
    # 1. Collect passenger flow data from simulation
    # 2. Train a model to predict passenger demand
    # 3. Use reinforcement learning to optimize train schedules

    # Placeholder for ML optimization
    print("Optimizing train schedules using machine learning...")

    # Example: Initialize a basic Q-learning system (highly simplified)
    q_table = {}

    # Simulate multiple days to learn optimal scheduling
    for day in range(days):
        # Run simulation and collect data
        metro_system.simulate(24)  # 24 steps per day

        # Update Q-values based on rewards (minimizing wait times, maximizing efficiency)
        # This is just a placeholder - you'd need a proper RL implementation

    print("Optimization complete.")

    return q_table  # Return optimized parameters


# Create a simple system with one line
def create_test_system():
    # Create a metro system
    system = MetroSystem()

    # Create a line
    line1 = MetroLine("Red Line")

    # Create stations with name, capacity, direction_bias, position
    stations = [
        Station("Terminal", 200, 0.8, (0, 0)),  # Mostly boarding
        Station("Downtown", 500, 0.2, (5, 0)),  # Mixed
        Station("Midtown", 300, 0.0, (10, 0)),  # Equal boarding/alighting
        Station("Uptown", 250, -0.3, (15, 0)),  # More alighting
        Station("Airport", 400, -0.9, (20, 0))  # Mostly alighting
    ]

    # Add stations to line
    for station in stations:
        line1.add_station(station)

    # Add trains
    line1.add_train(Train(capacity=200, current_station=stations[0], direction=1))
    line1.add_train(Train(capacity=200, current_station=stations[2], direction=-1))

    # Add line to system
    system.add_line(line1)

    return system


# Create and run a test system
test_system = create_test_system()
test_system.simulate(10)  # Simulate 10 time steps

# Later, you could try to optimize the system
optimized_parameters = optimize_train_schedules(test_system, days=5)