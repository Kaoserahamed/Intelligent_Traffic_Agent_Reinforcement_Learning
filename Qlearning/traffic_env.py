# traffic_env.py
import traci
import os
import sys
import numpy as np
import time

INCOMING_EDGES = ['2to1', '3to1', '5to1', '4to1']  # E, N, S, W
PHASES = [0, 3, 6, 9]  # Corresponding green phases
YELLOW_PHASE_OFFSET = 1
RED_PHASE_OFFSET = 2

class TrafficEnv:
    def __init__(self, sumo_config, max_steps=3600, gui=False):
        self.sumo_config = sumo_config
        self.max_steps = max_steps
        self.current_phase = 0
        self.steps = 0
        self.gui = gui
        self.sumo_cmd = ["sumo-gui" if gui else "sumo", "-c", sumo_config, "--no-warnings", "--start"]
        self.connection_active = False
    
    def reset(self):
        """Reset the environment and return initial state"""
        try:
            # Close existing connection if active
            if self.connection_active:
                traci.close()
                time.sleep(0.1)  # Small delay to ensure clean shutdown
            
            # Start new SUMO simulation
            traci.start(self.sumo_cmd)
            self.connection_active = True
            self.current_phase = 0
            self.steps = 0
            
            # Set initial traffic light phase
            traci.trafficlight.setPhase("1", PHASES[self.current_phase])
            
            # Run a few initial steps to stabilize
            for _ in range(5):
                traci.simulationStep()
                self.steps += 1
            
            return self.get_state()
            
        except Exception as e:
            print(f"Error in reset: {e}")
            self.connection_active = False
            return [0] * len(INCOMING_EDGES)
    
    def step(self, action):
        """Execute action and return next state, reward, and done flag"""
        try:
            if not self.connection_active:
                return [0] * len(INCOMING_EDGES), -100, True
            
            reward = 0
            old_metrics = self.get_traffic_metrics()
            
            # Action 0 = keep current phase, Action 1 = switch to next phase
            if action == 1:
                # Switch phase with proper yellow/red transition
                self.current_phase = (self.current_phase + 1) % len(PHASES)
                
                # Yellow phase
                traci.trafficlight.setPhase("1", PHASES[self.current_phase] + YELLOW_PHASE_OFFSET)
                for _ in range(3):
                    if self.steps < self.max_steps:
                        traci.simulationStep()
                        self.steps += 1
                
                # All-red phase
                traci.trafficlight.setPhase("1", PHASES[self.current_phase] + RED_PHASE_OFFSET)
                for _ in range(2):
                    if self.steps < self.max_steps:
                        traci.simulationStep()
                        self.steps += 1
                
                # Green phase
                traci.trafficlight.setPhase("1", PHASES[self.current_phase])
            
            # Simulate for several steps
            for _ in range(10):
                if self.steps < self.max_steps:
                    traci.simulationStep()
                    self.steps += 1
                else:
                    break
            
            # Calculate reward based on improvement in traffic flow
            new_metrics = self.get_traffic_metrics()
            
            # Multi-objective reward: reduce waiting time, queue length, and improve speed
            waiting_improvement = old_metrics['total_waiting'] - new_metrics['total_waiting']
            queue_penalty = sum(new_metrics['queue_lengths'])
            speed_bonus = new_metrics['avg_speed'] * 0.1
            
            reward = waiting_improvement * 0.1 - queue_penalty * 0.5 + speed_bonus
            
            # Check if simulation should end
            done = (self.steps >= self.max_steps) or (traci.simulation.getMinExpectedNumber() <= 0)
            
            next_state = self.get_state()
            return next_state, reward, done
            
        except Exception as e:
            print(f"Error in step: {e}")
            self.connection_active = False
            return [0] * len(INCOMING_EDGES), -100, True
    
    def get_state(self):
        """Get current state (queue lengths on incoming edges)"""
        try:
            if not self.connection_active:
                return [0] * len(INCOMING_EDGES)
            
            state = []
            for edge in INCOMING_EDGES:
                try:
                    # Get number of vehicles waiting (speed < 0.5 m/s)
                    vehicles = traci.edge.getLastStepVehicleIDs(edge)
                    waiting_vehicles = 0
                    for veh in vehicles:
                        if traci.vehicle.getSpeed(veh) < 0.5:
                            waiting_vehicles += 1
                    state.append(min(waiting_vehicles, 20))  # Cap at 20
                except:
                    state.append(0)
            
            return state
            
        except Exception as e:
            print(f"Error getting state: {e}")
            return [0] * len(INCOMING_EDGES)
    
    def get_total_vehicles(self):
        """Get total number of vehicles in the simulation"""
        try:
            if not self.connection_active:
                return 0
            
            total_vehicles = 0
            for edge in INCOMING_EDGES:
                try:
                    total_vehicles += traci.edge.getLastStepVehicleNumber(edge)
                except:
                    continue
            return total_vehicles
            
        except Exception as e:
            return 0
    
    def get_total_waiting_time(self):
        """Get total waiting time of all vehicles (for backward compatibility)"""
        metrics = self.get_traffic_metrics()
        return metrics['total_waiting']
    
    def get_traffic_metrics(self):
        """Get comprehensive traffic metrics for analysis"""
        try:
            if not self.connection_active:
                return {
                    'total_vehicles': 0,
                    'total_waiting': 0,
                    'avg_speed': 0,
                    'queue_lengths': [0] * len(INCOMING_EDGES),
                    'edge_densities': [0] * len(INCOMING_EDGES)
                }
            
            total_vehicles = 0
            total_waiting = 0
            total_speed = 0
            vehicle_count = 0
            queue_lengths = []
            edge_densities = []
            
            for edge in INCOMING_EDGES:
                try:
                    vehicles = traci.edge.getLastStepVehicleIDs(edge)
                    edge_vehicle_count = len(vehicles)
                    total_vehicles += edge_vehicle_count
                    
                    waiting_vehicles = 0
                    edge_speed_sum = 0
                    
                    for veh in vehicles:
                        speed = traci.vehicle.getSpeed(veh)
                        waiting_time = traci.vehicle.getWaitingTime(veh)
                        
                        if speed < 0.5:  # Waiting vehicle
                            waiting_vehicles += 1
                        
                        total_waiting += waiting_time
                        edge_speed_sum += speed
                        vehicle_count += 1
                    
                    queue_lengths.append(waiting_vehicles)
                    edge_densities.append(edge_vehicle_count)
                    total_speed += edge_speed_sum
                    
                except:
                    queue_lengths.append(0)
                    edge_densities.append(0)
            
            avg_speed = total_speed / max(vehicle_count, 1)
            
            return {
                'total_vehicles': total_vehicles,
                'total_waiting': total_waiting,
                'avg_speed': avg_speed,
                'queue_lengths': queue_lengths,
                'edge_densities': edge_densities
            }
            
        except Exception as e:
            return {
                'total_vehicles': 0,
                'total_waiting': 0,
                'avg_speed': 0,
                'queue_lengths': [0] * len(INCOMING_EDGES),
                'edge_densities': [0] * len(INCOMING_EDGES)
            }
    
    def close(self):
        """Close the SUMO simulation"""
        try:
            if self.connection_active:
                traci.close()
                self.connection_active = False
        except Exception as e:
            print(f"Error closing environment: {e}")
            self.connection_active = False