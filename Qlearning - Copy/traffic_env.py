import traci
import os
import sys
import numpy as np
import time
import random

# Define edge and phase mappings
INCOMING_EDGES = ['2to1', '3to1', '5to1', '4to1']  # E, N, S, W
OUTGOING_EDGES = ['1to2', '1to3', '1to5', '1to4']  # E, N, S, W
PHASES = [0, 2, 4, 6]  # Main green phases (assuming 4-phase operation)

class TrafficEnv:
    def __init__(self, sumo_config, max_steps=3600, gui=False):
        self.sumo_config = sumo_config
        self.max_steps = max_steps
        self.current_phase = 0
        self.steps = 0
        self.gui = gui
        self.min_phase_duration = 15  # Minimum 15 seconds per phase
        self.max_phase_duration = 60   # Maximum 60 seconds per phase
        self.current_phase_duration = 0
        
        # Enhanced SUMO command
        self.sumo_cmd = [
            "sumo-gui" if gui else "sumo", 
            "-c", sumo_config,
            "--no-warnings",
            "--start",
            "--quit-on-end",
            "--step-length", "1.0",
            "--time-to-teleport", "300",
            "--random"
        ]
        self.connection_active = False
        
        # Performance tracking
        self.initial_vehicles = 0
        self.vehicles_entered = 0
        self.vehicles_exited = 0
        
    def reset(self):
        """Reset environment with proper initialization"""
        try:
            if self.connection_active:
                traci.close()
                time.sleep(0.5)
            
            # Add random seed for varied scenarios
            seed = random.randint(1, 10000)
            cmd = self.sumo_cmd + ["--seed", str(seed)]
            
            traci.start(cmd)
            self.connection_active = True
            
            # Reset counters
            self.current_phase = 0
            self.steps = 0
            self.current_phase_duration = 0
            self.vehicles_entered = 0
            self.vehicles_exited = 0
            
            # Initialize traffic light
            traci.trafficlight.setPhase("1", PHASES[self.current_phase])
            
            # Run initial steps to generate traffic
            for _ in range(50):
                traci.simulationStep()
                self.steps += 1
                self.current_phase_duration += 1
            
            # Record initial state
            self.initial_vehicles = self.get_total_vehicles()
            
            return self.get_state()
            
        except Exception as e:
            print(f"Error in reset: {e}")
            self.connection_active = False
            return np.zeros(8)  # Return 8-dimensional state

    def step(self, action):
        """Execute action with improved reward calculation"""
        try:
            if not self.connection_active:
                return np.zeros(8), -1000, True
            
            # Store metrics before action
            old_metrics = self.get_traffic_metrics()
            old_total_waiting = sum([traci.vehicle.getWaitingTime(v) for v in traci.vehicle.getIDList()])
            
            # Action: 0 = keep current phase, 1 = switch to next phase
            phase_switched = False
            
            if action == 1 and self.current_phase_duration >= self.min_phase_duration:
                self._switch_phase()
                phase_switched = True
                self.current_phase_duration = 0
            
            # Simulate for 30 seconds (realistic decision interval)
            steps_to_run = 30
            for step in range(steps_to_run):
                if self.steps >= self.max_steps:
                    break
                    
                traci.simulationStep()
                self.steps += 1
                self.current_phase_duration += 1
                
                # Force phase switch if too long
                if self.current_phase_duration >= self.max_phase_duration:
                    self._switch_phase()
                    self.current_phase_duration = 0
            
            # Calculate new metrics and reward
            new_metrics = self.get_traffic_metrics()
            new_total_waiting = sum([traci.vehicle.getWaitingTime(v) for v in traci.vehicle.getIDList()])
            
            reward = self._calculate_reward(old_metrics, new_metrics, 
                                          old_total_waiting, new_total_waiting, 
                                          phase_switched)
            
            # Check if done
            done = (self.steps >= self.max_steps) or (new_metrics['total_vehicles'] == 0)
            
            return self.get_state(), reward, done
            
        except Exception as e:
            print(f"Error in step: {e}")
            return np.zeros(8), -1000, True

    def _switch_phase(self):
        """Switch to next phase with proper yellow/red timing"""
        try:
            # Yellow phase (3 seconds)
            yellow_phase = PHASES[self.current_phase] + 1
            traci.trafficlight.setPhase("1", yellow_phase)
            for _ in range(3):
                if self.steps < self.max_steps:
                    traci.simulationStep()
                    self.steps += 1
            
            # Switch to next phase
            self.current_phase = (self.current_phase + 1) % len(PHASES)
            
            # Green phase
            traci.trafficlight.setPhase("1", PHASES[self.current_phase])
            
        except Exception as e:
            print(f"Error switching phase: {e}")

    def get_state(self):
        """Get 8-dimensional state representation"""
        try:
            if not self.connection_active:
                return np.zeros(8)
            
            state = []
            
            # Queue lengths for each incoming edge (4 values)
            for edge in INCOMING_EDGES:
                try:
                    vehicles = traci.edge.getLastStepVehicleIDs(edge)
                    queue_length = sum(1 for v in vehicles if traci.vehicle.getSpeed(v) < 1.0)
                    state.append(min(queue_length, 20))  # Cap at 20
                except:
                    state.append(0)
            
            # Vehicle densities for each incoming edge (4 values)
            for edge in INCOMING_EDGES:
                try:
                    vehicle_count = traci.edge.getLastStepVehicleNumber(edge)
                    edge_length = traci.lane.getLength(edge + "_0")  # Assuming single lane
                    density = vehicle_count / max(edge_length / 100, 1)  # Vehicles per 100m
                    state.append(min(density, 5.0))  # Cap density
                except:
                    state.append(0)
            
            return np.array(state)
            
        except Exception as e:
            print(f"Error getting state: {e}")
            return np.zeros(8)

    def get_traffic_metrics(self):
        """Get comprehensive traffic metrics"""
        try:
            if not self.connection_active:
                return self._empty_metrics()
            
            all_vehicles = traci.vehicle.getIDList()
            total_vehicles = len(all_vehicles)
            
            if total_vehicles == 0:
                return self._empty_metrics()
            
            # Calculate metrics
            total_waiting_time = sum(traci.vehicle.getWaitingTime(v) for v in all_vehicles)
            total_speed = sum(traci.vehicle.getSpeed(v) for v in all_vehicles)
            avg_speed = total_speed / total_vehicles
            avg_waiting = total_waiting_time / total_vehicles
            
            # Count vehicles in queues
            queue_lengths = []
            for edge in INCOMING_EDGES:
                try:
                    vehicles = traci.edge.getLastStepVehicleIDs(edge)
                    queue = sum(1 for v in vehicles if traci.vehicle.getSpeed(v) < 1.0)
                    queue_lengths.append(queue)
                except:
                    queue_lengths.append(0)
            
            # Calculate throughput (vehicles that have exited)
            current_total = total_vehicles
            if hasattr(self, 'previous_total_vehicles'):
                throughput = max(0, self.previous_total_vehicles - current_total)
                self.vehicles_exited += throughput
            self.previous_total_vehicles = current_total
            
            return {
                'total_vehicles': total_vehicles,
                'avg_waiting_time': avg_waiting,
                'avg_speed': avg_speed,
                'total_waiting_time': total_waiting_time,
                'queue_lengths': queue_lengths,
                'total_queue': sum(queue_lengths),
                'throughput': getattr(self, 'vehicles_exited', 0)
            }
            
        except Exception as e:
            print(f"Error in metrics: {e}")
            return self._empty_metrics()

    def _empty_metrics(self):
        """Return empty metrics"""
        return {
            'total_vehicles': 0,
            'avg_waiting_time': 0,
            'avg_speed': 0,
            'total_waiting_time': 0,
            'queue_lengths': [0] * 4,
            'total_queue': 0,
            'throughput': 0
        }

    def _calculate_reward(self, old_metrics, new_metrics, old_waiting, new_waiting, phase_switched):
        """Calculate reward based on traffic improvement"""
        try:
            reward = 0
            
            # 1. Queue length improvement (most important)
            queue_improvement = old_metrics['total_queue'] - new_metrics['total_queue']
            reward += queue_improvement * 10
            
            # 2. Waiting time improvement
            waiting_improvement = old_waiting - new_waiting
            reward += waiting_improvement * 0.1
            
            # 3. Speed improvement
            speed_improvement = new_metrics['avg_speed'] - old_metrics['avg_speed']
            reward += speed_improvement * 5
            
            # 4. Throughput bonus
            if new_metrics['throughput'] > old_metrics['throughput']:
                reward += (new_metrics['throughput'] - old_metrics['throughput']) * 20
            
            # 5. Penalty for excessive phase switching
            if phase_switched:
                reward -= 5
            
            # 6. Penalty for long queues
            if new_metrics['total_queue'] > 15:
                reward -= (new_metrics['total_queue'] - 15) * 2
            
            # 7. Bonus for balanced traffic
            queue_balance = np.std(new_metrics['queue_lengths'])
            reward -= queue_balance * 2
            
            # 8. Penalty for very low speeds (traffic jam)
            if new_metrics['avg_speed'] < 2.0:
                reward -= 20
            
            # Cap reward to prevent extreme values
            return max(min(reward, 100), -100)
            
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return -50

    def get_total_vehicles(self):
        """Get total vehicles in simulation"""
        try:
            if self.connection_active:
                return len(traci.vehicle.getIDList())
            return 0
        except:
            return 0

    def get_total_waiting_time(self):
        """Return total waiting time of all vehicles in the simulation"""
        try:
            if self.connection_active:
                return sum(traci.vehicle.getWaitingTime(v) for v in traci.vehicle.getIDList())
            return 0.0
        except:
            return 0.0

    def close(self):
        """Close SUMO simulation"""
        try:
            if self.connection_active:
                traci.close()
                self.connection_active = False
                time.sleep(0.5)
        except Exception as e:
            print(f"Error closing: {e}")
            self.connection_active = False