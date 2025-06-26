import traci
import os
import sys
import numpy as np
import time
import random

# Define edge and phase mappings
INCOMING_EDGES = ['2to1', '3to1', '5to1', '4to1']  # E, N, S, W
OUTGOING_EDGES = ['1to2', '1to3', '1to5', '1to4']  # E, N, S, W
PHASES = [0, 2, 4, 6]  # Main green phases (E=0, N=2, S=4, W=6)

# Define vehicle types manually based on your SUMO config
VEHICLE_TYPES = ['car', 'bus', 'emergency', 'truck', 'bike']

class TrafficEnv:
    def __init__(self, sumo_config, max_steps, gui=False):
        self.sumo_config = sumo_config
        self.max_steps = max_steps
        self.current_phase = 0
        self.steps = 0
        self.gui = gui
        self.min_phase_duration = 15
        self.max_phase_duration = 60
        self.current_phase_duration = 0

        self.sumo_cmd = [
            "sumo-gui" if gui else "sumo", 
            "-c", sumo_config,
            "--no-warnings",
            "--start",
            "--quit-on-end",
            "--step-length", "1.0",
            "--time-to-teleport", "300",
            "--random",
            "--end", str(max_steps),
        ]
        self.connection_active = False

        self.initial_vehicles = 0
        self.vehicles_entered = 0
        self.vehicles_exited = 0

    def reset(self):
        try:
            if self.connection_active:
                traci.close()
                time.sleep(0.5)

            seed = random.randint(1, 10000)
            cmd = self.sumo_cmd + ["--seed", str(seed)]

            traci.start(cmd)
            self.connection_active = True

            self.current_phase = 0
            self.steps = 0
            self.current_phase_duration = 0
            self.vehicles_entered = 0
            self.vehicles_exited = 0

            traci.trafficlight.setPhase("1", PHASES[self.current_phase])

            for _ in range(20):  # Warm-up
                traci.simulationStep()
                self.steps += 1
                self.current_phase_duration += 1

            self._spawn_random_vehicles(num=random.randint(10, 50))

            self.initial_vehicles = self.get_total_vehicles()
            return self.get_state()

        except Exception as e:
            print(f"Error in reset: {e}")
            self.connection_active = False
            return np.zeros(len(VEHICLE_TYPES) * len(INCOMING_EDGES) + 8)
        
    def _spawn_random_vehicles(self, num=30):
        for i in range(num):
            veh_id = f"veh_{i}_{random.randint(0, 9999)}"
            from_edge = random.choice(INCOMING_EDGES)
            to_edge = random.choice([e for e in OUTGOING_EDGES if e != from_edge])
            veh_type = random.choice(VEHICLE_TYPES)
            route_id = f"route_{veh_id}"
            try:
                traci.route.add(route_id, [from_edge, to_edge])
                traci.vehicle.add(veh_id, route_id, typeID=veh_type)
                traci.vehicle.setSpeed(veh_id, random.uniform(0, 10))
            except Exception as e:
                print(f"⚠️ Could not spawn {veh_id}: {e}")

    def step(self, action):
        try:
            if not self.connection_active:
                return np.zeros(len(VEHICLE_TYPES) * len(INCOMING_EDGES) + 8), -1000, True

            old_metrics = self.get_traffic_metrics()
            old_total_waiting = sum([traci.vehicle.getWaitingTime(v) for v in traci.vehicle.getIDList()])

            # Force switch to specific direction
            if action in range(4) and action != self.current_phase:
                self._set_phase(action)
                self.current_phase_duration = 0

            steps_to_run = 30
            for step in range(steps_to_run):
                if self.steps >= self.max_steps:
                    break

                traci.simulationStep()
                self.steps += 1
                self.current_phase_duration += 1

            new_metrics = self.get_traffic_metrics()
            new_total_waiting = sum([traci.vehicle.getWaitingTime(v) for v in traci.vehicle.getIDList()])

            reward = self._calculate_reward(old_metrics, new_metrics, old_total_waiting, new_total_waiting, True)

            done = (self.steps >= self.max_steps)

            return self.get_state(), reward, done

        except Exception as e:
            print(f"Error in step: {e}")
            return np.zeros(len(VEHICLE_TYPES) * len(INCOMING_EDGES) + 8), -1000, True

    def _set_phase(self, direction_index):
        try:
            yellow_phase = PHASES[self.current_phase] + 1
            traci.trafficlight.setPhase("1", yellow_phase)
            for _ in range(3):
                if self.steps < self.max_steps:
                    traci.simulationStep()
                    self.steps += 1
            self.current_phase = direction_index
            traci.trafficlight.setPhase("1", PHASES[self.current_phase])
        except Exception as e:
            print(f"Error setting phase: {e}")

    def get_state(self):
        try:
            if not self.connection_active:
                return np.zeros(len(VEHICLE_TYPES) * len(INCOMING_EDGES) + 8)

            state = []

            for edge in INCOMING_EDGES:
                try:
                    counts = {vt: 0 for vt in VEHICLE_TYPES}
                    vehicles = traci.edge.getLastStepVehicleIDs(edge)
                    for v in vehicles:
                        vt = traci.vehicle.getTypeID(v)
                        if vt in counts:
                            counts[vt] += 1
                    state.extend([counts[vt] / 10.0 for vt in VEHICLE_TYPES])
                except:
                    state.extend([0.0 for _ in VEHICLE_TYPES])

            for edge in INCOMING_EDGES:
                try:
                    vehicles = traci.edge.getLastStepVehicleIDs(edge)
                    queue_length = sum(1 for v in vehicles if traci.vehicle.getSpeed(v) < 1.0)
                    state.append(min(queue_length / 20.0, 1.0))
                except:
                    state.append(0.0)

            for edge in INCOMING_EDGES:
                try:
                    vehicle_count = traci.edge.getLastStepVehicleNumber(edge)
                    edge_length = traci.lane.getLength(edge + "_0")
                    density = vehicle_count / max(edge_length / 100, 1)
                    state.append(min(density / 5.0, 1.0))
                except:
                    state.append(0.0)

            return np.array(state, dtype=np.float32)

        except Exception as e:
            print(f"Error getting state: {e}")
            return np.zeros(len(VEHICLE_TYPES) * len(INCOMING_EDGES) + 8)

    def get_traffic_metrics(self):
        try:
            if not self.connection_active:
                return self._empty_metrics()

            all_vehicles = traci.vehicle.getIDList()
            total_vehicles = len(all_vehicles)

            if total_vehicles == 0:
                return self._empty_metrics()

            total_waiting_time = sum(traci.vehicle.getWaitingTime(v) for v in all_vehicles)
            total_speed = sum(traci.vehicle.getSpeed(v) for v in all_vehicles)
            avg_speed = total_speed / total_vehicles
            avg_waiting = total_waiting_time / total_vehicles

            queue_lengths = []
            for edge in INCOMING_EDGES:
                try:
                    vehicles = traci.edge.getLastStepVehicleIDs(edge)
                    queue = sum(1 for v in vehicles if traci.vehicle.getSpeed(v) < 1.0)
                    queue_lengths.append(queue)
                except:
                    queue_lengths.append(0)

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

    def get_total_waiting_time(self):
        return self.get_traffic_metrics().get('total_waiting_time', 0)
    
    def get_total_vehicles_exit(self):
        return getattr(self, 'vehicles_exited', 0)

    def _empty_metrics(self):
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
        try:
            reward = 0
            queue_improvement = old_metrics['total_queue'] - new_metrics['total_queue']
            reward += queue_improvement * 10
            waiting_improvement = old_waiting - new_waiting
            reward += waiting_improvement * 0.1
            speed_improvement = new_metrics['avg_speed'] - old_metrics['avg_speed']
            reward += speed_improvement * 5
            if new_metrics['throughput'] > old_metrics['throughput']:
                reward += (new_metrics['throughput'] - old_metrics['throughput']) * 20
            if phase_switched:
                reward -= 5
            if new_metrics['total_queue'] > 15:
                reward -= (new_metrics['total_queue'] - 15) * 2
            queue_balance = np.std(new_metrics['queue_lengths'])
            reward -= queue_balance * 2
            if new_metrics['avg_speed'] < 2.0:
                reward -= 20
            return max(min(reward, 100), -100)

        except Exception as e:
            print(f"Error calculating reward: {e}")
            return -50

    def get_total_vehicles(self):
        try:
            if self.connection_active:
                return len(traci.vehicle.getIDList())
            return 0
        except:
            return 0

    def close(self):
        try:
            if self.connection_active:
                traci.close()
                self.connection_active = False
                time.sleep(0.5)
        except Exception as e:
            print(f"Error closing: {e}")
            self.connection_active = False
