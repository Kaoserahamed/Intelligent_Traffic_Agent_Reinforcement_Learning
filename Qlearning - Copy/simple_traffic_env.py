import traci
import numpy as np
import time
import random
from simple_dynamic_controller import SimpleDynamicController

class SimpleTrafficEnv:
    def __init__(self, sumo_config, max_steps, controller=None, gui=True):
        self.sumo_config = sumo_config
        self.max_steps = max_steps
        self.gui = bool(gui)
        
        self.controller = controller if controller else SimpleDynamicController()
        
        # CORRECT PHASE MAPPING based on your actual observations
        self.phase_cycle = [
            {'name': 'South', 'edge': '5to1', 'sumo_phase': 0},   # Phase 0: South green
            {'name': 'East',  'edge': '2to1', 'sumo_phase': 3},   # Phase 3: East green
            {'name': 'North', 'edge': '3to1', 'sumo_phase': 6},   # Phase 6: North green  
            {'name': 'West',  'edge': '4to1', 'sumo_phase': 9}    # Phase 9: West green
        ]
        
        # Current state
        self.current_phase_index = 0  # Start with South
        self.steps = 0
        self.current_phase_duration = 0
        self.required_phase_duration = 15  # Default
        self.phase_start_step = 0  # Track when phase started
        
        # Add phase tracking for debugging
        self.phase_history = []
        
        # SUMO setup
        if self.gui:
            self.sumo_cmd = [
                "sumo-gui", "-c", sumo_config, "--no-warnings", "true", 
                "--start", "true", "--step-length", "1.0", "--time-to-teleport", "300",
                "--end", str(max_steps), "--delay", "100"
            ]
        else:
            self.sumo_cmd = [
                "sumo", "-c", sumo_config, "--no-warnings", "true", 
                "--start", "true", "--quit-on-end", "true", "--step-length", "1.0",
                "--time-to-teleport", "300", "--end", str(max_steps)
            ]
        
        self.connection_active = False
        self.performance_data = {
            'total_waiting_time': 0, 'total_vehicles': 0, 
            'step_count': 0, 'phase_switches': 0
        }
        
    def run_simulation(self):
        """Run simulation with correct phase mapping"""
        try:
            if self.connection_active:
                traci.close()
                time.sleep(0.5)

            seed = random.randint(1, 10000)
            cmd = self.sumo_cmd + ["--seed", str(seed)]
            
            print(f"🚦 Starting simulation with {self.controller.stats['allocation_name']} allocation...")
            print(f"📋 Phase Order: {' → '.join([p['name'] for p in self.phase_cycle])}")
            
            traci.start(cmd)
            self.connection_active = True

            # Reset counters
            self.current_phase_index = 0  # Start with South
            self.steps = 0
            self.current_phase_duration = 0
            self.phase_start_step = 0
            self.phase_history = []
            self.performance_data = {
                'total_waiting_time': 0, 'total_vehicles': 0,
                'step_count': 0, 'phase_switches': 0
            }

            # Wait a few steps for traffic to load before starting
            for _ in range(5):
                traci.simulationStep()
                self.steps += 1

            # Set initial phase and calculate duration
            current_phase = self.phase_cycle[self.current_phase_index]
            traci.trafficlight.setPhase("1", current_phase['sumo_phase'])
            self.phase_start_step = self.steps  # Set start time
            self.required_phase_duration = self._calculate_phase_duration(current_phase)
            
            print(f"🟢 Starting with {current_phase['name']} phase for {self.required_phase_duration}s")

            if self.gui:
                self._setup_gui()

            # Main simulation loop
            while self.steps < self.max_steps:
                try:
                    # Calculate how long current phase has been running
                    self.current_phase_duration = self.steps - self.phase_start_step
                    current_phase = self.phase_cycle[self.current_phase_index]
                    
                    # Verify we're in the correct SUMO phase
                    actual_sumo_phase = traci.trafficlight.getPhase("1")
                    expected_sumo_phase = current_phase['sumo_phase']
                    
                    if actual_sumo_phase != expected_sumo_phase:
                        print(f"⚠️ PHASE MISMATCH! Expected {expected_sumo_phase}, got {actual_sumo_phase}. Correcting...")
                        traci.trafficlight.setPhase("1", expected_sumo_phase)
                    
                    # Check if current phase is complete
                    if self.current_phase_duration >= self.required_phase_duration:
                        print(f"✅ {current_phase['name']} Phase COMPLETED "
                              f"({self.current_phase_duration}s/{self.required_phase_duration}s)")
                        self._switch_to_next_phase()
                        self.performance_data['phase_switches'] += 1
                    
                    # Run simulation step
                    traci.simulationStep()
                    self.steps += 1
                    self.performance_data['step_count'] += 1
                    
                    # Update performance tracking
                    self._update_performance()
                    
                    # GUI updates (no printing)
                    if self.gui and self.steps % 30 == 0:
                        time.sleep(0.05)
                
                except traci.exceptions.FatalTraCIError:
                    print("🛑 GUI window was closed by user")
                    break
                except Exception as e:
                    print(f"⚠️ Step error: {e}")
                    break
            
            # Print phase execution history
            self._print_phase_history()
            
            final_results = self._calculate_final_results()
            print(f"✅ Simulation completed in {self.steps} steps")
            
            if self.gui:
                print("🖥️ Keeping GUI open for 5 seconds...")
                for i in range(50):
                    try:
                        traci.simulationStep()
                        time.sleep(0.1)
                    except:
                        break
            
            return final_results

        except Exception as e:
            print(f"❌ Error in simulation: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self.close()
    
    def _calculate_phase_duration(self, phase_info):
        """Calculate duration for this phase based on its edge traffic"""
        edge = phase_info['edge']
        phase_name = phase_info['name']
        
        try:
            # Get current traffic on this specific edge
            vehicles_on_edge = traci.edge.getLastStepVehicleIDs(edge)
            vehicle_count = len(vehicles_on_edge)
            
            duration, counts, total_vehicles = self.controller.calculate_dynamic_green_time(edge)
            
            # Ensure minimum duration
            final_duration = max(10, min(duration, 60))
            
            return final_duration
            
        except Exception as e:
            print(f"❌ Error calculating duration for {phase_name}: {e}")
            return 15  # Default fallback
    
    def _switch_to_next_phase(self):
        """FIXED: Switch to next phase with proper timing isolation"""
        try:
            current_phase = self.phase_cycle[self.current_phase_index]
            
            # Get current traffic stats for the completing phase
            edge_vehicles = len(traci.edge.getLastStepVehicleIDs(current_phase['edge']))
            total_vehicles = len(traci.vehicle.getIDList())
            
            # Record the completed phase for debugging
            phase_record = {
                'name': current_phase['name'],
                'edge': current_phase['edge'],
                'sumo_phase': current_phase['sumo_phase'],
                'planned_duration': self.required_phase_duration,
                'actual_duration': self.current_phase_duration,
                'start_step': self.phase_start_step,
                'end_step': self.steps,
                'edge_vehicles': edge_vehicles,
                'total_vehicles': total_vehicles
            }
            self.phase_history.append(phase_record)
            
            # Log the phase completion stats
            print(f"📊 {current_phase['name']} COMPLETED: {self.current_phase_duration}s/{self.required_phase_duration}s | "
                  f"Edge: {edge_vehicles} vehicles | Total: {total_vehicles} vehicles")
            
            # Move to next phase: South → East → North → West → South...
            self.current_phase_index = (self.current_phase_index + 1) % len(self.phase_cycle)
            next_phase = self.phase_cycle[self.current_phase_index]
            
            # CRITICAL FIX: Set phase and reset timing IMMEDIATELY
            traci.trafficlight.setPhase("1", next_phase['sumo_phase'])
            
            # IMPORTANT: Reset phase timing RIGHT NOW, before any calculations
            self.current_phase_duration = 0
            self.phase_start_step = self.steps  # Mark when THIS phase started
            
            # Wait a few steps for the new phase to activate properly
            for _ in range(3):
                traci.simulationStep()
                self.steps += 1
            
            # NOW calculate duration for the NEW phase based on ITS edge traffic
            self.required_phase_duration = self._calculate_phase_duration(next_phase)
            
            # Get traffic stats for the new phase
            new_edge_vehicles = len(traci.edge.getLastStepVehicleIDs(next_phase['edge']))
            new_total_vehicles = len(traci.vehicle.getIDList())
            
            print(f"🔄 {next_phase['name']} STARTED: {self.required_phase_duration}s allocated | "
                  f"Edge: {new_edge_vehicles} vehicles | Total: {new_total_vehicles} vehicles")
            print("-" * 80)
            
        except Exception as e:
            print(f"❌ Error switching phase: {e}")
            import traceback
            traceback.print_exc()
    
    def _print_phase_history(self):
        """Print complete phase execution history for debugging"""
        if not self.phase_history:
            return
            
        print(f"\n📚 PHASE EXECUTION HISTORY:")
        print("-" * 95)
        print(f"{'#':<3} {'Phase':<6} {'Edge':<6} {'SUMO':<5} {'Planned':<8} {'Actual':<7} {'Edge V.':<8} {'Total V.':<8} {'Steps':<15}")
        print("-" * 95)
        
        for i, record in enumerate(self.phase_history):
            edge_v = record.get('edge_vehicles', 'N/A')
            total_v = record.get('total_vehicles', 'N/A')
            print(f"{i+1:<3} {record['name']:<6} {record['edge']:<6} "
                  f"{record['sumo_phase']:<5} {record['planned_duration']:<8} "
                  f"{record['actual_duration']:<7} {edge_v:<8} {total_v:<8} "
                  f"{record['start_step']}-{record['end_step']}")
        
        # Check for timing issues
        print(f"\n🔍 TIMING ANALYSIS:")
        timing_issues = []
        
        for i in range(len(self.phase_history) - 1):
            current = self.phase_history[i]
            next_phase = self.phase_history[i + 1]
            
            gap = next_phase['start_step'] - current['end_step']
            if gap > 5:  # More than expected gap
                timing_issues.append(f"Large gap between {current['name']} and {next_phase['name']}: {gap} steps")
        
        if timing_issues:
            for issue in timing_issues:
                print(f"⚠️ {issue}")
        else:
            print("✅ No significant timing issues detected")
    
    def _update_performance(self):
        """Update performance metrics"""
        try:
            all_vehicles = traci.vehicle.getIDList()
            self.performance_data['total_vehicles'] = len(all_vehicles)
            
            total_waiting = sum(traci.vehicle.getWaitingTime(v) for v in all_vehicles)
            self.performance_data['total_waiting_time'] += total_waiting
            
        except:
            pass
    
    def _calculate_final_results(self):
        """Calculate final performance results"""
        try:
            all_vehicles = traci.vehicle.getIDList()
            total_vehicles = len(all_vehicles)
            
            avg_waiting_time = self.performance_data['total_waiting_time'] / max(self.steps, 1)
            throughput = total_vehicles / max(self.steps / 3600, 0.1)
            
            results = {
                'allocation_name': self.controller.stats['allocation_name'],
                'time_allocations': self.controller.time_allocations.copy(),
                'total_steps': self.steps,
                'total_vehicles': total_vehicles,
                'avg_waiting_time': avg_waiting_time,
                'throughput': throughput,
                'phase_switches': self.performance_data['phase_switches'],
                'performance_score': throughput * 10 - avg_waiting_time * 0.1,
                'phase_order': [p['name'] for p in self.phase_cycle],
                'phase_history': self.phase_history.copy()  # Add phase history to results
            }
            
            return results
            
        except Exception as e:
            print(f"Error calculating results: {e}")
            return None
    
    def _setup_gui(self):
        """Setup GUI view"""
        try:
            traci.gui.setZoom("View #0", 800)
            traci.gui.setOffset("View #0", 500, 500)
            traci.gui.setSchema("View #0", "real world")
        except Exception as e:
            print(f"⚠️ GUI setup warning: {e}")
    
    def debug_phase_states(self):
        """Debug function to verify phase mappings"""
        print(f"\n🔍 DEBUGGING PHASE STATES:")
        print("-" * 50)
        
        original_phase = traci.trafficlight.getPhase("1")
        
        for phase_info in self.phase_cycle:
            try:
                phase_num = phase_info['sumo_phase']
                phase_name = phase_info['name']
                
                traci.trafficlight.setPhase("1", phase_num)
                for _ in range(3):  # Wait for phase to activate
                    traci.simulationStep()
                
                # Get traffic light state
                program = traci.trafficlight.getAllProgramLogics("1")[0]
                if phase_num < len(program.phases):
                    state = program.phases[phase_num].state
                    print(f"Phase {phase_num} ({phase_name}): {state}")
                else:
                    print(f"Phase {phase_num} ({phase_name}): PHASE NOT FOUND!")
                    
            except Exception as e:
                print(f"Error checking phase {phase_num} ({phase_name}): {e}")
        
        # Restore original phase
        try:
            traci.trafficlight.setPhase("1", original_phase)
        except:
            pass
        
        print("-" * 50)
    
    def get_current_traffic_summary(self):
        """Get current traffic summary for all edges"""
        summary = {}
        total_vehicles = 0
        
        for phase_info in self.phase_cycle:
            edge = phase_info['edge']
            direction = phase_info['name']
            
            try:
                vehicles = traci.edge.getLastStepVehicleIDs(edge)
                vehicle_count = len(vehicles)
                
                # Get vehicle types
                vehicle_types = {}
                for vehicle in vehicles:
                    try:
                        vtype = traci.vehicle.getTypeID(vehicle)
                        vehicle_types[vtype] = vehicle_types.get(vtype, 0) + 1
                    except:
                        continue
                
                summary[direction] = {
                    'edge': edge,
                    'total_vehicles': vehicle_count,
                    'vehicle_types': vehicle_types
                }
                total_vehicles += vehicle_count
                
            except Exception as e:
                print(f"Error getting traffic for {direction} ({edge}): {e}")
                summary[direction] = {
                    'edge': edge,
                    'total_vehicles': 0,
                    'vehicle_types': {}
                }
        
        summary['total_all_edges'] = total_vehicles
        return summary
    
    def print_current_traffic(self):
        """Print current traffic state across all edges"""
        summary = self.get_current_traffic_summary()
        
        print(f"\n📍 CURRENT TRAFFIC STATE (Step {self.steps}):")
        print("-" * 50)
        
        for phase_info in self.phase_cycle:
            direction = phase_info['name']
            if direction in summary:
                info = summary[direction]
                print(f"   {direction:>5} ({info['edge']}): {info['total_vehicles']:2d} vehicles")
                
                # Show breakdown if there are vehicles
                if info['total_vehicles'] > 0 and info['vehicle_types']:
                    breakdown = [f"{vtype}:{count}" for vtype, count in info['vehicle_types'].items() if count > 0]
                    if breakdown:
                        print(f"         → {', '.join(breakdown)}")
        
        print(f"   {'Total':>5}: {summary['total_all_edges']} vehicles")
        print("-" * 50)
    
    def force_phase_switch(self, target_phase_name):
        """Force switch to a specific phase (for testing)"""
        try:
            # Find the target phase
            target_index = None
            for i, phase_info in enumerate(self.phase_cycle):
                if phase_info['name'].lower() == target_phase_name.lower():
                    target_index = i
                    break
            
            if target_index is None:
                print(f"❌ Phase '{target_phase_name}' not found!")
                return False
            
            print(f"🔧 FORCING switch to {target_phase_name} phase...")
            
            # Record current phase completion
            current_phase = self.phase_cycle[self.current_phase_index]
            phase_record = {
                'name': current_phase['name'],
                'edge': current_phase['edge'],
                'sumo_phase': current_phase['sumo_phase'],
                'planned_duration': self.required_phase_duration,
                'actual_duration': self.current_phase_duration,
                'start_step': self.phase_start_step,
                'end_step': self.steps,
                'forced_switch': True
            }
            self.phase_history.append(phase_record)
            
            # Switch to target phase
            self.current_phase_index = target_index
            target_phase = self.phase_cycle[target_index]
            
            # Set the phase and reset timing
            traci.trafficlight.setPhase("1", target_phase['sumo_phase'])
            self.current_phase_duration = 0
            self.phase_start_step = self.steps
            
            # Calculate duration for new phase
            self.required_phase_duration = self._calculate_phase_duration(target_phase)
            
            print(f"✅ Forced switch to {target_phase_name} phase completed")
            return True
            
        except Exception as e:
            print(f"❌ Error forcing phase switch: {e}")
            return False
    
    def close(self):
        try:
            if self.connection_active:
                if self.gui:
                    print("🖥️ Closing GUI...")
                traci.close()
                self.connection_active = False
                time.sleep(0.5)
        except:
            pass

    def __del__(self):
        """Destructor to ensure SUMO connection is closed"""
        self.close()