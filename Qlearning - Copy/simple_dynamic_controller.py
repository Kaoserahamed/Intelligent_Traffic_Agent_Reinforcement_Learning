import numpy as np
import traci

class SimpleDynamicController:
    def __init__(self, time_allocations=None):
        # Default time allocations (seconds per vehicle)
        self.default_allocations = {
            'car': 2.0,
            'bus': 3.5,
            'emergency': 1.5,
            'truck': 4.0,
            'bike': 1.0
        }
        
        # Use provided allocations or defaults
        self.time_allocations = time_allocations if time_allocations else self.default_allocations.copy()
        
        # CORRECTED phase information based on actual observations:
        # South (Phase 0) → East (Phase 1) → North (Phase 4) → West (Phase 7)
        self.phase_info = {
            'South': {'edge': '5to1', 'sumo_phase': 0},
            'East':  {'edge': '2to1', 'sumo_phase': 3}, 
            'North': {'edge': '3to1', 'sumo_phase': 6},
            'West':  {'edge': '4to1', 'sumo_phase': 9}
        }
        
        # Edge list in phase order
        self.incoming_edges = ['5to1', '2to1', '3to1', '4to1']  # S, E, N, W
        self.phases = [0, 3, 6, 9]  # Corresponding SUMO phase numbers
        
        # Stats tracking
        self.stats = {
            'total_phases': 0,
            'phase_durations': [],
            'vehicle_counts_per_phase': [],
            'edge_traffic_history': {},  # Track traffic per edge over time
            'allocation_name': getattr(time_allocations, 'name', 'Default')
        }
        
        # Initialize edge traffic history
        for edge in self.incoming_edges:
            self.stats['edge_traffic_history'][edge] = []
        
    def get_vehicle_counts_by_type(self, edge):
        """Get count of each vehicle type on a specific edge"""
        try:
            vehicles = traci.edge.getLastStepVehicleIDs(edge)
            counts = {vtype: 0 for vtype in self.time_allocations.keys()}
            
            for vehicle in vehicles:
                try:
                    vtype = traci.vehicle.getTypeID(vehicle)
                    if vtype in counts:
                        counts[vtype] += 1
                except:
                    continue
                    
            return counts
        except:
            return {vtype: 0 for vtype in self.time_allocations.keys()}
    
    def get_edge_direction_name(self, edge):
        """Get the direction name for an edge"""
        edge_to_direction = {
            '5to1': 'South',
            '2to1': 'East', 
            '3to1': 'North',
            '4to1': 'West'
        }
        return edge_to_direction.get(edge, 'Unknown')
    
    def calculate_dynamic_green_time(self, edge):
        """Calculate green time based on vehicle types and counts on specific edge"""
        vehicle_counts = self.get_vehicle_counts_by_type(edge)
        direction = self.get_edge_direction_name(edge)
        
        total_time = 0
        total_vehicles = 0
        
        print(f"  📊 {direction} Edge ({edge}):", end=" ")
        
        # Calculate time needed for each vehicle type
        time_breakdown = {}
        for vtype, count in vehicle_counts.items():
            if count > 0:
                time_needed = count * self.time_allocations[vtype]
                total_time += time_needed
                total_vehicles += count
                time_breakdown[vtype] = time_needed
                print(f"{vtype}({count}×{self.time_allocations[vtype]}s={time_needed:.1f}s)", end=" ")
        
        # Set minimum and maximum green time limits
        min_green_time = 10
        max_green_time = 90
        
        if total_vehicles == 0:
            green_time = min_green_time
            print("No vehicles - using minimum time")
        else:
            # Add 20% buffer for safety
             green_time = max(10, min(total_time * 1.3 + 5, 60))  # 30% buffer + 5s startup
        return int(green_time), vehicle_counts, total_vehicles
    
    def get_edge_statistics(self, edge):
        """Get statistics for a specific edge"""
        if edge not in self.stats['edge_traffic_history']:
            return None
            
        history = self.stats['edge_traffic_history'][edge]
        if not history:
            return None
            
        total_vehicles = sum(h['total_vehicles'] for h in history)
        total_time = sum(h['allocated_time'] for h in history)
        avg_vehicles_per_phase = total_vehicles / len(history) if history else 0
        avg_time_per_phase = total_time / len(history) if history else 0
        
        return {
            'edge': edge,
            'direction': self.get_edge_direction_name(edge),
            'total_phases': len(history),
            'total_vehicles': total_vehicles,
            'total_allocated_time': total_time,
            'avg_vehicles_per_phase': avg_vehicles_per_phase,
            'avg_time_per_phase': avg_time_per_phase
        }
    
    def print_stats_summary(self):
        """Print comprehensive summary statistics"""
        if self.stats['total_phases'] == 0:
            print("❌ No phase data available")
            return
            
        avg_duration = np.mean(self.stats['phase_durations'])
        min_duration = np.min(self.stats['phase_durations'])
        max_duration = np.max(self.stats['phase_durations'])
        
        print(f"\n📈 ALLOCATION PERFORMANCE ({self.stats['allocation_name']}):")
        print(f"   Time Allocations: {self.time_allocations}")
        print(f"   Total Phases: {self.stats['total_phases']}")
        print(f"   Average Phase Duration: {avg_duration:.1f}s")
        print(f"   Min/Max Duration: {min_duration:.0f}s / {max_duration:.0f}s")
        
        # Overall vehicle type statistics
        total_vehicles_by_type = {vtype: 0 for vtype in self.time_allocations.keys()}
        for phase_counts in self.stats['vehicle_counts_per_phase']:
            for vtype, count in phase_counts.items():
                total_vehicles_by_type[vtype] += count
        
        print(f"   Total Vehicles Processed:")
        for vtype, total in total_vehicles_by_type.items():
            if total > 0:
                print(f"     {vtype.capitalize()}: {total}")
        
        # Edge-specific statistics
        print(f"\n📊 EDGE-SPECIFIC STATISTICS:")
        print("-" * 50)
        
        for edge in self.incoming_edges:
            edge_stats = self.get_edge_statistics(edge)
            if edge_stats:
                print(f"   {edge_stats['direction']} ({edge}):")
                print(f"     Phases: {edge_stats['total_phases']}")
                print(f"     Total Vehicles: {edge_stats['total_vehicles']}")
                print(f"     Total Time: {edge_stats['total_allocated_time']:.1f}s")
                print(f"     Avg per Phase: {edge_stats['avg_vehicles_per_phase']:.1f} vehicles, {edge_stats['avg_time_per_phase']:.1f}s")
        
        # Traffic distribution analysis
        print(f"\n🚦 TRAFFIC DISTRIBUTION:")
        print("-" * 30)
        
        edge_totals = {}
        for edge in self.incoming_edges:
            edge_stats = self.get_edge_statistics(edge)
            if edge_stats:
                edge_totals[self.get_edge_direction_name(edge)] = edge_stats['total_vehicles']
        
        total_all_edges = sum(edge_totals.values())
        if total_all_edges > 0:
            for direction, count in edge_totals.items():
                percentage = (count / total_all_edges) * 100
                print(f"   {direction}: {count} vehicles ({percentage:.1f}%)")
    
    def get_current_traffic_summary(self):
        """Get current traffic summary across all edges"""
        summary = {}
        total_vehicles = 0
        
        for edge in self.incoming_edges:
            counts = self.get_vehicle_counts_by_type(edge)
            edge_total = sum(counts.values())
            direction = self.get_edge_direction_name(edge)
            
            summary[direction] = {
                'edge': edge,
                'total_vehicles': edge_total,
                'vehicle_types': counts
            }
            total_vehicles += edge_total
        
        summary['total_all_edges'] = total_vehicles
        return summary
    
    def print_current_traffic(self):
        """Print current traffic state"""
        summary = self.get_current_traffic_summary()
        
        print(f"\n📍 CURRENT TRAFFIC STATE:")
        print("-" * 40)
        
        for direction in ['South', 'East', 'North', 'West']:
            if direction in summary:
                info = summary[direction]
                print(f"   {direction} ({info['edge']}): {info['total_vehicles']} vehicles")
                
                # Show breakdown if there are vehicles
                if info['total_vehicles'] > 0:
                    breakdown = [f"{vtype}:{count}" for vtype, count in info['vehicle_types'].items() if count > 0]
                    if breakdown:
                        print(f"     → {', '.join(breakdown)}")
        
        print(f"   Total: {summary['total_all_edges']} vehicles")