from simple_traffic_env import SimpleTrafficEnv
from simple_dynamic_controller import SimpleDynamicController
import time

class FixedTimeController:
    """Simple fixed time controller for comparison"""
    
    def __init__(self, fixed_time=30):
        self.fixed_time = fixed_time
        
        # Add time_allocations attribute for compatibility
        self.time_allocations = {
            'fixed_duration': fixed_time  # Just for reference
        }
        
        self.stats = {
            'allocation_name': f'Fixed_{fixed_time}s',
            'total_phases': 0,
            'phase_durations': []
        }
    
    def calculate_dynamic_green_time(self, edge):
        """Return fixed time regardless of traffic"""
        self.stats['total_phases'] += 1
        self.stats['phase_durations'].append(self.fixed_time)
        
        print(f"  🕐 Fixed time: {self.fixed_time}s for all phases")
        return self.fixed_time, {}, 0
    
    def print_stats_summary(self):
        print(f"\n📈 FIXED TIME PERFORMANCE:")
        print(f"   Fixed Duration: {self.fixed_time}s")
        print(f"   Total Phases: {self.stats['total_phases']}")

def compare_dynamic_vs_fixed():
    """Compare dynamic allocation vs fixed timing"""
    
    sumo_config = "intersection.sumocfg"
    max_steps = 1800
    
    print("🔄 DYNAMIC vs FIXED TIME COMPARISON")
    print("=" * 60)
    
    results = []
    
    # Test Dynamic
    print(f"\n1️⃣ Testing Dynamic Allocation...")
    dynamic_allocations = {'car': 2.0, 'bus': 3.5, 'emergency': 1.5, 'truck': 4.0, 'bike': 1.0}
    dynamic_controller = SimpleDynamicController(dynamic_allocations)
    dynamic_controller.stats['allocation_name'] = 'Dynamic'
    
    env = SimpleTrafficEnv(sumo_config, max_steps, dynamic_controller, gui=True)
    dynamic_result = env.run_simulation()
    
    if dynamic_result:
        results.append(dynamic_result)
        dynamic_controller.print_stats_summary()
        print(f"   Performance Score: {dynamic_result['performance_score']:.2f}")
    
    time.sleep(2)
    
    # Test Fixed Times
    fixed_times = [20, 30, 45, 60]
    
    for fixed_time in fixed_times:
        print(f"\n2️⃣.{fixed_times.index(fixed_time)+1} Testing Fixed {fixed_time}s...")
        
        fixed_controller = FixedTimeController(fixed_time)
        env = SimpleTrafficEnv(sumo_config, max_steps, fixed_controller, gui=True)
        fixed_result = env.run_simulation()
        
        if fixed_result:
            results.append(fixed_result)
            fixed_controller.print_stats_summary()
            print(f"   Performance Score: {fixed_result['performance_score']:.2f}")
        
        time.sleep(2)
    
    # Compare results
    if results:
        print("\n" + "=" * 60)
        print("📊 DYNAMIC vs FIXED COMPARISON")
        print("=" * 60)
        
        print(f"\n{'Method':<12} {'Score':<8} {'Waiting':<10} {'Throughput':<12}")
        print("-" * 45)
        
        for result in sorted(results, key=lambda x: x['performance_score'], reverse=True):
            print(f"{result['allocation_name']:<12} "
                  f"{result['performance_score']:<8.1f} "
                  f"{result['avg_waiting_time']:<10.1f} "
                  f"{result['throughput']:<12.1f}")
        
        best = max(results, key=lambda x: x['performance_score'])
        dynamic_result = next((r for r in results if 'Dynamic' in r['allocation_name']), None)
        
        print(f"\n🏆 Best Method: {best['allocation_name']}")
        
        if dynamic_result and 'Dynamic' in best['allocation_name']:
            fixed_results = [r for r in results if 'Fixed' in r['allocation_name']]
            if fixed_results:
                best_fixed_score = max(r['performance_score'] for r in fixed_results)
                improvement = ((best['performance_score'] - best_fixed_score) / best_fixed_score) * 100
                print(f"💡 Dynamic allocation is {improvement:.1f}% better than best fixed timing!")
        
        elif dynamic_result:
            best_fixed = max((r for r in results if 'Fixed' in r['allocation_name']), 
                           key=lambda x: x['performance_score'])
            comparison = ((dynamic_result['performance_score'] - best_fixed['performance_score']) / 
                         best_fixed['performance_score']) * 100
            if comparison > 0:
                print(f"💡 Dynamic allocation is {comparison:.1f}% better than best fixed timing!")
            else:
                print(f"⚠️ Fixed timing ({best_fixed['allocation_name']}) is {abs(comparison):.1f}% better than dynamic!")

if __name__ == "__main__":
    compare_dynamic_vs_fixed()