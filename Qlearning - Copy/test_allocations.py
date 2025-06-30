from simple_traffic_env import SimpleTrafficEnv
from simple_dynamic_controller import SimpleDynamicController
import time

def test_different_allocations():
    """Test different time allocation strategies"""
    
    sumo_config = "intersection.sumocfg"  # Update with your path
    max_steps = 1800  # 30 minutes simulation
    
    # Define different allocation strategies to test
    allocation_strategies = [
        {
            'name': 'Default',
            'allocations': {'car': 2.0, 'bus': 3.5, 'emergency': 1.5, 'truck': 4.0, 'bike': 1.0}
        },
        {
            'name': 'Fast_Cars',
            'allocations': {'car': 1.5, 'bus': 3.5, 'emergency': 1.0, 'truck': 4.0, 'bike': 1.0}
        },
        {
            'name': 'Slow_Cars',
            'allocations': {'car': 3.0, 'bus': 4.0, 'emergency': 1.5, 'truck': 5.0, 'bike': 1.5}
        },
        {
            'name': 'Emergency_Priority',
            'allocations': {'car': 2.0, 'bus': 3.5, 'emergency': 0.8, 'truck': 4.0, 'bike': 1.0}
        },
        {
            'name': 'Equal_Time',
            'allocations': {'car': 2.5, 'bus': 2.5, 'emergency': 2.5, 'truck': 2.5, 'bike': 2.5}
        },
        {
            'name': 'Bike_Friendly',
            'allocations': {'car': 2.0, 'bus': 3.5, 'emergency': 1.5, 'truck': 4.0, 'bike': 0.5}
        }
    ]
    
    print("🚦 TESTING DIFFERENT TIME ALLOCATION STRATEGIES")
    print("=" * 70)
    
    results = []
    
    for strategy in allocation_strategies:
        print(f"\n🔄 Testing: {strategy['name']}")
        print(f"   Allocations: {strategy['allocations']}")
        print("-" * 50)
        
        # Create controller with specific allocations
        controller = SimpleDynamicController(strategy['allocations'])
        controller.stats['allocation_name'] = strategy['name']
        
        # Create environment
        env = SimpleTrafficEnv(sumo_config, max_steps, controller, gui=True)
        
        # Run simulation
        result = env.run_simulation()
        
        if result:
            results.append(result)
            
            # Print controller stats
            controller.print_stats_summary()
            
            # Print performance results
            print(f"\n🎯 PERFORMANCE RESULTS:")
            print(f"   Total Vehicles: {result['total_vehicles']}")
            print(f"   Average Waiting Time: {result['avg_waiting_time']:.2f} seconds")
            print(f"   Throughput: {result['throughput']:.2f} vehicles/hour")
            print(f"   Performance Score: {result['performance_score']:.2f}")
            print(f"   Phase Switches: {result['phase_switches']}")
        else:
            print("❌ Simulation failed")
        
        # Wait between simulations
        time.sleep(2)
    
    # Compare all results
    print("\n" + "=" * 70)
    print("📊 COMPARISON OF ALL STRATEGIES")
    print("=" * 70)
    
    if results:
        _compare_results(results)
        _print_recommendations(results)
    else:
        print("❌ No valid results to compare")

def _compare_results(results):
    """Compare and rank all results"""
    
    # Sort by performance score
    sorted_results = sorted(results, key=lambda x: x['performance_score'], reverse=True)
    
    print(f"\n🏆 RANKING BY PERFORMANCE SCORE:")
    print("-" * 50)
    for i, result in enumerate(sorted_results, 1):
        print(f"{i}. {result['allocation_name']}: {result['performance_score']:.2f}")
    
    print(f"\n⏱️ RANKING BY WAITING TIME (Lower is Better):")
    print("-" * 50)
    waiting_sorted = sorted(results, key=lambda x: x['avg_waiting_time'])
    for i, result in enumerate(waiting_sorted, 1):
        print(f"{i}. {result['allocation_name']}: {result['avg_waiting_time']:.2f}s")
    
    print(f"\n🚗 RANKING BY THROUGHPUT (Higher is Better):")
    print("-" * 50)
    throughput_sorted = sorted(results, key=lambda x: x['throughput'], reverse=True)
    for i, result in enumerate(throughput_sorted, 1):
        print(f"{i}. {result['allocation_name']}: {result['throughput']:.2f} vehicles/hour")
    
    # Detailed comparison table
    print(f"\n📋 DETAILED COMPARISON TABLE:")
    print("-" * 90)
    print(f"{'Strategy':<15} {'Waiting(s)':<12} {'Throughput':<12} {'Vehicles':<10} {'Score':<8}")
    print("-" * 90)
    
    for result in sorted_results:
        print(f"{result['allocation_name']:<15} "
              f"{result['avg_waiting_time']:<12.2f} "
              f"{result['throughput']:<12.2f} "
              f"{result['total_vehicles']:<10} "
              f"{result['performance_score']:<8.2f}")

def _print_recommendations(results):
    """Print recommendations based on results"""
    
    if not results:
        return
    
    best_overall = max(results, key=lambda x: x['performance_score'])
    best_waiting = min(results, key=lambda x: x['avg_waiting_time'])
    best_throughput = max(results, key=lambda x: x['throughput'])
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 50)
    
    print(f"🥇 Best Overall Performance: {best_overall['allocation_name']}")
    print(f"   Time Allocations: {best_overall['time_allocations']}")
    print(f"   Performance Score: {best_overall['performance_score']:.2f}")
    
    print(f"\n⏱️ Best for Low Waiting Time: {best_waiting['allocation_name']}")
    print(f"   Average Waiting: {best_waiting['avg_waiting_time']:.2f}s")
    print(f"   Time Allocations: {best_waiting['time_allocations']}")
    
    print(f"\n🚗 Best for High Throughput: {best_throughput['allocation_name']}")
    print(f"   Throughput: {best_throughput['throughput']:.2f} vehicles/hour")
    print(f"   Time Allocations: {best_throughput['time_allocations']}")
    
    # Analysis insights
    print(f"\n🔍 KEY INSIGHTS:")
    
    # Find which vehicle type allocations correlate with performance
    car_times = [r['time_allocations']['car'] for r in results]
    scores = [r['performance_score'] for r in results]
    
    if len(set(car_times)) > 1:  # If car times vary
        high_car_time = max(car_times)
        low_car_time = min(car_times)
        high_car_results = [r for r in results if r['time_allocations']['car'] == high_car_time]
        low_car_results = [r for r in results if r['time_allocations']['car'] == low_car_time]
        
        if high_car_results and low_car_results:
            high_avg = sum(r['performance_score'] for r in high_car_results) / len(high_car_results)
            low_avg = sum(r['performance_score'] for r in low_car_results) / len(low_car_results)
            
            if high_avg > low_avg:
                print(f"   • Higher car allocation times ({high_car_time}s) tend to perform better")
            else:
                print(f"   • Lower car allocation times ({low_car_time}s) tend to perform better")

def test_custom_allocation():
    """Test a custom allocation that you can easily modify"""
    
    print("\n🎛️ TESTING CUSTOM ALLOCATION")
    print("=" * 50)
    
    # MODIFY THESE VALUES TO TEST DIFFERENT COMBINATIONS:
    custom_allocation = {
        'car': 2.2,        # seconds per car
        'bus': 3.0,        # seconds per bus  
        'emergency': 1.2,  # seconds per emergency vehicle
        'truck': 3.8,      # seconds per truck
        'bike': 0.8        # seconds per bike
    }
    
    print(f"Testing custom allocation: {custom_allocation}")
    
    sumo_config = "intersection.sumocfg"  # Update with your path
    max_steps = 1800
    
    controller = SimpleDynamicController(custom_allocation)
    controller.stats['allocation_name'] = 'Custom'
    
    env = SimpleTrafficEnv(sumo_config, max_steps, controller, gui=True)
    result = env.run_simulation()
    
    if result:
        controller.print_stats_summary()
        print(f"\n🎯 CUSTOM ALLOCATION RESULTS:")
        print(f"   Performance Score: {result['performance_score']:.2f}")
        print(f"   Average Waiting Time: {result['avg_waiting_time']:.2f}s")
        print(f"   Throughput: {result['throughput']:.2f} vehicles/hour")
        
        return result
    else:
        print("❌ Custom allocation test failed")
        return None

def quick_comparison():
    """Quick comparison between just a few key strategies"""
    
    print("⚡ QUICK COMPARISON - Top 3 Strategies")
    print("=" * 50)
    
    quick_strategies = [
        {
            'name': 'Baseline',
            'allocations': {'car': 2.0, 'bus': 3.5, 'emergency': 1.5, 'truck': 4.0, 'bike': 1.0}
        },
        {
            'name': 'Optimized',
            'allocations': {'car': 1.8, 'bus': 3.0, 'emergency': 1.2, 'truck': 3.5, 'bike': 0.8}
        },
        {
            'name': 'Conservative',
            'allocations': {'car': 2.5, 'bus': 4.0, 'emergency': 1.8, 'truck': 4.5, 'bike': 1.2}
        }
    ]
    
    sumo_config = "intersection.sumocfg"
    max_steps = 900  # Shorter simulation for quick test
    results = []
    
    for strategy in quick_strategies:
        print(f"\n🔄 Testing {strategy['name']}...")
        
        controller = SimpleDynamicController(strategy['allocations'])
        controller.stats['allocation_name'] = strategy['name']
        env = SimpleTrafficEnv(sumo_config, max_steps, controller, gui=True)
        
        result = env.run_simulation()
        if result:
            results.append(result)
            print(f"   Score: {result['performance_score']:.2f}, "
                  f"Waiting: {result['avg_waiting_time']:.1f}s, "
                  f"Throughput: {result['throughput']:.1f}/hr")
        
        time.sleep(1)
    
    if results:
        best = max(results, key=lambda x: x['performance_score'])
        print(f"\n🏆 Winner: {best['allocation_name']} with score {best['performance_score']:.2f}")
        print(f"   Best allocations: {best['time_allocations']}")

if __name__ == "__main__":
    # Choose which test to run:
    
    print("Choose test to run:")
    print("1. Full comparison of all strategies")
    print("2. Test custom allocation")
    print("3. Quick comparison")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        test_different_allocations()
    elif choice == "2":
        test_custom_allocation()
    elif choice == "3":
        quick_comparison()
    else:
        print("Running full comparison by default...")
        test_different_allocations()