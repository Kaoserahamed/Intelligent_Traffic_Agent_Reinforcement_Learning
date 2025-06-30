import numpy as np
import matplotlib.pyplot as plt
import json
import time
from enhanced_traffic_env import EnhancedTrafficEnv

class TrafficComparison:
    def __init__(self, sumo_config, max_steps=3600, num_episodes=10):
        self.sumo_config = sumo_config
        self.max_steps = max_steps
        self.num_episodes = num_episodes
        
        self.results = {
            'dynamic': [],
            'fixed': []
        }
    
    def run_comparison(self):
        """Run comparison between dynamic and fixed timing"""
        
        print("🚦 Starting Traffic Light Control Comparison")
        print(f"Episodes: {self.num_episodes}, Steps per episode: {self.max_steps}")
        print("=" * 60)
        
        # Test Dynamic Controller
        print("🔄 Testing Dynamic Time Allocation...")
        for episode in range(self.num_episodes):
            print(f"  Episode {episode + 1}/{self.num_episodes}")
            
            env = EnhancedTrafficEnv(
                sumo_config=self.sumo_config,
                max_steps=self.max_steps,
                gui=True,
                controller_type='dynamic'
            )
            
            result = self._run_episode(env, f"Dynamic-{episode}")
            self.results['dynamic'].append(result)
            env.close()
            time.sleep(1)
        
        # Test Fixed Controller
        print("\n⏱️ Testing Fixed Time Allocation...")
        for episode in range(self.num_episodes):
            print(f"  Episode {episode + 1}/{self.num_episodes}")
            
            env = EnhancedTrafficEnv(
                sumo_config=self.sumo_config,
                max_steps=self.max_steps,
                gui=True,
                controller_type='fixed'
            )
            
            result = self._run_episode(env, f"Fixed-{episode}")
            self.results['fixed'].append(result)
            env.close()
            time.sleep(1)
        
        # Analyze and display results
        self._analyze_results()
        self._plot_results()
        self._save_results()
    
    def _run_episode(self, env, episode_name):
        """Run a single episode and collect metrics"""
        try:
            env.reset()
            
            total_reward = 0
            step_count = 0
            
            while step_count < self.max_steps:
                state, reward, done = env.step()
                total_reward += reward
                step_count += 1
                
                if done:
                    break
            
            # Get final performance summary
            performance = env.get_performance_summary()
            
            return {
                'episode_name': episode_name,
                'total_reward': total_reward,
                'steps': step_count,
                'avg_waiting_time': performance['avg_waiting_time'],
                'throughput': performance['throughput'],
                'total_vehicles_served': performance['total_vehicles_served'],
                'performance_score': performance['throughput'] * 10 - performance['avg_waiting_time'] * 0.1,
                'allocation_summary': performance.get('allocation_summary'),
                'phase_stats': performance.get('phase_stats')
            }
            
        except Exception as e:
            print(f"Error in episode {episode_name}: {e}")
            return {
                'episode_name': episode_name,
                'error': str(e),
                'total_reward': -1000,
                'steps': 0,
                'avg_waiting_time': 999,
                'throughput': 0,
                'total_vehicles_served': 0,
                'performance_score': -1000
            }
    
    def _analyze_results(self):
        """Analyze and compare results"""
        print("\n" + "=" * 60)
        print("📊 RESULTS ANALYSIS")
        print("=" * 60)
        
        # Calculate statistics for each method
        for method in ['dynamic', 'fixed']:
            results = self.results[method]
            valid_results = [r for r in results if 'error' not in r]
            
            if not valid_results:
                print(f"❌ {method.upper()}: No valid results")
                continue
            
            avg_waiting = np.mean([r['avg_waiting_time'] for r in valid_results])
            avg_throughput = np.mean([r['throughput'] for r in valid_results])
            avg_performance = np.mean([r['performance_score'] for r in valid_results])
            avg_vehicles_served = np.mean([r['total_vehicles_served'] for r in valid_results])
            
            std_waiting = np.std([r['avg_waiting_time'] for r in valid_results])
            std_throughput = np.std([r['throughput'] for r in valid_results])
            
            print(f"\n🎯 {method.upper()} CONTROLLER:")
            print(f"   Average Waiting Time: {avg_waiting:.2f} ± {std_waiting:.2f} seconds")
            print(f"   Average Throughput: {avg_throughput:.2f} ± {std_throughput:.2f} vehicles/hour")
            print(f"   Average Vehicles Served: {avg_vehicles_served:.0f}")
            print(f"   Performance Score: {avg_performance:.2f}")
        
        # Compare methods
        if self.results['dynamic'] and self.results['fixed']:
            dynamic_valid = [r for r in self.results['dynamic'] if 'error' not in r]
            fixed_valid = [r for r in self.results['fixed'] if 'error' not in r]
            
            if dynamic_valid and fixed_valid:
                dynamic_avg_waiting = np.mean([r['avg_waiting_time'] for r in dynamic_valid])
                fixed_avg_waiting = np.mean([r['avg_waiting_time'] for r in fixed_valid])
                
                dynamic_avg_throughput = np.mean([r['throughput'] for r in dynamic_valid])
                fixed_avg_throughput = np.mean([r['throughput'] for r in fixed_valid])
                
                waiting_improvement = ((fixed_avg_waiting - dynamic_avg_waiting) / fixed_avg_waiting) * 100
                throughput_improvement = ((dynamic_avg_throughput - fixed_avg_throughput) / fixed_avg_throughput) * 100
                
                print(f"\n🚀 IMPROVEMENT WITH DYNAMIC ALLOCATION:")
                print(f"   Waiting Time: {waiting_improvement:+.1f}%")
                print(f"   Throughput: {throughput_improvement:+.1f}%")
                
                if waiting_improvement > 0 and throughput_improvement > 0:
                    print("   ✅ Dynamic allocation performs better!")
                elif waiting_improvement > 0 or throughput_improvement > 0:
                    print("   ⚖️ Mixed results - some improvement")
                else:
                    print("   ❌ Fixed allocation performs better")
        
        # Show best dynamic allocation if available
        dynamic_valid = [r for r in self.results['dynamic'] if 'error' not in r and r.get('allocation_summary')]
        if dynamic_valid:
            best_result = max(dynamic_valid, key=lambda x: x['performance_score'])
            allocation_summary = best_result.get('allocation_summary')
            if allocation_summary and allocation_summary.get('best_allocation'):
                print(f"\n🎯 BEST FOUND TIME ALLOCATIONS:")
                for vtype, time_val in allocation_summary['best_allocation'].items():
                    print(f"   {vtype.capitalize()}: {time_val:.1f} seconds")
    
    def _plot_results(self):
        """Create comparison plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Traffic Light Control Comparison', fontsize=16)
            
            # Extract valid results
            dynamic_valid = [r for r in self.results['dynamic'] if 'error' not in r]
            fixed_valid = [r for r in self.results['fixed'] if 'error' not in r]
            
            if not dynamic_valid or not fixed_valid:
                print("⚠️ Insufficient data for plotting")
                return
            
            # Waiting Time Comparison
            dynamic_waiting = [r['avg_waiting_time'] for r in dynamic_valid]
            fixed_waiting = [r['avg_waiting_time'] for r in fixed_valid]
            
            axes[0, 0].boxplot([dynamic_waiting, fixed_waiting], labels=['Dynamic', 'Fixed'])
            axes[0, 0].set_title('Average Waiting Time')
            axes[0, 0].set_ylabel('Seconds')
            
            # Throughput Comparison
            dynamic_throughput = [r['throughput'] for r in dynamic_valid]
            fixed_throughput = [r['throughput'] for r in fixed_valid]
            
            axes[0, 1].boxplot([dynamic_throughput, fixed_throughput], labels=['Dynamic', 'Fixed'])
            axes[0, 1].set_title('Throughput')
            axes[0, 1].set_ylabel('Vehicles/Hour')
            
            # Performance Score
            dynamic_performance = [r['performance_score'] for r in dynamic_valid]
            fixed_performance = [r['performance_score'] for r in fixed_valid]
            
            axes[1, 0].boxplot([dynamic_performance, fixed_performance], labels=['Dynamic', 'Fixed'])
            axes[1, 0].set_title('Performance Score')
            axes[1, 0].set_ylabel('Score')
            
            # Episode-wise comparison
            episodes = range(1, min(len(dynamic_valid), len(fixed_valid)) + 1)
            axes[1, 1].plot(episodes, dynamic_waiting[:len(episodes)], 'b-o', label='Dynamic')
            axes[1, 1].plot(episodes, fixed_waiting[:len(episodes)], 'r-s', label='Fixed')
            axes[1, 1].set_title('Waiting Time by Episode')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Average Waiting Time (s)')
            axes[1, 1].legend()
            
            plt.tight_layout()
            plt.savefig('traffic_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("📊 Comparison plots saved as 'traffic_comparison.png'")
            
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    def _save_results(self):
        """Save results to JSON file"""
        try:
            with open('traffic_comparison_results.json', 'w') as f:
                json.dump(self.results, f, indent=2)
            print("💾 Results saved to 'traffic_comparison_results.json'")
        except Exception as e:
            print(f"Error saving results: {e}")

def main():
    """Main function to run the comparison"""
    
    # Configuration
    sumo_config = "intersection.sumocfg"  # Update with your SUMO config file path
    max_steps = 3600  # 1 hour simulation
    num_episodes = 5  # Number of episodes per method
    
    # Run comparison
    comparison = TrafficComparison(sumo_config, max_steps, num_episodes)
    comparison.run_comparison()

if __name__ == "__main__":
    main()