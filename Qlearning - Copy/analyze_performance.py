# analyze_performance.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_training_performance():
    """Analyze the training performance from logs"""
    
    try:
        # Load training log
        df = pd.read_csv("logs/training_log.csv")
        
        print("="*80)
        print("📈 TRAINING PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Basic statistics
        print(f"Total Episodes: {len(df)}")
        print(f"Final Reward: {df['Total_Reward'].iloc[-1]:.2f}")
        print(f"Best Reward: {df['Total_Reward'].max():.2f}")
        print(f"Final Epsilon: {df['Epsilon'].iloc[-1]:.3f}")
        print(f"Final Q-Table Size: {df['Q_Table_Size'].iloc[-1]}")
        
        # Calculate improvement
        early_performance = df['Total_Reward'].head(10).mean()
        late_performance = df['Total_Reward'].tail(10).mean()
        improvement = ((late_performance - early_performance) / abs(early_performance)) * 100
        
        print(f"\nPerformance Improvement:")
        print(f"  Early Average (first 10): {early_performance:.2f}")
        print(f"  Late Average (last 10): {late_performance:.2f}")
        print(f"  Improvement: {improvement:.1f}%")
        
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reward over time
        axes[0,0].plot(df['Episode'], df['Total_Reward'])
        axes[0,0].set_title('Total Reward Over Episodes')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Total Reward')
        axes[0,0].grid(True)
        
        # Queue length over time
        axes[0,1].plot(df['Episode'], df['Avg_Queue_Length'])
        axes[0,1].set_title('Average Queue Length Over Episodes')
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Average Queue Length')
        axes[0,1].grid(True)
        
        # Waiting time over time
        axes[1,0].plot(df['Episode'], df['Total_Waiting_Time'])
        axes[1,0].set_title('Total Waiting Time Over Episodes')
        axes[1,0].set_xlabel('Episode')
        axes[1,0].set_ylabel('Total Waiting Time (s)')
        axes[1,0].grid(True)
        
        # Epsilon decay
        axes[1,1].plot(df['Episode'], df['Epsilon'])
        axes[1,1].set_title('Epsilon (Exploration Rate) Over Episodes')
        axes[1,1].set_xlabel('Episode')
        axes[1,1].set_ylabel('Epsilon')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig('logs/training_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Training analysis plots saved to: logs/training_analysis.png")
        
    except FileNotFoundError:
        print("❌ Training log file not found. Make sure you have run the training first.")
    except Exception as e:
        print(f"❌ Error analyzing performance: {e}")

if __name__ == "__main__":
    analyze_training_performance()