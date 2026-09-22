# RL Models & Reward Design

## Algorithms

TrafficRL implements four reinforcement learning algorithms:

### Q-Learning (Tabular)

- **State**: discretized into bins (default 5 bins per dimension)
- **Action**: select one of 4 green phases
- **Update**: Q-learning temporal difference update
- **Persistence**: Q-table saved as JSON with SHA-256 checksum
- **Use case**: Fast prototyping, small state spaces

```python
# Q-Learning update rule
Q(s, a) ← Q(s, a) + α [r + γ max Q(s', a') - Q(s, a)]
```

### Deep Q-Network (DQN)

- **Network**: 3-layer MLP (state_size → 128 → 128 → action_size)
- **Experience replay**: 10,000 transition buffer
- **Target network**: Updated every 100 steps
- **Optimizer**: Adam (lr=0.001)
- **Exploration**: Epsilon-greedy with decay (0.1 → 0.01, decay=0.995)

### Double DQN

- Extends DQN to reduce overestimation bias
- **Action selection**: Uses online network (argmax)
- **Action evaluation**: Uses target network (Q-value)
- Reduces the positive bias in Q-value estimates

### Proximal Policy Optimization (PPO)

- **Architecture**: Shared backbone (256×256) with separate policy/value heads
- **Objective**: Clipped surrogate objective with GAE
- **Hyperparameters**:
  - `clip_epsilon`: 0.2 (PPO clipping range)
  - `ppo_epochs`: 10 (optimization epochs per rollout)
  - `batch_size`: 256
  - `rollout_steps`: 2048
  - `entropy_coef`: 0.01 (exploration bonus)
  - `value_coef`: 0.5
  - `max_grad_norm`: 0.5 (gradient clipping)
  - `gae_lambda`: 0.95 (advantage estimation)
  - `target_kl`: 0.02 (early stopping for KL divergence)

## State Representation

The observation is a **28-dimensional vector** computed by `compute_state()`:

```
Per incoming edge (4 edges × 7 features):
├── 5 normalized vehicle-type counts (car, bus, emergency, truck, bike)
├── 1 normalized queue length (vehicles with speed < 1.0 m/s)
└── 1 normalized density (vehicles / edge_length)

Total: 4 × (5 + 1 + 1) = 28 dimensions
```

All features are normalized to [0, 1] for stable neural network training.

## Reward Function

The reward balances multiple traffic objectives. All weights are explicit
and auditable in `traffic_rl/metrics.py`:

```python
reward = 0.0
reward += (old_queue - new_queue) * 10.0          # queue improvement
reward += (old_waiting - new_waiting) * 0.1      # waiting time reduction
reward += (new_speed - old_speed) * 5.0          # speed improvement
reward += throughput_delta * 20.0                 # vehicles exited
reward -= 5.0 if phase_switched else 0            # switch penalty
reward -= queue_excess * 2.0                      # queue penalty (>15)
reward -= queue_std * 2.0                         # queue imbalance
reward -= 20.0 if avg_speed < 2.0 else 0          # low speed penalty
reward = clamp(reward, -100, 100)
```

## Evaluation Protocol

Production evaluation follows a rigorous protocol defined in
`configs/prod.yaml`:

1. **Multi-seed**: 5 random seeds (1, 2, 3, 4, 5)
2. **Multi-split**: Validation and test scenarios
3. **Baselines**: Compared against fixed-time, actuated, and max-pressure controllers
4. **Quality gates**:
   - Minimum 10% improvement over fixed-time baseline
   - Average delay ≤ 45 seconds
   - 95th percentile queue ≤ 25 vehicles

## Training Configuration

### Local

```yaml
run:
  episodes: 20
  seed: 42
environment:
  max_steps: 240  # 20 minutes of traffic
model:
  hidden_sizes: [128, 128]
agent:
  kind: ppo
  hyperparams:
    batch_size: 64
    ppo_epochs: 4
```

### Production

```yaml
run:
  episodes: 1000
  seed: 42
environment:
  max_steps: 720  # 1 hour of traffic
model:
  hidden_sizes: [512, 512]
  attention_heads: 8
agent:
  kind: ppo
  hyperparams:
    batch_size: 512
    ppo_epochs: 10
    use_amp: true
```

## Domain Randomization

Training includes stochastic perturbations to improve generalization:

| Perturbation           | Range / Probability |
|------------------------|---------------------|
| Demand scale           | 0.7–1.3×            |
| Vehicle mix jitter     | ±20%                |
| Weather probability    | 20% (prod: 30%)     |
| Incident probability   | 10% (prod: 15%)     |
| Sensor noise           | 10% (prod: 15%)     |
