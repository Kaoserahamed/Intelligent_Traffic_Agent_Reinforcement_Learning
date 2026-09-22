# API Reference

## Public API

Importing `traffic_rl` exposes the following public API:

```python
import traffic_rl

# Package metadata
traffic_rl.__version__       # "0.1.0"

# Configuration
from traffic_rl import resolve_config, get_default_config

# CLI entry point (lazily resolved to avoid importing traci)
from traffic_rl import main
```

### Configuration

```python
from traffic_rl.config import (
    AgentConfig,
    Config,
    EnvironmentConfig,
    TrainingConfig,
    default_config_path,
    default_sim_dir,
    project_root,
)
```

| Class               | Description                          |
|---------------------|--------------------------------------|
| `Config`            | Top-level application configuration  |
| `EnvironmentConfig` | SUMO simulation parameters           |
| `TrainingConfig`    | Training hyperparameters             |
| `AgentConfig`       | Agent-specific configuration         |

### CLI

```python
from traffic_rl.cli import main

# Programmatic usage
main(["train", "--agent", "dqn", "--episodes", "200", "--seed", "42"])
```

#### CLI Commands

| Command | Description                     | Key Options                              |
|---------|---------------------------------|------------------------------------------|
| `train` | Train an RL agent            | `--agent`, `--episodes`, `--seed`, `--gui` |
| `eval`  | Evaluate a trained model     | `--agent`, `--model`, `--episodes`         |

### Agents

```python
from traffic_rl.agents import (
    BaseAgent,
    NeuralAgent,
    QLearningAgent,
    DQNAgent,
    DoubleDQNAgent,
    PPOAgent,
)
```

| Agent           | Base Class  | Persistence | Description                        |
|-----------------|-------------|-------------|------------------------------------|
| `QLearningAgent`| `BaseAgent` | JSON        | Tabular Q-learning                 |
| `DQNAgent`      | `NeuralAgent`| `.pth`     | Deep Q-Network with replay         |
| `DoubleDQNAgent`| `NeuralAgent`| `.pth`     | Double DQN (less overestimation)  |
| `PPOAgent`      | `NeuralAgent`| `.pth`     | Proximal Policy Optimization       |

### BaseAgent Interface

```python
class BaseAgent(ABC):
    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int: ...
    def learn(self, state, action, reward, next_state, done) -> None: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
    @property
    def name(self) -> str: ...
```

### SUMO Environment

```python
from traffic_rl.sumo.environment import TrafficEnvironment
from traffic_rl.sumo.binary import find_sumo_binary, build_sumo_command
from traffic_rl.sumo.controller import FixedTimeController, SimpleDynamicController
```

### Metrics

```python
from traffic_rl.metrics import (
    WorldSnapshot,
    TrafficMetrics,
    compute_state,
    compute_reward,
    compute_metrics,
    metrics_to_dict,
)
```

### I/O Utilities

```python
from traffic_rl.io_utils import (
    save_q_table,
    load_q_table,
    save_torch_checkpoint,
    safe_torch_load,
    load_legacy_q_table_pickle,
    sha256_file,
    CheckpointError,
    SafetyError,
)
```

### Settings

```python
from traffic_rl.settings import (
    AppSettings,
    get_settings,
    reset_settings_cache,
    describe_settings,
)
```

### Configuration Loader

```python
from traffic_rl.config_loader import (
    load_experiment_config,
    config_hash,
    ExperimentConfig,
)
```

### Observability

```python
from traffic_rl.observability import (
    configure_structured_logging,
    get_structured_logger,
    bind_context,
    run_context,
    log_event,
    EVENT_CATALOGUE,
    MultiSink,
    JsonlMetricsSink,
    TensorBoardSink,
    PhaseTimer,
    StepRateMeter,
)
```

### Utilities

```python
from traffic_rl.utils import (
    seed_everything,
    numpy_rng,
    ReplayBuffer,
    default_device,
)
```
