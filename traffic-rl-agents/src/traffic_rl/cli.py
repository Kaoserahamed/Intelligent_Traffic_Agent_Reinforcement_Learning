"""Command-line interface for TrafficRL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from traffic_rl import __version__, resolve_config
from traffic_rl.agents import DoubleDQNAgent, DQNAgent, PPOAgent, QLearningAgent
from traffic_rl.config import AgentConfig, EnvironmentConfig
from traffic_rl.logging import configure_logging, get_logger, silence_third_party
from traffic_rl.sumo.environment import TrafficEnvironment
from traffic_rl.utils.random import seed_everything

log = get_logger("traffic_rl.cli")


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"traffic_rl version {__version__}")
        return 0

    configure_logging(verbose=args.verbose)
    silence_third_party()

    if args.command == "train":
        return _cmd_train(args)
    elif args.command == "eval":
        return _cmd_eval(args)
    else:
        parser.print_help()
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traffic_rl",
        description="TrafficRL - RL for Traffic Signal Control",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train an RL agent")
    train.add_argument("--agent", choices=["q_learning", "dqn", "double_dqn", "ppo"],
                       default="dqn", help="Agent type")
    train.add_argument("--name", default=None, help="Agent name for logging")
    train.add_argument("--episodes", type=int, default=None, help="Number of episodes")
    train.add_argument("--max-steps", type=int, default=None, help="Max steps per episode")
    train.add_argument("--seed", type=int, default=None, help="Random seed")
    train.add_argument("--gui", action="store_true", help="Run SUMO with GUI")
    train.add_argument("--save-interval", type=int, default=25, help="Save checkpoint interval")
    train.add_argument("--log-dir", type=Path, default=None, help="Log directory")

    ev = sub.add_parser("eval", help="Evaluate a trained agent")
    ev.add_argument("--agent", choices=["q_learning", "dqn", "double_dqn", "ppo"],
                    required=True, help="Agent type")
    ev.add_argument("--model", type=Path, required=True, help="Path to model checkpoint")
    ev.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    ev.add_argument("--seed", type=int, default=None, help="Random seed")
    ev.add_argument("--gui", action="store_true", help="Run SUMO with GUI")
    ev.add_argument("--max-steps", type=int, default=None, help="Max steps per episode")

    return parser


def _make_agent(config: AgentConfig):
    kwargs = {"config": config}
    if config.kind == "q_learning":
        return QLearningAgent(**kwargs)
    elif config.kind == "dqn":
        return DQNAgent(**kwargs)
    elif config.kind == "double_dqn":
        return DoubleDQNAgent(**kwargs)
    elif config.kind == "ppo":
        return PPOAgent(**kwargs)
    else:
        raise ValueError(f"Unknown agent: {config.kind}")


def _cmd_train(args) -> int:
    overrides = {}
    if args.episodes:
        overrides["episodes"] = args.episodes
    if args.max_steps:
        overrides["max_steps"] = args.max_steps
    if args.gui:
        overrides["gui"] = True
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.log_dir:
        overrides["log_dir"] = args.log_dir
    config = resolve_config(overrides)
    env_config = EnvironmentConfig(
        config_path=config.environment.config_path,
        max_steps=config.environment.max_steps,
        gui=config.environment.gui,
        seed=config.environment.seed,
        step_length=config.environment.step_length,
        min_phase_duration=config.environment.min_phase_duration,
        max_phase_duration=config.environment.max_phase_duration,
        yellow_duration=config.environment.yellow_duration,
    )
    env = TrafficEnvironment(env_config)
    agent_name = args.name or f"train_{args.agent}"
    agent_config = AgentConfig(
        name=agent_name,
        kind=args.agent,
        model_path=config.agent_dir(agent_name) / "best_model.pth",
        hyperparams={"gamma": 0.95, "lr": 0.001, "epsilon_decay": 0.995, "epsilon_min": 0.01},
    )
    agent = _make_agent(agent_config)
    seed_everything(config.training.seed or 42)
    log.info("starting training: %s (%s)", agent.name, args.agent)
    best_reward = float("-inf")
    patience_counter = 0
    for ep in range(config.training.episodes):
        state = env.reset(seed=config.training.seed)
        episode_reward = 0.0
        done = False
        epsilon = agent.current_epsilon() if hasattr(agent, "current_epsilon") else 0.0
        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
        if hasattr(agent, "end_episode"):
            episode_reward = agent.end_episode()
        if hasattr(agent, "decay_epsilon"):
            agent.decay_epsilon()
        if episode_reward > best_reward:
            best_reward = episode_reward
            patience_counter = 0
            agent.save(agent_config.model_path)
            log.info("new best model saved (reward=%.2f)", episode_reward)
        else:
            patience_counter += 1
        if (ep + 1) % config.training.save_interval == 0:
            log.info(
                "episode %d/%d | reward=%.2f | best=%.2f | epsilon=%.3f",
                ep + 1, config.training.episodes, episode_reward,
                best_reward, agent.current_epsilon() if hasattr(agent, "current_epsilon") else 0.0,
            )
        if patience_counter >= config.training.early_stopping_patience:
            log.info("early stopping at episode %d", ep + 1)
            break
    log.info("training complete. Best reward: %.2f", best_reward)
    env.close()
    return 0


def _cmd_eval(args) -> int:
    from traffic_rl.config import EnvironmentConfig
    from traffic_rl.defaults import get_default_config

    base_cfg = get_default_config()
    env_config = EnvironmentConfig(
        config_path=base_cfg.environment.config_path,
        max_steps=args.max_steps or base_cfg.environment.max_steps,
        gui=args.gui,
        seed=args.seed,
    )
    env = TrafficEnvironment(env_config)
    agent_config = AgentConfig(
        name=f"eval_{args.agent}",
        kind=args.agent,
        model_path=args.model,
    )
    agent = _make_agent(agent_config)
    agent.load(args.model)
    if hasattr(agent, "decay_epsilon"):
        agent._epsilon = 0.0
    log.info("evaluating %s for %d episodes", args.agent, args.episodes)
    rewards = []
    for ep in range(args.episodes):
        state = env.reset(seed=args.seed + ep if args.seed is not None else None)
        episode_reward = 0.0
        done = False
        while not done:
            action = agent.act(state, epsilon=0.0)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
        rewards.append(episode_reward)
        log.info("episode %d: reward=%.2f", ep + 1, episode_reward)
    env.close()
    import numpy as np
    rewards = np.array(rewards)
    log.info("Evaluation summary: avg=%.2f std=%.2f min=%.2f max=%.2f",
             rewards.mean(), rewards.std(), rewards.min(), rewards.max())
    return 0


if __name__ == "__main__":
    sys.exit(main())
