"""
Evaluation script for manipulation tasks.

This script loads a trained policy and runs inference in the manipulation environment.
The policy and environment remain compatible - observation/action spaces are unchanged.

Usage:
    python scripts/eval_manipulation.py --checkpoint outputs/xxx/model.pt --task G1_gentle
"""

import torch
import hydra
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from omegaconf import OmegaConf, DictConfig
from isaaclab.app import AppLauncher
from scripts.utils.helpers import make_env_policy

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")


def eval_manipulation(cfg, checkpoint_path=None, num_episodes=5, max_steps=1000):
    """
    Run manipulation task evaluation with trained policy.
    
    Args:
        cfg: Hydra configuration
        checkpoint_path: Path to model checkpoint. If None, uses cfg.checkpoint_path
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
    """
    print("\n" + "="*60)
    print("MANIPULATION TASK EVALUATION (Teleoperation Mode)")
    print("="*60)
    
    # Load policy and environment
    print(f"\n[1/4] Loading checkpoint...")
    if checkpoint_path:
        cfg.checkpoint_path = checkpoint_path
    
    try:
        env, agent, vecnorm, cfg = make_env_policy(cfg)
    except Exception as e:
        print(f"❌ Error loading policy: {e}")
        raise
    
    print(f"✅ Loaded: {type(env.unwrapped).__name__}")
    print(f"   Policy: {type(agent).__name__}")
    print(f"   Num envs: {env.num_envs}")
    print(f"   UDP Port: 15000 (waiting for VR input...)")
    
    # Run episodes
    print(f"\n[2/4] Running {num_episodes} episodes...")
    print("   Press Ctrl+C to stop, or send VR teleoperation commands via UDP")
    
    episode_rewards = []
    episode_lengths = []
    
    try:
        for episode in range(num_episodes):
            print(f"\n  Episode {episode+1}/{num_episodes}:")
            
            # Reset environment
            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            # Run episode
            for step in range(max_steps):
                # Normalize observations
                obs_norm = vecnorm("policy", obs)
                
                # Get action from policy
                with torch.no_grad():
                    action = agent(obs_norm)
                
                # Step environment
                obs, reward, done, info = env.step(action)
                
                episode_reward += reward.mean().item()
                episode_length += 1
                
                # Check termination
                if done.any():
                    print(f"    Step {step+1}: Episode terminated")
                    break
            
            # Log episode stats
            avg_reward = episode_reward / episode_length if episode_length > 0 else 0
            episode_rewards.append(avg_reward)
            episode_lengths.append(episode_length)
            
            print(f"    Length: {episode_length}, Avg Reward: {avg_reward:.4f}")
            
            # Get object state if available
            if hasattr(env.unwrapped, 'get_object_state'):
                obj_state = env.unwrapped.get_object_state()
                if obj_state:
                    print(f"    Objects: {obj_state}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    
    finally:
        # Summary statistics
        if episode_rewards:
            print(f"\n[3/4] Summary:")
            print(f"  Episodes: {len(episode_rewards)}")
            print(f"  Avg Length: {sum(episode_lengths)/len(episode_lengths):.1f}")
            print(f"  Avg Reward: {sum(episode_rewards)/len(episode_rewards):.4f}")
            print(f"  Min Reward: {min(episode_rewards):.4f}")
            print(f"  Max Reward: {max(episode_rewards):.4f}")
        
        print(f"\n[4/4] Evaluation complete!")
        print("="*60 + "\n")
    
    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }


@hydra.main(config_path=CONFIG_PATH, config_name="manipulation", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for script execution."""
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    # Setup AppLauncher
    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    
    # Parse command line arguments for evaluation-specific settings
    parser = argparse.ArgumentParser(description="Run manipulation task evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode",
    )
    
    args = parser.parse_known_args()[0]
    
    # Override config if checkpoint provided
    if args.checkpoint:
        cfg.checkpoint_path = args.checkpoint
    
    # Run evaluation with AppLauncher
    with app_launcher.context():
        results = eval_manipulation(
            cfg,
            checkpoint_path=args.checkpoint,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()
