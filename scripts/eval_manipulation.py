"""
Evaluation script for manipulation tasks with teleoperation.

This script loads a trained policy and runs inference in the manipulation environment.
The policy and environment remain compatible - observation/action spaces are unchanged.

Usage (wandb):
    python scripts/eval_manipulation.py --run_path luoxinyuan-duke-university/gentle_humanoid/run_name -p
    
Usage (local checkpoint):
    python scripts/eval_manipulation.py --checkpoint outputs/xxx/model.pt
"""

import torch
import wandb
import hydra
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from omegaconf import OmegaConf, DictConfig
from isaaclab.app import AppLauncher
from scripts.utils.play import play

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")


def main():
    parser = argparse.ArgumentParser(description="Manipulation evaluation with teleoperation")
    parser.add_argument("-r", "--run_path", type=str, help="WandB run path")
    parser.add_argument("--checkpoint", type=str, help="Local checkpoint path (alternative to wandb)")
    parser.add_argument("--task", type=str, default=None, help="Override task config")
    parser.add_argument("-p", "--play", action="store_true", default=False, help="Play mode (visualize)")
    parser.add_argument("-i", "--iterations", type=int, default=None, help="Checkpoint iteration to load")
    parser.add_argument("-n", "--num_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("-e", "--export", action="store_true", default=False, help="Export policy")
    parser.add_argument("--objects", type=str, default=None, 
                        help="Objects config file (e.g., cfg/objects/test_scene.yaml)")
    parser.add_argument("--full_collision", action="store_true", default=False,
                        help="Use robot USD with full collision meshes (g1_col_full)")
    parser.add_argument("--obs_source", choices=["udp", "motion"], default="udp",
                        help="Observation source for command/root_and_wrist_6d in play mode")
    args = parser.parse_args()

    # Determine checkpoint source
    if args.run_path:
        # Load from wandb
        api = wandb.Api()
        run = api.run(args.run_path)
        print(f"Loading run: {run.name}")

        root = os.path.join(os.path.dirname(__file__), "wandb", run.name)
        os.makedirs(root, exist_ok=True)

        # Download config and checkpoints
        checkpoints = []
        for file in run.files():
            print(file.name)
            if "checkpoint" in file.name:
                checkpoints.append(file)
            elif file.name == "cfg.yaml":
                file.download(root, replace=True)
            elif file.name == "files/cfg.yaml":
                file.download(root, replace=True)
            elif file.name == "config.yaml":
                file.download(root, replace=True)
        
        # Select checkpoint
        if args.iterations is None:
            def sort_by_time(file):
                number_str = file.name[:-3].split("_")[-1]
                if number_str == "final":
                    return 100000
                else:
                    return int(number_str)
            checkpoints.sort(key=sort_by_time)
            checkpoint = checkpoints[-1]
        else:
            for file in checkpoints:
                if file.name == f"checkpoint_{args.iterations}.pt":
                    checkpoint = file
                    break
        
        print(f"Downloading {checkpoint.name}")
        checkpoint.download(root, replace=True)

        # Load config
        try:
            cfg = OmegaConf.load(os.path.join(root, "files", "cfg.yaml"))
        except FileNotFoundError:
            cfg = OmegaConf.load(os.path.join(root, "cfg.yaml"))
        OmegaConf.set_struct(cfg, False)

        cfg["checkpoint_path"] = os.path.join(root, checkpoint.name)
        cfg["vecnorm"] = "eval"

    elif args.checkpoint:
        # Load from local checkpoint with default config
        with hydra.initialize(config_path="../cfg", job_name="eval_manipulation", version_base=None):
            cfg = hydra.compose(config_name="eval", overrides=[])
        OmegaConf.set_struct(cfg, False)
        cfg["checkpoint_path"] = args.checkpoint
        cfg["vecnorm"] = "eval"
    else:
        print("Error: Must specify either --run_path or --checkpoint")
        sys.exit(1)

    # Note: UdpTeleopReceiver is already started by default in MotionTrackingCommand
    # The robot will listen on UDP port 15000 for teleoperation commands

    # Override task config if specified
    if args.task is not None:
        with hydra.initialize(config_path="../cfg", job_name="eval_manipulation", version_base=None):
            _cfg = hydra.compose(config_name="eval", overrides=[f"task={args.task}"])
        cfg["task"]["reward"] = _cfg.task.reward
        cfg["task"]["termination"] = _cfg.task.termination
        cfg["task"]["observation"] = _cfg.task.observation
        cfg["task"]["action"] = _cfg.task.action
        cfg["task"]["randomization"] = _cfg.task.randomization
        cfg["task"]["robot"] = _cfg.task.robot
        cfg["task"]["command"] = _cfg.task.command
        cfg["task"]["flags"] = _cfg.task.flags

    # Play mode settings
    if args.play:
        cfg["app"]["headless"] = False
        cfg["task"]["num_envs"] = args.num_envs
        cfg["task"]["max_episode_length"] = 1000000  # Very long episode for teleop (no auto-reset)
        # Disable all termination conditions for teleop mode
        cfg["task"]["termination"] = {}
        # Disable motion finish triggering reset only for live teleop mode.
        cfg["task"]["command"]["disable_motion_finish"] = args.obs_source == "udp"
        cfg["task"]["command"]["teleop"] = {
            "enabled": args.obs_source == "udp",
            "obs_source": args.obs_source,
        }
        cfg["export_policy"] = args.export
        cfg["perf_test"] = False
        
        # Disable all init noise for consistent starting pose in teleop mode
        cfg["task"]["command"]["init_noise"] = {
            "root_pos": 0.0,
            "root_ori": 0.0,
            "root_lin_vel": 0.0,
            "root_ang_vel": 0.0,
            "joint_pos": 0.0,
            "joint_vel": 0.0,
        }
        
        # Use full collision robot USD if requested
        if args.full_collision:
            cfg["task"]["robot"]["name"] = "g1_col_full"
            print("  Robot: Using full collision USD (g1_col_full)")
        
        # Load objects configuration
        if args.objects:
            objects_cfg = OmegaConf.load(args.objects)
            cfg["task"]["objects"] = objects_cfg.get("objects", [])
            print(f"  Objects: Loaded {len(cfg['task']['objects'])} objects from {args.objects}")
        
        print("\n" + "="*60)
        print("MANIPULATION TASK (Teleoperation Mode)")
        print("="*60)
        print(f"  Run: {args.run_path or args.checkpoint}")
        print(f"  Num envs: {args.num_envs}")
        print(f"  Max episode length: 1000000 steps (~5.5 hours at 50Hz)")
        if args.obs_source == "udp":
            print(f"  Obs source: UDP teleop (waiting for input on port 15000)")
        else:
            print(f"  Obs source: motion dataset")
        if args.objects:
            print(f"  Objects: {args.objects}")
        print("="*60 + "\n")
        
        play(cfg)
    else:
        print("Error: Currently only play mode (-p) is supported for manipulation")
        sys.exit(1)
    
    exit(0)


if __name__ == "__main__":
    main()
