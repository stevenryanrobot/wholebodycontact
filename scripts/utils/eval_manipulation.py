"""
Utility functions for manipulation task evaluation.

Provides helper functions specific to manipulation tasks.
"""

import torch
from typing import Dict, Tuple
from pathlib import Path


def load_manipulation_checkpoint(checkpoint_path: str):
    """
    Load a manipulation task checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        dict: Checkpoint data
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


def get_grasping_action(action: torch.Tensor, gripper_open_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split action into arm control and gripper control.
    
    Args:
        action: Full action tensor
        gripper_open_threshold: Threshold for gripper open/close
        
    Returns:
        arm_action: Action for arm joints
        gripper_action: Action for gripper (open/close)
    """
    # This is a placeholder - adjust based on your actual action space
    # Example: if last action dimension is gripper
    arm_action = action[..., :-1]
    gripper_action = (action[..., -1] > gripper_open_threshold).float()
    
    return arm_action, gripper_action


def compute_object_distance(robot_ee_pos: torch.Tensor, object_pos: torch.Tensor) -> torch.Tensor:
    """
    Compute distance between robot end-effector and object.
    
    Args:
        robot_ee_pos: End-effector position [batch, 3]
        object_pos: Object position [batch, 3]
        
    Returns:
        torch.Tensor: Distance [batch]
    """
    return torch.norm(robot_ee_pos - object_pos, dim=-1)


def check_object_grasped(gripper_contact_force: torch.Tensor, force_threshold: float = 1.0) -> torch.Tensor:
    """
    Check if object is grasped based on contact forces.
    
    Args:
        gripper_contact_force: Contact force [batch, 2] (left, right fingers)
        force_threshold: Minimum force to consider grasped
        
    Returns:
        torch.Tensor: Boolean tensor indicating grasp [batch]
    """
    # Both fingers must have contact
    grasped = (gripper_contact_force > force_threshold).all(dim=-1)
    return grasped


def log_manipulation_metrics(
    episode_num: int,
    step: int,
    reward: torch.Tensor,
    object_dist: torch.Tensor = None,
    is_grasped: torch.Tensor = None,
):
    """
    Log manipulation task metrics.
    
    Args:
        episode_num: Episode number
        step: Step number
        reward: Current reward
        object_dist: Distance to object (optional)
        is_grasped: Whether object is grasped (optional)
    """
    log_str = f"[Ep {episode_num:3d}, Step {step:4d}] Reward: {reward.mean():.4f}"
    
    if object_dist is not None:
        log_str += f" | Dist: {object_dist.mean():.3f}"
    
    if is_grasped is not None:
        grasp_rate = is_grasped.float().mean().item()
        log_str += f" | Grasp: {grasp_rate:.1%}"
    
    print(log_str)
