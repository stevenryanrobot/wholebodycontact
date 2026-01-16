"""
Teleoperation command system for manipulation tasks.

Provides command interface compatible with VR input (UDP) for robot control.
Similar to motion_tracking.py but simplified for direct teleoperation.
"""

import torch
import numpy as np
from typing import Optional
import threading
import socket
import struct
import time

from active_adaptation.envs.mdp.commands.base import Command


# UDP protocol for VR teleoperation
MAGIC = 0x12345678
PACK_FMT = f"=II{7*4}f"  # magic, seq, 4 bodies * (pos3 + quat4)
PACK_SIZE = struct.calcsize(PACK_FMT)


class UdpTeleopReceiver:
    """
    Receive UDP packets: root/head/left/right, each (pos3 + quat4) in WORLD frame.
    Thread updates latest sample (CPU tensors).
    """
    def __init__(self, bind_ip="0.0.0.0", bind_port=15000, timeout=0.2):
        self.bind_ip = bind_ip
        self.bind_port = bind_port
        self.timeout = timeout

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._seq: int = -1
        self._t_recv: float = 0.0

        # store latest as torch CPU tensors
        self._root_pos = torch.zeros(3, dtype=torch.float32)
        self._root_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        self._head_pos = torch.zeros(3, dtype=torch.float32)
        self._head_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        self._l_pos = torch.zeros(3, dtype=torch.float32)
        self._l_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        self._r_pos = torch.zeros(3, dtype=torch.float32)
        self._r_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.bind_ip, self.bind_port))
        sock.settimeout(self.timeout)

        while self._running:
            try:
                data, _ = sock.recvfrom(2048)
                if len(data) != PACK_SIZE:
                    continue
                magic, seq, *floats = struct.unpack(PACK_FMT, data)
                if magic != MAGIC:
                    continue

                # 4 bodies * 7 floats
                vals = torch.tensor(floats, dtype=torch.float32)  # shape (28,)
                root = vals[0:7]
                head = vals[7:14]
                left = vals[14:21]
                right = vals[21:28]

                with self._lock:
                    self._seq = int(seq)
                    self._t_recv = time.time()

                    self._root_pos = root[0:3].clone()
                    self._root_quat = root[3:7].clone()
                    self._head_pos = head[0:3].clone()
                    self._head_quat = head[3:7].clone()
                    self._l_pos = left[0:3].clone()
                    self._l_quat = left[3:7].clone()
                    self._r_pos = right[0:3].clone()
                    self._r_quat = right[3:7].clone()

            except socket.timeout:
                continue
            except Exception:
                continue

    def get_latest(self):
        """
        Returns:
            (seq, t_recv, root_pos, root_quat, head_pos, head_quat, l_pos, l_quat, r_pos, r_quat)
        """
        with self._lock:
            return (
                self._seq,
                self._t_recv,
                self._root_pos.clone(),
                self._root_quat.clone(),
                self._head_pos.clone(),
                self._head_quat.clone(),
                self._l_pos.clone(),
                self._l_quat.clone(),
                self._r_pos.clone(),
                self._r_quat.clone(),
            )


class TeleopCommand(Command):
    """
    Teleoperation command system for manipulation tasks.
    
    Receives target poses for root, head, and hands from VR input (UDP).
    Converts to observations and rewards compatible with the trained policy.
    """

    def __init__(self, env, bind_port: int = 15000):
        """
        Initialize teleoperation command system.
        
        Args:
            env: Base environment
            bind_port: UDP port for receiving VR input
        """
        super().__init__(env)
        
        self.bind_port = bind_port
        self._teleop = UdpTeleopReceiver(bind_port=bind_port)
        self._teleop.start()
        
        # Store latest target poses [N, 3] and [N, 4]
        self._target_root_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_root_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_root_quat[:, -1] = 1.0  # identity quaternion
        
        self._target_head_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_head_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_head_quat[:, -1] = 1.0
        
        self._target_l_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_l_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_l_quat[:, -1] = 1.0
        
        self._target_r_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_r_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_r_quat[:, -1] = 1.0

    def update(self):
        """
        Update target poses from UDP input.
        Called once per environment step.
        """
        # Get latest UDP data
        seq, t_recv, root_pos, root_quat, head_pos, head_quat, l_pos, l_quat, r_pos, r_quat = self._teleop.get_latest()
        
        # Convert to device and expand to all environments
        root_pos = root_pos.to(self.device)
        root_quat = root_quat.to(self.device)
        head_pos = head_pos.to(self.device)
        head_quat = head_quat.to(self.device)
        l_pos = l_pos.to(self.device)
        l_quat = l_quat.to(self.device)
        r_pos = r_pos.to(self.device)
        r_quat = r_quat.to(self.device)
        
        # All environments receive the same command from VR
        self._target_root_pos[:] = root_pos.unsqueeze(0)
        self._target_root_quat[:] = root_quat.unsqueeze(0)
        self._target_head_pos[:] = head_pos.unsqueeze(0)
        self._target_head_quat[:] = head_quat.unsqueeze(0)
        self._target_l_pos[:] = l_pos.unsqueeze(0)
        self._target_l_quat[:] = l_quat.unsqueeze(0)
        self._target_r_pos[:] = r_pos.unsqueeze(0)
        self._target_r_quat[:] = r_quat.unsqueeze(0)

    def reset(self, env_ids: torch.Tensor):
        """Reset command state."""
        self._target_root_pos[env_ids] = 0.0
        self._target_root_quat[env_ids, :3] = 0.0
        self._target_root_quat[env_ids, 3] = 1.0

    def __del__(self):
        """Cleanup UDP receiver thread."""
        if hasattr(self, '_teleop'):
            self._teleop.stop()
