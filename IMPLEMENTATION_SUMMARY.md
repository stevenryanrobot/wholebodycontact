# Teleoperation System Implementation Summary

## Overview

A complete teleoperation system has been implemented for the G1 humanoid robot. This system allows controlling a trained policy using VR/teleoperation input while preserving the original training code and architecture.

## Key Features

✅ **Policy Compatibility**
- Same observation/action spaces as training
- Uses same `make_env_policy()` function
- No modifications to training code required

✅ **Isolated Environment**
- Separate `ManipulationEnv` class
- Does not affect training pipeline
- Can be extended independently

✅ **UDP-Based Teleoperation**
- VR input via UDP protocol
- Port 15000 (configurable)
- 4 bodies: root, head, left hand, right hand

✅ **Complete Inference System**
- Hydra-based configuration
- AppLauncher for graphics/physics
- VecNorm observation normalization

## Files Created

### 1. Configuration Files

**`cfg/task/G1/G1_gentle_manipulation.yaml`**
```yaml
# Inherits from G1_gentle training task
# Replaces command system with TeleopCommand
# Specifies UDP bind_port: 15000
```

**`cfg/manipulation.yaml`**
```yaml
# Main inference configuration
# Uses G1_gentle_manipulation task
# Specifies app settings and vecnorm
```

### 2. Environment & Command System

**`active_adaptation/envs/manipulation.py`**
- `ManipulationEnv(SimpleEnv)`
  - Inherits from SimpleEnv (training environment)
  - Replaces command system with TeleopCommand
  - Maintains complete policy compatibility
  - Methods: `reset()`, `step()`, `get_object_state()`, `get_command_manager()`

**`active_adaptation/envs/mdp/commands/teleoperation.py`**
- `UdpTeleopReceiver`
  - Listens on UDP port 15000
  - Receives VR command packets
  - Stores latest command in CPU tensors
- `TeleopCommand(Command)`
  - Extends base Command interface
  - Converts UDP input to target poses
  - Expands single command to all N environments
  - Stores 3 sets: root, head, left/right hands

**`active_adaptation/envs/__init__.py` (Updated)**
- Added import: `from .manipulation import ManipulationEnv`

### 3. Inference Scripts

**`scripts/eval_manipulation.py`**
- Main evaluation script
- Uses Hydra decorator: `@hydra.main()`
- Uses AppLauncher for graphics/physics
- Function: `eval_manipulation(cfg, checkpoint_path, num_episodes, max_steps)`
- Features:
  - Policy loading with `make_env_policy()`
  - Episode loop with UDP teleoperation input
  - VecNorm observation normalization
  - Statistics logging
  - Ctrl+C interrupt handling

**`scripts/utils/teleop_client.py` (New Utility)**
- Standalone teleoperation client
- Classes:
  - `TeleopClient`: UDP sender for commands
  - Methods: `send_command()`, `send_idle()`, `close()`
- Demos: standing, reaching, walking
- Interactive mode for manual control
- Usage: `python scripts/utils/teleop_client.py --demo interactive`

### 4. Documentation

**`TELEOPERATION_GUIDE.md`**
- Quick start guide
- Command format specification
- Architecture overview
- Troubleshooting tips
- Example Python code

**`IMPLEMENTATION_SUMMARY.md` (This File)**
- Overview of all components
- File descriptions
- Usage instructions
- Technical details

## UDP Protocol

### Packet Format
```
[MAGIC_BYTE: 1 byte]
[SEQUENCE: 4 bytes, big-endian]
[BODY_0: 28 bytes] (root)
[BODY_1: 28 bytes] (head)
[BODY_2: 28 bytes] (left_hand)
[BODY_3: 28 bytes] (right_hand)

Total: 1 + 4 + 4*28 = 117 bytes per packet
```

### Body Data Format
```
[Position: 3 floats] (x, y, z)
[Quaternion: 4 floats] (x, y, z, w)
Total: 28 bytes per body
```

### Body Indices
- **BODY_0**: Root (hips)
- **BODY_1**: Head
- **BODY_2**: Left Hand
- **BODY_3**: Right Hand

## Usage

### 1. Start Evaluation with Trained Model

```bash
python scripts/eval_manipulation.py \
  --checkpoint outputs/2026-01-16/10-26-24-G1GENTLE-ppo/checkpoints/model.pt \
  --num-episodes 3 \
  --max-steps 1000
```

### 2. Send Teleoperation Commands (In Another Terminal)

**Option A: Using teleop_client.py**
```bash
python scripts/utils/teleop_client.py --demo standing
python scripts/utils/teleop_client.py --demo interactive
```

**Option B: Custom Python Script**
```python
from scripts.utils.teleop_client import TeleopClient

client = TeleopClient(host="127.0.0.1", port=15000)
for i in range(100):
    client.send_command(
        root_pos=(0.0, 0.0, 0.0),
        root_quat=(0.0, 0.0, 0.0, 1.0),
        head_pos=(0.0, 0.0, 0.25),
        head_quat=(0.0, 0.0, 0.0, 1.0),
        left_hand_pos=(0.15, 0.1, 0.0),
        left_hand_quat=(0.0, 0.0, 0.0, 1.0),
        right_hand_pos=(0.15, -0.1, 0.0),
        right_hand_quat=(0.0, 0.0, 0.0, 1.0),
    )
    time.sleep(0.02)
client.close()
```

## Architecture Diagram

```
Training (Unchanged)
  ├── scripts/train.py
  ├── cfg/train.yaml
  └── active_adaptation/envs/locomotion.py (SimpleEnv)

Teleoperation (New)
  ├── scripts/eval_manipulation.py
  ├── cfg/manipulation.yaml
  ├── cfg/task/G1/G1_gentle_manipulation.yaml
  ├── active_adaptation/envs/manipulation.py (ManipulationEnv)
  ├── active_adaptation/envs/mdp/commands/teleoperation.py (TeleopCommand)
  └── scripts/utils/teleop_client.py (UDP sender)

Shared
  ├── scripts/utils/helpers.py (make_env_policy)
  ├── active_adaptation/learning/ (Policy networks)
  └── outputs/ (Trained checkpoints)
```

## Component Interactions

### Initialization Sequence

```
eval_manipulation.py
  └─> make_env_policy(cfg)
      ├─> Load config: cfg/manipulation.yaml
      ├─> Create env: ManipulationEnv(cfg)
      │   ├─> Load parent: SimpleEnv.__init__()
      │   ├─> Replace command: TeleopCommand(cfg)
      │   └─> Create receiver: UdpTeleopReceiver(port=15000)
      ├─> Load checkpoint: model.pt
      └─> Create vecnorm: VecNorm(obs_dim)
```

### Inference Loop

```
Episode:
  reset()
    └─> env.reset()
        ├─> TeleopCommand.reset()
        └─> UdpTeleopReceiver listens...

  for step in steps:
    obs_norm = vecnorm("policy", obs)
    action = agent(obs_norm)
    
    step(action)
      ├─> TeleopCommand.step() // Updates from UDP
      └─> Physics simulation
```

## Testing Checklist

- [ ] Verify eval_manipulation.py imports correctly
- [ ] Load trained checkpoint successfully
- [ ] TeleopCommand starts UDP receiver
- [ ] Send test command via teleop_client.py
- [ ] Verify robot responds to teleoperation input
- [ ] Check observation normalization
- [ ] Verify episode statistics logging
- [ ] Test interrupt handling (Ctrl+C)

## Future Enhancements

### Phase 1: Object Interaction
- [ ] Load objects into scene (boxes, tables)
- [ ] Add contact sensors
- [ ] Implement grasping detection
- [ ] Add object-state to observations

### Phase 2: Advanced Teleoperation
- [ ] VR hand controller interface
- [ ] Full-body inverse kinematics
- [ ] Haptic feedback
- [ ] Motion capture integration

### Phase 3: Learning from Demonstrations
- [ ] Record teleoperation trajectories
- [ ] Fine-tune policy on demonstrations
- [ ] Implement behavioral cloning
- [ ] Create demonstration dataset

## Known Limitations

1. **No Autonomous Planning**: Requires continuous teleoperation input
2. **No Object Interaction**: Placeholder methods only
3. **Single Humanoid**: One robot per environment
4. **UDP-Only**: No TCP fallback
5. **No Error Correction**: Lost packets cause brief command delay

## Troubleshooting

### "Port already in use"
```bash
# Kill existing process
lsof -ti:15000 | xargs kill -9
```

### "Module not found" errors
```bash
# Ensure project root is in Python path
export PYTHONPATH=/home/xl521/ee-gentle-humanoid:$PYTHONPATH
```

### "CUDA out of memory"
```bash
# Run on CPU or with fewer environments
python scripts/eval_manipulation.py --checkpoint ... --num-episodes 1
```

## References

- **Training Script**: `scripts/train.py`
- **Original Eval**: `scripts/eval.py`
- **Configuration System**: `cfg/`
- **Environment Base**: `active_adaptation/envs/`
- **Command Base**: `active_adaptation/envs/mdp/`
- **Helper Functions**: `scripts/utils/helpers.py`

## Contact & Support

For issues or questions about the teleoperation system:
1. Check `TELEOPERATION_GUIDE.md` for quick troubleshooting
2. Review this implementation summary
3. Check UDP protocol format and timing
4. Verify checkpoint path and config loading
