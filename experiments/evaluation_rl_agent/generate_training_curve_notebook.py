"""
Notebook cell code to generate training reward curve.

Copy this code into a Jupyter notebook cell to generate the training curve figure.
"""

# Cell 1: Setup
import sys
sys.path.append("../../")

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Cell 2: Load TensorBoard data
# Update this path to your training run directory
run_dir = Path("../train_rl_agent/runs/para/20251209-212049")  # Change to your run directory
log_dir = run_dir / "logs"

event_acc = EventAccumulator(str(log_dir), size_guidance={"tensors": 10000})
event_acc.Reload()

# Get scalar tags
scalar_tags = event_acc.Tags()['tensors']
print(f"Available metrics: {scalar_tags}")

# Extract mean reward data
step_dict = {}
value_dict = {}

# Try mean_reward_update first, then mean_reward_all
for tag in ['mean_reward_update', 'mean_reward_all']:
    if tag in scalar_tags:
        events = event_acc.Tensors(tag)
        steps = []
        values = []
        for event in events:
            steps.append(event.step)
            values.append(tf.make_ndarray(event.tensor_proto).item())
        
        step_dict[tag] = np.array(steps)
        value_dict[tag] = np.array(values)
        print(f"Loaded {len(steps)} data points for '{tag}'")
        break

# Cell 3: Plot training curve
tag = list(step_dict.keys())[0]
steps = step_dict[tag]
values = value_dict[tag]

fig, ax = plt.subplots(figsize=(10, 6))

# Plot the curve
ax.plot(steps, values, linewidth=2, label='Mean Reward', color='#648fff')

# Optional: Add smoothed curve (moving average)
if len(values) > 10:
    window_size = max(10, len(values) // 50)
    smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
    smoothed_steps = steps[window_size-1:]
    ax.plot(smoothed_steps, smoothed, linewidth=2, alpha=0.7, 
            label=f'Smoothed (window={window_size})', linestyle='--', color='#ffb000')

ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Mean Reward', fontsize=12)
ax.set_title('Training Convergence of PPO Agent', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.tick_params(direction='in', right=True, top=True)

plt.tight_layout()

# Save figure
output_path = 'training_reward_curve.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

plt.show()



