"""
Generate training reward curve figure from TensorBoard logs.

Usage:
    python generate_training_curve.py [--run_dir <path>] [--output <filename>]

If no arguments provided, uses the most recent run in runs/para/
"""
import sys
sys.path.append("../../")

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_data(log_dir):
    """Load scalar data from TensorBoard event files."""
    event_acc = EventAccumulator(str(log_dir), size_guidance={"tensors": 10000, "scalars": 10000})
    event_acc.Reload()
    
    # Get all available tags
    tags = event_acc.Tags()
    tensor_tags = tags.get('tensors', [])
    scalar_tags = tags.get('scalars', [])
    
    # Extract data for mean_reward_update (or mean_reward_all as fallback)
    step_dict = {}
    value_dict = {}
    
    # Try mean_reward_update first, then mean_reward_all
    for tag in ['mean_reward_update', 'mean_reward_all']:
        # Try Tensors first (TensorFlow 2.x format)
        if tag in tensor_tags:
            events = event_acc.Tensors(tag)
            steps = []
            values = []
            for event in events:
                steps.append(event.step)
                values.append(tf.make_ndarray(event.tensor_proto).item())
            
            step_dict[tag] = np.array(steps)
            value_dict[tag] = np.array(values)
            print(f"Loaded {len(steps)} data points for '{tag}' (from Tensors)")
            break
        # Fallback to Scalars (older TensorBoard format)
        elif tag in scalar_tags:
            scalar_events = event_acc.Scalars(tag)
            steps = []
            values = []
            for event in scalar_events:
                steps.append(event.step)
                values.append(event.value)
            
            step_dict[tag] = np.array(steps)
            value_dict[tag] = np.array(values)
            print(f"Loaded {len(steps)} data points for '{tag}' (from Scalars)")
            break
    
    if not step_dict:
        print(f"Available tensor tags: {tensor_tags}")
        print(f"Available scalar tags: {scalar_tags}")
        raise ValueError("Could not find 'mean_reward_update' or 'mean_reward_all' in logs")
    
    return step_dict, value_dict


def plot_training_curve(step_dict, value_dict, output_path=None, figsize=(10, 6)):
    """Plot training reward curve."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use the first available metric
    tag = list(step_dict.keys())[0]
    steps = step_dict[tag]
    values = value_dict[tag]
    
    # Plot the curve
    ax.plot(steps, values, linewidth=2, label='Mean Reward')
    
    # Optional: Add smoothed curve (moving average)
    if len(values) > 10:
        window_size = max(10, len(values) // 50)  # Adaptive window size
        smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
        smoothed_steps = steps[window_size-1:]
        ax.plot(smoothed_steps, smoothed, linewidth=2, alpha=0.7, 
                label=f'Smoothed (window={window_size})', linestyle='--')
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Training Convergence of PPO Agent', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.tick_params(direction='in', right=True, top=True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()
    
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description='Generate training reward curve')
    parser.add_argument('--run_dir', type=str, default=None,
                       help='Path to training run directory (e.g., runs/para/20251209-212049)')
    parser.add_argument('--output', type=str, default='training_reward_curve.png',
                       help='Output filename for the figure')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'pdf', 'svg'],
                       help='Output format (png, pdf, or svg)')
    
    args = parser.parse_args()
    
    # Find log directory
    if args.run_dir:
        log_dir = Path(args.run_dir) / 'logs'
    else:
        # Use most recent run
        runs_dir = Path('../../experiments/train_rl_agent/runs/para')
        if not runs_dir.exists():
            runs_dir = Path('../train_rl_agent/runs/para')
        
        runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], 
                     key=lambda x: x.name, reverse=True)
        
        if not runs:
            raise ValueError(f"No training runs found in {runs_dir}")
        
        log_dir = runs[0] / 'logs'
        print(f"Using most recent run: {runs[0].name}")
    
    if not log_dir.exists():
        raise ValueError(f"Log directory not found: {log_dir}")
    
    print(f"Loading TensorBoard logs from: {log_dir}")
    
    # Load data
    step_dict, value_dict = load_tensorboard_data(log_dir)
    
    # Determine output path
    if args.output.endswith(('.png', '.pdf', '.svg')):
        output_path = args.output
    else:
        output_path = f"{args.output}.{args.format}"
    
    # Generate plot
    plot_training_curve(step_dict, value_dict, output_path=output_path)
    
    print("Done!")


if __name__ == '__main__':
    main()

