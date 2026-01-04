# Training Parameter Suggestions for Stuck Model

## Current Issue
Model stuck at 3.7 mean return - suggests the agent is not learning effectively.

## Recent Changes (Updated Reward Function)

The reward function has been improved to:
1. **Add incremental progress rewards**: `0.05 * (spiders_removed + edges_removed)` - encourages continued simplification even when ratio plateaus
2. **Scale flow rewards proportionally**: Flow rewards now scale with base reward (up to 5% of base), making them more meaningful without dominating
3. **Reduce flow penalties**: Standard "no flow" penalty reduced from `-flow_preservation_weight` to `-0.5 * flow_preservation_weight` to allow more exploration

## Recommended Parameter Adjustments

### Option 1: Balanced Approach (Recommended)
Good balance of exploration and learning:

```bash
python runner_final.py \
    --n_envs 4 \
    --max_sample_steps 100 \
    --minibatch_size 200 \
    --total_timesteps 1e6 \
    --no-multiprocess \
    --train_iterations 5 \
    --ent_coeff 0.2 \
    --learning_rate 5e-4 \
    --flow_preservation_weight 0.1 \
    --spider_web_weight 0.1
```

### Option 2: High Exploration
If the agent is stuck in a local minimum, increase exploration:

```bash
python runner_final.py \
    --n_envs 4 \
    --max_sample_steps 100 \
    --minibatch_size 200 \
    --total_timesteps 1e6 \
    --no-multiprocess \
    --train_iterations 5 \
    --ent_coeff 0.3 \
    --learning_rate 5e-4 \
    --flow_preservation_weight 0.1 \
    --spider_web_weight 0.1
```

### Option 3: Aggressive Learning
Higher learning rate and more training iterations:

```bash
python runner_final.py \
    --n_envs 4 \
    --max_sample_steps 100 \
    --minibatch_size 200 \
    --total_timesteps 1e6 \
    --no-multiprocess \
    --train_iterations 10 \
    --ent_coeff 0.15 \
    --learning_rate 1e-3 \
    --flow_preservation_weight 0.1 \
    --spider_web_weight 0.1
```

### Option 4: Conservative (If Option 1 is too aggressive)
Slightly more conservative parameters:

```bash
python runner_final.py \
    --n_envs 4 \
    --max_sample_steps 100 \
    --minibatch_size 200 \
    --total_timesteps 1e6 \
    --no-multiprocess \
    --train_iterations 5 \
    --ent_coeff 0.15 \
    --learning_rate 3e-4 \
    --flow_preservation_weight 0.1 \
    --spider_web_weight 0.1
```

## Understanding the Updated Reward

The reward now consists of:
- **Base reward**: `(max_spiders / current_spiders) * (max_edges / current_edges)`
  - This can be large (10-100+) if simplifying well
  - Low (1-5) if not simplifying much
  
- **Progress reward**: `0.05 * (spiders_removed + edges_removed)`
  - Incremental reward for each spider/edge removed
  - Helps when ratio plateaus but there's still room for improvement
  
- **Flow reward**: Scaled with base reward (up to 5% of base)
  - `+flow_preservation_weight * min(1.0, base_reward * 0.05)` if has flow
  - `-0.5 * flow_preservation_weight * min(1.0, base_reward * 0.05)` if no flow (never had it)
  - `-1.5 * flow_preservation_weight * min(1.0, base_reward * 0.05)` if lost flow
  
- **Spider web reward**: Small contribution from reducing spider webs

A mean return of 3.7 suggests:
- The model is simplifying diagrams to ~60-70% of original size
- The new progress reward should help push it further
- Increased exploration and learning rate should help escape the plateau

## Diagnostic Steps

1. **Check TensorBoard**: Look at:
   - `mean_reward_update`: Is it increasing or flat?
   - `losses/pol_grad_loss`: Is it decreasing?
   - `entropy_loss`: Is exploration happening?
   - `kl`: Is the policy changing?

2. **Check if flow is being preserved**: If flow is lost frequently, the penalties might dominate.

3. **Try without flow penalty first**: Set `--flow_preservation_weight 0` to see if the base reward improves, then gradually increase it.


