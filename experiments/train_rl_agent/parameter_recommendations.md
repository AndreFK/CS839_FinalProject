# Parameter Recommendations for Low Mean Return (4.0)

## Current Issue
- Mean return: ~4.0 (very low)
- Episodic return: ~397 (suggests long episodes but low per-step reward)
- Agent is not simplifying diagrams effectively

## Problem Analysis

With your current parameters:
- `flow_preservation_weight=0.8` is **too high** - making agent overly cautious
- `learning_rate=2e-4` might be too conservative
- `minibatch_size=200` might be too small for effective learning
- Base reward of ~4.0 suggests only ~50% simplification (e.g., 10→5 spiders, 15→7.5 edges)

## Recommended Parameter Sets

### Option 1: Balanced (Recommended First)
```bash
python runner_final.py \
    --n_envs 10 \
    --max_sample_steps 100 \
    --minibatch_size 500 \
    --total_timesteps 1e6 \
    --no-multiprocess \
    --train_iterations 5 \
    --ent_coeff 0.2 \
    --learning_rate 5e-4 \
    --flow_preservation_weight 0.1 \
    --spider_web_weight 0.1 \
    --resetter brickwork
```

**Changes:**
- `flow_preservation_weight`: 0.8 → 0.1 (reduce flow penalty dominance)
- `spider_web_weight`: 0.4 → 0.1 (balance all reward components)
- `learning_rate`: 2e-4 → 5e-4 (faster learning)
- `minibatch_size`: 200 → 500 (better gradient estimates)

**Expected:** Mean return should reach 20-50 within 100-200 epochs

### Option 2: More Aggressive Learning
```bash
python runner_final.py \
    --n_envs 10 \
    --max_sample_steps 100 \
    --minibatch_size 500 \
    --total_timesteps 1e6 \
    --no-multiprocess \
    --train_iterations 10 \
    --ent_coeff 0.3 \
    --learning_rate 1e-3 \
    --flow_preservation_weight 0.05 \
    --spider_web_weight 0.1 \
    --resetter brickwork
```

**Changes:**
- `learning_rate`: 2e-4 → 1e-3 (much faster learning)
- `ent_coeff`: 0.2 → 0.3 (more exploration)
- `train_iterations`: 5 → 10 (more gradient updates per epoch)
- `flow_preservation_weight`: 0.8 → 0.05 (minimal flow penalty)

**Expected:** Mean return should reach 30-80 within 50-100 epochs

### Option 3: Conservative (If Option 1 is too aggressive)
```bash
python runner_final.py \
    --n_envs 10 \
    --max_sample_steps 100 \
    --minibatch_size 250 \
    --total_timesteps 1e6 \
    --no-multiprocess \
    --train_iterations 5 \
    --ent_coeff 0.15 \
    --learning_rate 3e-4 \
    --flow_preservation_weight 0.2 \
    --spider_web_weight 0.15 \
    --resetter brickwork
```

**Note:** minibatch_size=250 works because batch_size=1000 (10 envs × 100 steps) is divisible by 250

**Changes:**
- Moderate adjustments from current settings
- `flow_preservation_weight`: 0.8 → 0.2 (still significant but not dominant)

**Expected:** Mean return should reach 15-40 within 150-300 epochs

## Why These Changes?

1. **Reduce Flow Weight (0.8 → 0.1)**: 
   - Current: Flow penalties dominate, agent is too cautious
   - With base_reward=4.0, flow component is small but penalties discourage exploration
   - Lower weight allows agent to explore simplification strategies

2. **Increase Learning Rate (2e-4 → 5e-4 or 1e-3)**:
   - Current rate is too conservative for the reward scale
   - Faster learning helps escape local optima

3. **Increase Minibatch Size (200 → 500)**:
   - Better gradient estimates
   - More stable training
   - Note: Must satisfy `(n_envs * max_sample_steps) % minibatch_size == 0`
   - With n_envs=10, max_sample_steps=100: batch_size=1000, so minibatch_size=500 works (1000 % 500 == 0)

4. **Balance Spider-Web Weight (0.4 → 0.1)**:
   - Current weight might be too high relative to base reward
   - Balance all components for better learning

## Expected Progress

**Current State:**
- Mean return: ~4.0
- Episodic return: ~397
- Per-step reward: ~4.0 (very low)

**Target State (after 200-500 epochs):**
- Mean return: 30-100
- Episodic return: 1000-3000
- Per-step reward: 10-30 (good simplification)

## Monitoring

Watch for:
- **Mean return increasing**: Should see gradual increase from 4.0 → 10 → 20 → 30+
- **Episodic return increasing**: Should see 397 → 500 → 800 → 1000+
- **Consistent improvement**: Should see steady upward trend, not stuck at same value

If mean return stays below 10 after 300 epochs, try Option 2 (more aggressive).

