## Jupyter Notebooks from the Docker
```
# port forwarding
ssh -L localhost:8889:REMOTE:8889 user@REMOTE

# docker with port forwarding
docker run -it --gpus all --runtime nvidia -p 8889:8889 NAME /bin/bash -c "cd zxreinforce_small; jupyter notebook --no-browser --ip=0.0.0.0 --port=8889 --allow-root"

# in the Docker (or start the docker with this command)
jupyter notebook --no-browser --ip=0.0.0.0
```

## Once image is there, start training
```
docker run -d --gpus all --runtime=nvidia NAME /bin/bash -c "cd zxreinforce_small; git pull; cd experiments/train_rl_agent ; python3.10 runner_final.py"
```


## Preparing a Docker image to run the code

We need the TF2.12 Docker image
```
# add the current user to the docker group
sudo usermod -a -G docker $USER
# close the ssh/bash and reopen it

# prepare and install the necessary packages to get nvidia support in docker
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install nvidia-docker2

# get the image
docker pull tensorflow/tensorflow:2.12.0-gpu
```

After the Docker image is downloaded, we need to start it and to prepare it with the necessary dependencies
```
docker run -it --gpus all --runtime nvidia --rm tensorflow/tensorflow:2.12.0-gpu bash
```

Within the Docker image that is started:
```
# install git and python3.10
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install git
apt install python3.10
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

#clone the repo and install the requirements
git clone https://github.com/alexandrupaler/zxreinforce_small.git
cd zxreinforce_small
pip3.10 install -r requirements.txt
```

After installing everything in the container
* `exit` the image
* `docker commit <container_id> dreamy_heisenberg` where the container_id is from `docker ps -a`

Finally, whenever the image is needed
```
docker run --gpus all --runtime=nvidia -it dreamy_heisenberg bash
```

## Training without Docker

To train the model without Docker, follow these steps:

### Prerequisites

1. **Python 3.10**: The project is tested with Python 3.10. Install it if needed:
   ```bash
   # On Ubuntu/Debian
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt update
   sudo apt install python3.10 python3.10-venv python3.10-dev
   ```

2. **CUDA and cuDNN** (for GPU training): The project requires CUDA 11.x and cuDNN 8.6.0.163. Install them if you plan to use GPU:
   - Download CUDA 11.x from [NVIDIA](https://developer.nvidia.com/cuda-11-0-0-download-archive)
   - Download cuDNN from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
   - Follow installation instructions for your system

### Setup

1. **Create a virtual environment**:
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   
   pip install -r requirements.txt
   ```

3. **Verify GPU setup** (if using GPU):
   ```bash
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

### Running Training

1. **Navigate to the training directory**:
   ```bash
   cd experiments/train_rl_agent
   ```

2. **Run the training script**:
   ```bash
   python runner_final.py [--ent_coeff 0.1] [--learning_rate 3e-4] [--add_reward_per_step 0.0]
   ```

   Optional arguments:
   - `--ent_coeff`: Entropy coefficient (default: 0.1)
   - `--learning_rate`: Learning rate (default: 3e-4)
   - `--add_reward_per_step`: Reward per step, should be non-positive (default: 0.0)

3. **GPU Configuration**: The script is configured to use 2 GPUs by default (line 170 in `runner_final.py`). If you have a different number of GPUs:
   - **Single GPU**: Change line 170 from:
     ```python
     strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
     ```
     to:
     ```python
     strategy = tf.distribute.MirroredStrategy(["GPU:0"])
     ```
   - **CPU only**: Change to:
     ```python
     strategy = tf.distribute.get_strategy()  # Default strategy (CPU)
     ```
   - **Multiple GPUs**: Adjust the list, e.g., `["GPU:0", "GPU:1", "GPU:2"]`

4. **Monitoring training**: Training logs are saved in `runs/para/<timestamp>/logs/`. View with TensorBoard:
   ```bash
   tensorboard --logdir runs/para
   ```

5. **Model checkpoints**: Saved models are stored in `runs/para/<timestamp>/saved_agent/` every 10 updates (configurable via `save_every` parameter).

### Notes

- Training uses 90 parallel environments by default (`n_envs = 90`). You can reduce this if you have limited CPU/memory.
- The script uses multiprocessing for environments (`multiprocess = True`). Set to `False` if you encounter issues.
- Total training timesteps: 36 million (configurable via `total_timesteps` parameter).
- The minibatch size (3000) should be divisible by the number of GPUs used.

# ZXreinforce
## This is a repository that does not include the saved data

This project contains the code used to produce the results in "Optimizing ZX-Diagrams with Deep Reinforcement Learning".
* Main code of the algorithm is in zxreinforce
* A script showing how to train an agent is at experiments/train_rl_agent/runner_final.py
* The agent's training progress can be monitored with experiments/evaluation_rl_agent/evaluation_training_logger.ipynb
* An example notebook showing how to simplify a diagram with the trained agent is at experiments/evaluation_rl_agent/simplify_example_traj.ipynb
* Scripts to compare the performance of the RL agent to a greedy strategy and simulated annealing are in experiments/evaluation_performance
* The evaluation of the Copy action is done in experiments/eval_copy_action
* The evaluation of the action dependence on the local environment is done in experiments/prob_vs_layer
* The network weights of the agents trained for the ablation studies can be found in saved_agents
