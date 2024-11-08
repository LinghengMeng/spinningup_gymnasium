This branch is to make SpinningUp work with [Gymnasium](https://gymnasium.farama.org/) and simplify it by removing mpi4py related code. In addition, it is also made to work with [IsaacLab](https://isaac-sim.github.io/IsaacLab/main/index.html).

# Installation
The installation assumes you have conda installed.
* Install SpinningUp for Gymnasium:
    * `conda create -n spinup_env python=3.10`
    * `conda activate spinup_env`
    * `cd spinningup_gymnasium`
    * `pip install -e .`
* Install IsaacLab
    * Following [Isaac Lab Doc: Installation using Isaac Sim Binaries](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html) to install Isaac Sim
    * Install Isaac Lab
        * clone IsaacLab: `git clone https://github.com/isaac-sim/IsaacLab.git`
        * Create IsaacSim symbolic link: 
            * `cd IsaacLab`
            * `ln -s path_to_isaac_sim _isaac_sim`
        * Add complimentery packages to conda environment
            * `./isaaclab.sh --conda spinup_env`
                * Note: use the same virtual env for Spinup.
        * Reactive virtula env: `conda activate spinup_env`
        * Install: `./isaaclab.sh --install`
    * Test IsaacLab installation: `./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py`
        * If you are remotely accessing your compute, add `--headless` to the argument as `./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py --headless`

# Test 
## Test Spinup and Gymnasium:
* `python -m spinup.run ppo --env HalfCheetah-v4 --exp_name installtest_ppo`
    * Note: you can also run PPO with `python ./spinup/algos/pytorch/ppo/ppo.py --env HalfCheetah-v4 --exp_name installtest_ppo`
* `python -m spinup.run ddpg --env HalfCheetah-v4 --exp_name installtest_ddpg`
* `python -m spinup.run td3 --env HalfCheetah-v4 --exp_name installtest_td3`
* `python -m spinup.run sac --env HalfCheetah-v4 --exp_name installtest_sac`
* `python -m spinup.run vpg --env HalfCheetah-v4 --exp_name installtest_vpg`
## Test Spinup and IsaacLab:
* PPO: `python ./spinup/algos/pytorch/ppo/ppo_isaaclab.py --task Isaac-Lift-Cube-Franka-v0 --num_envs 1 --headless`
    * Note: 
        * Currently, vecterized environment is not included, so please set `--num_envs 1`.
        * The adapted code is also using [Weights&Biases](https://wandb.ai/) to log data, so you will need to login your wand.
* TD3: `python ./spinup/algos/pytorch/td3/td3_isaaclab.py --task Isaac-Lift-Cube-Franka-v0 --num_envs 1 --headless`
* SAC: `python ./spinup/algos/pytorch/sac/sac_isaaclab.py --task Isaac-Lift-Cube-Franka-v0 --num_envs 1 --headless`


# Changes made compare to master branch of the original repo
* mpi4py related code is removed
* tf1 related code is removed
* gym is replaced with gymnasium

#
++++++++++++++++++++++++++++++++++++++++++++++++++++\
**Note:** The following content from the original repo is kept as reference.
++++++++++++++++++++++++++++++++++++++++++++++++++++


**Status:** Maintenance (expect bug fixes and minor updates)

Welcome to Spinning Up in Deep RL! 
==================================

This is an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning (deep RL).

For the unfamiliar: [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (RL) is a machine learning approach for teaching agents how to solve tasks by trial and error. Deep RL refers to the combination of RL with [deep learning](http://ufldl.stanford.edu/tutorial/).

This module contains a variety of helpful resources, including:

- a short [introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) to RL terminology, kinds of algorithms, and basic theory,
- an [essay](https://spinningup.openai.com/en/latest/spinningup/spinningup.html) about how to grow into an RL research role,
- a [curated list](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) of important papers organized by topic,
- a well-documented [code repo](https://github.com/openai/spinningup) of short, standalone implementations of key algorithms,
- and a few [exercises](https://spinningup.openai.com/en/latest/spinningup/exercises.html) to serve as warm-ups.

Get started at [spinningup.openai.com](https://spinningup.openai.com)!


Citing Spinning Up
------------------

If you reference or use Spinning Up in your research, please cite:

```
@article{SpinningUp2018,
    author = {Achiam, Joshua},
    title = {{Spinning Up in Deep Reinforcement Learning}},
    year = {2018}
}
```