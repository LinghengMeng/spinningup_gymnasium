This branch is to make SpinningUp work with Gymnasium and simplify it by removing mpi4py related code.

# Installation
The installation assumes you have conda installed.
* `conda create -n spinup_env python=3.10`
* `conda activate spinup_env`
* `cd spinningup_gymnasium`
* `pip install -e .`

# Test installation
* `python -m spinup.run ppo --env HalfCheetah-v4 --exp_name installtest_ppo`
    * Note: you can also run PPO with `python ./spinup/algos/pytorch/ppo/ppo.py --env HalfCheetah-v4 --exp_name installtest_ppo`
* `python -m spinup.run ddpg --env HalfCheetah-v4 --exp_name installtest_ddpg`
* `python -m spinup.run td3 --env HalfCheetah-v4 --exp_name installtest_td3`
* `python -m spinup.run sac --env HalfCheetah-v4 --exp_name installtest_sac`
* `python -m spinup.run vpg --env HalfCheetah-v4 --exp_name installtest_vpg`


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