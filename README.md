# Continuous-state MLCI

This is the public code repository for the paper "Maximum likelihood constraint inference on continuous state spaces," presented at the ICRA 2022 conference.

## Setup
Ubuntu 20.04 and Python 3.8 is recommended. First, make sure required system packages are installed:
```sudo apt install mpich swig```

Next, install required Python packages (consider making a virtual environment first with ```python -m venv venv_MLCI```):

```pip install -r requirements.txt```

```pip install -e custom_envs```

## Example code use


```python ppo_pendulum.py --logdir logs/pendulum --experiment 1 --entropy 1.5 --numEpochs 40```

```python do_rollouts.py logs/pendulum --runExperiments startsEnds5.pkl --singleExperiment 1 --samplesPerExperiment 10000 --outputFile data/pendulum/expected_rollouts.pkl```

```python contraints_from_trajectories.py --sampleFile data/pendulum/expected_rollouts.pkl --outputFile data/pendulum/expected_constraints.pkl --envType pendulum```

```python constraints_from_trajectories.py --demoFileBase logs/pendulum/pendulumAgentShort_NoLowVeloAt5Tenths --outputFile data/pendulum/demo_constraints.pkl --envType pendulum```
