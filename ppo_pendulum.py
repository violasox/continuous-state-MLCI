#!/usr/bin/env python3
"""This is an example to train a task with PPO algorithm (PyTorch).

Here it runs InvertedDoublePendulum-v2 environment with 100 iterations.
"""
import argparse
import pickle
import sys
import torch

from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler
from garage.torch.algos import PPO
from garage.torch.optimizers import OptimizerWrapper
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

from custom_envs import GymEnvWithMeta, CategoricalPolicy, EXP5_FILE, EXP8_FILE

@wrap_experiment
def ppo_pendulum(ctxt=None, 
                 seed=1, 
                 n_epochs=100, 
                 entropy=1, 
                 experiment=-1, 
                 constraint=5,
                 discount=0.9999,
                 vf_hidden_nonlinearity=torch.tanh,
                 vf_output_nonlinearity=None,
                 policy_hidden_nonlinearity=torch.tanh,
                 policy_output_nonlinearity=None):
    """Train PPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """

    trainer = ppo_setup(ctxt=ctxt, 
                        seed=seed, 
                        entropy=entropy, 
                        experiment=experiment, 
                        constraint=constraint, 
                        discount=discount,
                        vf_hidden_nonlinearity=vf_hidden_nonlinearity,
                        vf_output_nonlinearity=vf_output_nonlinearity,
                        policy_hidden_nonlinearity=policy_hidden_nonlinearity, 
                        policy_output_nonlinearity=policy_output_nonlinearity)
    avg_return = ppo_train(trainer, n_epochs=n_epochs)
    return (trainer, avg_return)
    
def ppo_setup(ctxt=None,
              seed=1, 
              entropy=1, 
              experiment=-1, 
              constraint=5,
              discount=0.9999,
              vf_hidden_nonlinearity=torch.tanh,
              vf_output_nonlinearity=None,
              policy_hidden_nonlinearity=torch.tanh,
              policy_output_nonlinearity=None):

    set_seed(seed)
    env = GymEnvWithMeta('InvertedPendulumDiscreteActionEnv-v0')

    if experiment >= 0:
        if constraint == 5:
            exp_file = EXP5_FILE
        else:
            exp_file = EXP8_FILE
        with open(exp_file, 'rb') as f:
            exp_dict = pickle.load(f)
        goal_state = exp_dict['goalPoints'][:,experiment]
        env.env.set_fixed_goal(goal_state)

    trainer = Trainer(ctxt)

    policy = CategoricalPolicy(env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=policy_hidden_nonlinearity,
                               output_nonlinearity=policy_output_nonlinearity)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=vf_hidden_nonlinearity,
                                              output_nonlinearity=vf_output_nonlinearity)

    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length)

    policy_optimizer = OptimizerWrapper((torch.optim.Adam, dict(lr=2.5e-4)), 
                                        policy,
                                        max_optimization_epochs=10, 
                                        minibatch_size=64)
                                                
    vf_optimizer = OptimizerWrapper((torch.optim.Adam, dict(lr=2.5e-4)),
                                    value_function,
                                    max_optimization_epochs=10,
                                    minibatch_size=64)

    algo = PPO(env_spec=env.spec,
               policy=policy,
               policy_optimizer=policy_optimizer,
               value_function=value_function,
               vf_optimizer=vf_optimizer,
               sampler=sampler,
               discount=discount,
               policy_ent_coeff=entropy,
               entropy_method='max',
               stop_entropy_gradient=True,
               center_adv=False)

    trainer.setup(algo, env)
    return trainer

def ppo_train(trainer, n_epochs=100, batch_size=30000):
    avg_return = trainer.train(n_epochs=n_epochs, batch_size=batch_size, store_episodes=True)
    return avg_return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, help='Log directory for this experiment')
    parser.add_argument('--numEpochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--entropy', type=float, default=2, help='1/beta')
    parser.add_argument('--experiment', type=int, default=-1, help='Use a single fixed goal state')
    parser.add_argument('--constraint', type=int, default=5, help='Select which obstacle to use')
    args = parser.parse_args()
    if args.logdir:
        ctxt = {'log_dir': args.logdir}
    else:
        ctxt = {}
    if args.constraint != 5 and args.constraint != 8:
        raise ValueError(args.constraint)
    (trainer, avg_return) = ppo_pendulum(ctxt, 
                                         seed=args.seed, 
                                         n_epochs=args.numEpochs, 
                                         entropy=args.entropy, 
                                         experiment=args.experiment,
                                         constraint=args.constraint)
