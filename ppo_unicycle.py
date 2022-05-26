#!/usr/bin/env python3
import argparse
import pickle
import sys
import torch

from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler
from garage.torch.algos import PPO
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.torch.optimizers import OptimizerWrapper
from garage.trainer import Trainer

from custom_envs import CategoricalPolicy
from custom_envs.envs import UnicycleRacetrackEnv, StayInStraightLaneEnv, RelativeToLaneEnv

@wrap_experiment
def ppo_unicycle(ctxt=None, seed=1, n_epochs=100, entropy=1, discount=0.99):
    set_seed(seed)
    env = RelativeToLaneEnv()
    trainer = Trainer(ctxt)
    policy = CategoricalPolicy(env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length)
    policy_optimizer = OptimizerWrapper((torch.optim.Adam, {'lr': 2e-4}),
                                        policy,
                                        max_optimization_epochs=10,
                                        minibatch_size=64)
    
    vf_optimizer = OptimizerWrapper((torch.optim.Adam, {'lr': 2e-4}),
                                     value_function,
                                     max_optimization_epochs=10,
                                     minibatch_size=64)

    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               sampler=sampler,
               discount=discount,
               policy_optimizer=policy_optimizer,
               vf_optimizer=vf_optimizer,
               policy_ent_coeff=entropy,
               entropy_method='max',
               stop_entropy_gradient=True,
               center_adv=False
               )

    trainer.setup(algo, env)
    trainer.train(n_epochs=n_epochs, batch_size=10000, store_episodes=True)

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Log directory for this experiment')
parser.add_argument('--numEpochs', type=int, default=100, help='Number of epochs to train for')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--entropy', type=float, default=2, help='1/beta')
args = parser.parse_args()
if args.logdir:
    ctxt = {'log_dir': args.logdir}
else:
    ctxt = {}
ppo_unicycle(ctxt, 
             seed=args.seed, 
             n_epochs=args.numEpochs, 
             entropy=args.entropy)
