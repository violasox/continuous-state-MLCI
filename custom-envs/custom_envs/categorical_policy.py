import torch

from garage.torch.modules import MultiHeadedMLPModule
from garage.torch.policies.stochastic_policy import StochasticPolicy
from torch.distributions import Categorical

class CategoricalPolicy(StochasticPolicy):
    def __init__(self,
                 env_spec,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 output_nonlinearity=None):

        super().__init__(env_spec, 'CategoricalPolicy')
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._module = MultiHeadedMLPModule(
            # TODO why am I using multi headed again?
            n_heads=1,
            input_dim=self._obs_dim,
            output_dims=[self._action_dim],
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearities=[output_nonlinearity])

    def forward(self, observations):
        logits = torch.softmax(self._module(observations)[0], axis=1)
        dist = Categorical(logits=logits)
        return (dist, {})
