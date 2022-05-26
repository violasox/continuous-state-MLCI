import argparse
import numpy as np
import cloudpickle
import os
from garage.np import stack_tensor_dict_list
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from PIL import Image

# Rollout function adapted from garage
def rollout(env,
            agent,
            *,
            max_episode_length=np.inf,
            animated=False,
            pause_per_frame=None,
            deterministic=False):
    """Sample a single episode of the agent in the environment.
    Args:
        agent (Policy): Policy used to select actions.
        env (Environment): Environment to perform actions in.
        max_episode_length (int): If the episode reaches this many timesteps,
            it is truncated.
        animated (bool): If true, render the environment after each step.
        pause_per_frame (float): Time to sleep between steps. Only relevant if
            animated == true.
        deterministic (bool): If true, use the mean action returned by the
            stochastic policy instead of sampling from the returned action
            distribution.
    Returns:
        dict[str, np.ndarray or dict]: Dictionary, with keys:
            * observations(np.array): Flattened array of observations.
                There should be one more of these than actions. Note that
                observations[i] (for i < len(observations) - 1) was used by the
                agent to choose actions[i]. Should have shape
                :math:`(T + 1, S^*)`, i.e. the unflattened observation space of
                    the current environment.
            * actions(np.array): Non-flattened array of actions. Should have
                shape :math:`(T, S^*)`, i.e. the unflattened action space of
                the current environment.
            * rewards(np.array): Array of rewards of shape :math:`(T,)`, i.e. a
                1D array of length timesteps.
            * agent_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `agent_info` arrays.
            * env_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `env_info` arrays.
            * dones(np.array): Array of termination signals.
    """
    env_steps = []
    agent_infos = []
    observations = []
    frames = []
    last_obs, episode_infos = env.reset()
    agent.reset()
    episode_length = 0
    while episode_length < (max_episode_length or np.inf):
        if pause_per_frame is not None:
            time.sleep(pause_per_frame)
        a, agent_info = agent.get_action(last_obs)
        if deterministic and 'mean' in agent_info:
            a = agent_info['mean']
        es = env.step(a)
        env_steps.append(es)
        observations.append(last_obs)
        agent_infos.append(agent_info)
        if animated:
            frames.append(env.render(mode='rgb_array'))
        episode_length += 1
        if es.last:
            break
        last_obs = es.observation

    return dict(
        episode_infos=episode_infos,
        observations=np.array(observations),
        actions=np.array([es.action for es in env_steps]),
        rewards=np.array([es.reward for es in env_steps]),
        agent_infos=stack_tensor_dict_list(agent_infos),
        env_infos=stack_tensor_dict_list([es.env_info for es in env_steps]),
        dones=np.array([es.terminal for es in env_steps]),
        frames=frames,
    )

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=str, help='Path to the model log directory')
parser.add_argument('--parameterFile', type=str, help='Name of the parameter file')
parser.add_argument('--numRollouts', type=int, default=1, help='Number of rollouts to conduct')
parser.add_argument('--outputFile', type=str, help='Name of file to save rollout trajectory to')
parser.add_argument('--runExperiments', type=str,
                    help='Name of file holding start and end points for a sequence of fixed rollouts')
parser.add_argument('--samplesPerExperiment', type=int, default=1,
                    help='Number of rollouts to generate for each start and end point')
parser.add_argument('--singleExperiment', type=int, help='Only run a single experiment')
parser.add_argument('--makeFrames', action='store_true', help='save images of the rollout')
parser.add_argument('--frameFolder', type=str, help='Name of the folder to save rollout images to')
                     
args = parser.parse_args()
if args.parameterFile is None:
    parameterFile = os.path.join(args.directory, 'params.pkl')
else:
    parameterFile = os.path.join(args.directory, args.parameterFile)
with open(parameterFile, 'rb') as f:
    data = cloudpickle.load(f)
set_seed(2)
policy = data['algo'].policy
env = data['env']
env.reset()

# This is overwritten by experiment metadata if running set experiments
numRollouts = args.numRollouts
if args.runExperiments:
    experimentFile = args.runExperiments
    with open(experimentFile, 'rb') as f:
        experimentData = cloudpickle.load(f)
    if args.singleExperiment is not None:
        numRollouts = 1
    else:
        numRollouts = experimentData['numTrials']
    startPoints = experimentData['startPoints']
    goalPoints = experimentData['goalPoints']

if args.makeFrames:
    if args.frameFolder is None:
        framePath = os.path.join(args.directory, 'frames')
    else:
        framePath = os.path.join(args.directory, args.frameFolder)

allResults = {'numRollouts': numRollouts, 'samplesPerExperiment': args.samplesPerExperiment}
if not isinstance(env, GymEnv):
    env_object = env
else:
    env_object = env.env
if args.singleExperiment is not None:
    experiments_to_run = [args.singleExperiment]
else:
    experiments_to_run = range(numRollouts)
for i in experiments_to_run:
    if args.runExperiments:
        env_object.set_fixed_start(startPoints[:,i])
        env_object.set_fixed_goal(goalPoints[:,i])
    # For fixed unicycle start
    # env_object.set_fixed_start(np.zeros(4))
    for j in range(args.samplesPerExperiment):
        results = rollout(env, policy, animated=args.makeFrames)
    
        if args.makeFrames:
            frames = results['frames']
            curPath = os.path.join(framePath, '{}_{}'.format(i, j))
            os.makedirs(curPath, exist_ok=True)
            for k in range(len(frames)):
                im = Image.fromarray(frames[k])
                im.save(os.path.join(curPath, '{:04d}.png'.format(k)))
            del results['frames'] # Don't want to save renders in log data
         
        allResults[(i,j)] = results

if args.outputFile:
    out_file = args.outputFile
else:
    if args.runExperiments:
        if args.singleExperiment is not None:
            tag = 'exp{}_{}'.format(args.singleExperiment, args.samplesPerExperiment)
        else:
            tag = 'exp_{}'.format(args.samplesPerExperiment)
    else:
        tag = numRollouts
    out_file = os.path.join(args.directory, 'rollouts_{}.pkl'.format(tag))

with open(out_file, 'wb') as f:
    cloudpickle.dump(allResults, f)
