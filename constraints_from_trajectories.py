import argparse
import csv
import math
import numpy as np
import os
import pickle

class Axis():
    def __init__(self, num_cells, min_val, max_val):
        self.num_cells = num_cells
        self.min_val = min_val
        self.max_val = max_val
        self.cell_size = (max_val - min_val) / num_cells

    def cell_from_val(self, value):
        # Wrap to axis boundaries (prevent landing at highest value to round down correctly)
        value = max(min(value, self.max_val - (self.cell_size/10)), self.min_val)
        cell = math.floor((value - self.min_val) / self.cell_size)
        return cell

    def get_cell_range(self, index):
        min_val = self.min_val + self.cell_size*index
        max_val = min_val + self.cell_size
        print('[{}, {}]'.format(min_val, max_val))

def load_samples(file_path, env_type):
    with open(file_path, 'rb') as f:
        traj_dict = pickle.load(f)
    (num_exp, num_samples, exp_ids) = load_sample_metadata(traj_dict)[0:3]

    all_states = []
    all_actions = []
    for i in exp_ids:
        for j in range(num_samples):
            if env_type == 'unicycle':
                if not traj_dict[(i,j)]['env_infos']['success'][-1]:
                    continue
            obs = traj_dict[(i,j)]['observations']
            states = np.zeros((obs.shape[0], 2))
            # Get theta and theta_dot from observations
            for k in range(obs.shape[0]):
                if env_type == 'pendulum':
                    states[k,0] = np.mod(np.arctan2(obs[k,1], obs[k,0]), 2*np.pi)
                    states[k,1] = obs[k,4]
                else:
                    states[k,0] = obs[k,0]
                    states[k,1] = obs[k,1]
            all_states.append(states)
            all_actions.append(traj_dict[(i,j)]['actions'])
    return (all_states, all_actions)

def load_sample_metadata_file(file_path):
    with open(file_path, 'rb') as f:
        traj_dict = pickle.load(f)
    return load_sample_metadata(traj_dict)

def load_sample_metadata(traj_dict):
    num_exp = traj_dict['numRollouts']
    num_samples = traj_dict['samplesPerExperiment']
    if num_exp == 1:
        for key in traj_dict.keys():
            if key[1] == 0:
                exp_ids = [key[0]]
                break
    else:
        exp_ids = range(num_exp)
    obs_dim = traj_dict[(exp_ids[0],0)]['observations'].shape[1]
    try:
        state_dim = traj_dict[(exp_ids[0],0)]['episode_infos']['start_state'].shape[0] 
    except:
        state_dim = traj_dict[(exp_ids[0],0)]['episode_infos']['start_vec'].shape[0] 
    return (num_exp, num_samples, exp_ids, state_dim, obs_dim)

def load_demonstrations(file_base, demo_indices, state_dim=2):
    all_states = []
    for i in demo_indices:
        states = []
        file_path = '{}{}.csv'.format(file_base, int(i))
        with open(file_path) as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                states.append(row)
        states = np.array(states)[:state_dim]
        states[0,:] = np.mod(states[0,:], 2*np.pi)
        all_states.append(states.T)
    return all_states

def write_sample_csv(out_file, accrual_dict):
    num_traj = accrual_dict['numTrajectories']
    num_constraints = accrual_dict['numConstraints']
    num_exp = accrual_dict['numExperiments']
    metadata = ['numTrajectories', num_traj, 'numConstraints', num_constraints, 'numExperiments', num_exp]
    with open(out_file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(metadata)
        for i in range(num_constraints*num_exp):
            if len(accrual_dict[i]) > 0:
                writer.writerow(accrual_dict[i])
            else:
                writer.writerow(['None'])

def write_dict(out_file, data):
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)

def check_grid_constraints(axes, states):
    dimensions = []
    for (i, axis) in enumerate(axes):
        dimensions.append(axis.num_cells)
    dimensions = tuple(dimensions)
    num_steps = states.shape[0]
    constraints = np.zeros(num_steps, dtype=np.int32)
    cells = np.zeros(len(axes), dtype=np.int32)
    for t in range(num_steps):
        for i in range(len(axes)):
            cells[i] = axes[i].cell_from_val(states[t,i])
        cell_tuple = tuple(cells)
        constraints[t] = np.ravel_multi_index(cell_tuple, dimensions)
    return np.unique(constraints)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampleFile', type=str, nargs='+', help='Path to pickle file for samples')
    parser.add_argument('--sampleDir', type=str, help='Path to directory containing per-experiment pickle files')
    parser.add_argument('--demoFileBase', type=str, help='Path to csv files for demos')
    parser.add_argument('--outputFile', type=str, help='File to save constraint accrual to')
    parser.add_argument('--envType', type=str, default='pendulum')
    args = parser.parse_args()

    axes = []
    if args.envType == 'pendulum':
        axes.append(Axis(10, 0, 2*math.pi)) # theta
        axes.append(Axis(10, -6, 6)) # theta_dot
    elif args.envType == 'unicycle':
        axes.append(Axis(2, -0.8, 0.2)) # x
        axes.append(Axis(14, 0, 8.5)) # y
    else:
        raise ValueError
    constraint_dim = []
    for axis in axes:
        constraint_dim.append(axis.num_cells)
    num_constraints = math.prod(constraint_dim)

    if args.outputFile:
        out_file = args.outputFile
    else:
        out_file = 'accrual.pkl'

    if args.sampleFile:
        num_exp = load_sample_metadata_file(args.sampleFile[0])[0]
        accrual_dict = {ind: [] for ind in range(num_constraints*num_exp)} 
        accrual_dict['numConstraints'] = num_constraints
        accrual_dict['numExperiments'] = num_exp
        num_traj = 0
        for sample_file in args.sampleFile:
            (all_states, all_actions) = load_samples(sample_file, args.envType)
            offset = num_traj
            new_num_traj = len(all_states)
            samples_per_exp = max(int(new_num_traj / num_exp), 1)
            # Contains a list of indices of every trajectory that violates each constraint
            for i in range(new_num_traj):
                violated_constraints = check_grid_constraints(axes, all_states[i])
                #this_exp = int(i / samples_per_exp)
                this_exp = 0
                for c in violated_constraints:
                    accrual_dict[num_constraints*this_exp + c].append(i + offset)
            num_traj += new_num_traj
        accrual_dict['numTrajectories'] = num_traj
        if out_file.endswith('.csv'):
            write_sample_csv(out_file, accrual_dict)
        else:
            write_dict(out_file, accrual_dict)
    elif args.sampleDir:
        accrual_dict = {}
        fileNames = os.listdir(args.sampleDir)
        num_exp = 0
        for f_name in fileNames:
            if f_name.endswith('.pkl'):
                f_path = os.path.join(args.sampleDir, f_name)
                exp_id = load_sample_metadata_file(f_path)[2][0]
                for c in range(num_constraints):
                    accrual_dict[num_constraints*exp_id + c] = []

                (all_states, all_actions) = load_samples(f_path, args.envType)
                num_traj = len(all_states) # Must be the same for all files!
                for i in range(num_traj):
                    violated_constraints = check_grid_constraints(axes, all_states[i])
                    for c in violated_constraints:
                        accrual_dict[num_constraints*exp_id + c].append(i)
                num_exp += 1
        accrual_dict['numTrajectories'] = num_traj*num_exp
        accrual_dict['numConstraints'] = num_constraints
        accrual_dict['numExperiments'] = num_exp
        if out_file.endswith('.csv'):
            write_sample_csv(out_file, accrual_dict)
        else:
            write_dict(out_file, accrual_dict)
                
    elif args.demoFileBase:
        # Load the experiment indices to use
        trials_file = os.path.join(os.path.dirname(args.demoFileBase), 'trial_indices.csv')
        with open(trials_file) as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                trial_indices = row
                break

        all_states = load_demonstrations(args.demoFileBase, trial_indices)
        num_demo = len(all_states)
        accrual_dict = {}
        for i in range(num_demo):
            accrual_dict[i] = check_grid_constraints(axes, all_states[i])
        accrual_dict['numConstraints'] = num_constraints
        accrual_dict['numTrajectories'] = num_demo
        
        if args.outputFile.endswith('csv'):
            with open(out_file, 'w') as f:
                writer = csv.writer(f, delimiter=',')
                for d in range(num_demo):
                    writer.writerow(accrual_dict[d])
        else: 
            with open(out_file, 'wb') as f:
                pickle.dump(accrual_dict, f)
