import numpy as np
import pickle

demo_file = 'data/pendulum/demo_constraints.pkl'
sample_file = 'data/pendulum/expected_constraints.pkl'
trial = 0

with open(demo_file, 'rb') as f:
    all_demo_accrual = pickle.load(f)
    demo_accrual = all_demo_accrual[trial]

with open(sample_file, 'rb') as f:
    sample_accrual = pickle.load(f)

num_constraints = sample_accrual['numConstraints']
num_samples = sample_accrual['numTrajectories']
total_accrual = []
for i in range(num_constraints):
    total_accrual.append(len(sample_accrual[i]))

# sorted_constraints = np.argsort(total_accrual)[::-1] # descending order
constraint_set = set(range(num_constraints))
accrued_constraints = set(demo_accrual)
unaccrued_constraints = list(constraint_set - accrued_constraints)
# likeliest_constraints = [sorted_constraints[ind] for ind in unaccrued_constraints]

unaccrued_sample_accrual = [total_accrual[c] for c in unaccrued_constraints]
descending_indices = np.argsort(unaccrued_sample_accrual)[::-1]
likeliest_constraints = [unaccrued_constraints[ind] for ind in descending_indices]
print(likeliest_constraints)
