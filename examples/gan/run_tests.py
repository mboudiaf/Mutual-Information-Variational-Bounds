import os
import itertools

beta = [0.5, 1, 1.5]
unroll_critic = [1, 5, 10]

comb_test_params = itertools.product(beta, unroll_critic)
labels_test_params = ['beta', 'unroll_critic']

fixed_params = {"epochs": 25, "data_size":100000}

for param_comb in comb_test_params:
    command_line = "python3 demo_script_gan.py "
    command_line += "--labels_test_params {} ".format(' '.join(labels_test_params))
    for fix_param_name, fix_param_value in fixed_params.items():
            command_line += "--{} {} ".format(fix_param_name, fix_param_value)
    for i, param in enumerate(param_comb):
            command_line += "--{} {} ".format(labels_test_params[i], param)
    print(command_line)
    os.system(command_line)
print("All experiments were successfully carried out. ")