import os
import itertools

update_critic = ["mine"]
critic_types = ["joint", "separate"]

comb_test_params = itertools.product(regularizers, critic_types)
labels_test_params = ['regularizer', 'critic_type']

fixed_params = {"epochs": 2, "data_size":10000}

for param_comb in comb_test_params:
    command_line = "python3 demo_script_gan.py "
    command_line += "--labels_test_params {} ".format(' '.join(labels_test_params))
    for fix_param_name, fix_param_value in fixed_params.items():
            command_line += "--{} {} ".format(fix_param_name, fix_param_value)
    for i, param in enumerate(param_comb):
            command_line += "--{} {} ".format(labels_test_params[i], param)
    print(command_line)
    os.system(command_line)
print("All tests were successfully passed. ")