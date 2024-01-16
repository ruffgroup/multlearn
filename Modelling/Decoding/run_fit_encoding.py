import subprocess
import os
import numpy as np

script_name = '/mnt/d/multlearn-sns/Modelling/Decoding/fit_encoding.py'
output_prefix = '/mnt/d/multlearn-sns/Modelling/Decoding/run_encoding/out_'
IDs = range(2,65)
spaces = ['T1w']

for space in spaces:
    for i in IDs:
        if i not in [8, 13, 16, 31, 32, 44]:
            output_file = output_prefix + space + '_' + str(i) + '.txt'
            with open(output_file, 'w') as f:
                process = subprocess.run(['python', script_name, str(i), '--space', space, '--mask', 'visual'], stdout=f, stderr=subprocess.STDOUT)
            print('Finished subject', i, 'in', space, 'space')