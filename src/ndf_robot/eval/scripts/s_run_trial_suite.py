# Script to run all trials used in standard experiment
# Based on: https://janakiev.com/blog/python-shell-commands/
# DOES NOT WORK!!!

import subprocess
import os

log_file = 'trial_log.txt'

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

with open(log_file, 'w') as f:
    process = subprocess.Popen(['python3', 'evaluate_general.py', '--config_fname', 'grasp.yml'],
                            stdout=f,
                            universal_newlines=True)