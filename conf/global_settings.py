""" configurations for this project

author Yunli
"""
import os
from datetime import datetime

#For 100 epochs:
'''
CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 100 
step_size = 10
i = 1
MILESTONES = []
while i * 5 <= EPOCH:
    MILESTONES.append(i* step_size)
    i += 1

#initial learning rate
#INIT_LR = 0.1

#time of we run the script
TIME_NOW = datetime.now().strftime("%F_%H-%M-%S.%f")

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10
'''
#for 10 epochs:

CHECKPOINT_PATH = 'checkpoint'

EPOCH = 10 
step_size = 5  # Now milestones will be generated every 5 epochs
i = 1
MILESTONES = []
while i * step_size <= EPOCH:
    MILESTONES.append(i * step_size)
    i += 1
#initial learning rate
#INIT_LR = 0.1

#time of we run the script
TIME_NOW = datetime.now().strftime("%F_%H-%M-%S.%f")

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10