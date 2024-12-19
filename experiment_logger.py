import threading
import os
import numpy as np
import time
import copy
from ui import *

class ExperimentLogger(threading.Thread):
    def __init__(self, log_name : str, log_dir : str):
        # log_name: name of the log file
        # log_dir: directory of the log file
        super().__init__()
        self.log_name = log_name
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, log_name)
        self.start_time = time.time()
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # open file for writing
        self.log_file = open(self.log_path, 'w')
        self.log_file.write("Experiment Log\n")
        self.log_file.write("Time; Log Type; Log Message\n")
        
        self.current_amount_of_time_moving = 0.0
        self.current_amount_of_time_manual_mode_switching = 0.0
        self.time_of_last_actual_movement_or_mode_switch = None
        
        self.current_mode = None
        self.previous_mode = None
        
    def run(self):
        # # make log directory if it doesn't exist
        # if not os.path.exists(self.log_dir):
        #     os.makedirs(self.log_dir)
        # # open file for writing
        # self.log_file = open(self.log_path, 'w')
        # self.log_file.write("Experiment Log\n")
        # self.log_file.write("Time, Log Type, Log Message\n")
        pass
        
    def log(self, log_message : str):
        self.log_file.write(log_message + "\n")
        
    def set_current_mode_without_changing_previous_mode(self, mode):
        self.current_mode = copy.deepcopy(mode)
        
    def update_current_mode(self, mode):
        self.previous_mode = copy.deepcopy(self.current_mode)
        self.current_mode = copy.deepcopy(mode)
        
    def revert_mode(self):
        self.current_mode = copy.deepcopy(self.previous_mode)
        print(f"Reverted mode to {self.current_mode.command_mapping}")
        print(f"Previous mode was {self.previous_mode.command_mapping}")
        date_time_now = time.strftime("%Y-%m-%d_%H-%M-%S-%f")
        self.log(f"{date_time_now}; Mode Switch; initiator: reversion, new_mode: {self.current_mode.command_mapping}")
    
class LogType():
    def __init__(self, log_message : str):
        self.log_message = log_message
        
    def log(self, logger : ExperimentLogger):
        logger.log(self.log_message)
            
class ActionLog(LogType):
    def __init__(self, time: str, action : np.array, gripper: float, joystick_directions : np.array):
        self.time = time
        self.action = copy.deepcopy(action)
        self.gripper = copy.deepcopy(gripper)
        self.joystick_directions = joystick_directions
        # format action and joystick_directions with fixed number of decimal places in string format
        NUMBER_OF_DECIMAL_PLACES = 4
        MAX_WIDTH = 6
        # check if list or numpy array
        if isinstance(self.action, list):
            self.action = np.round(np.array(self.action), NUMBER_OF_DECIMAL_PLACES)
        if isinstance(self.joystick_directions, list):
            self.joystick_directions = np.round(np.array(self.joystick_directions), NUMBER_OF_DECIMAL_PLACES)
        formatter = {'float_kind': lambda x: f"{x: {MAX_WIDTH}.{NUMBER_OF_DECIMAL_PLACES}f}"}
        
        self.action = np.array2string(self.action, precision=NUMBER_OF_DECIMAL_PLACES, separator=',', formatter=formatter)
        self.joystick_directions = np.array2string(self.joystick_directions, precision=NUMBER_OF_DECIMAL_PLACES, separator=',', formatter=formatter)
        
    def log(self, logger : ExperimentLogger):
        log_message = f"{self.time}; Action; joystick_directions (up,down,left,right): {self.joystick_directions}, gripper: {self.gripper}, action_vector: {self.action}"
        logger.log(log_message)
    
class ModeSwitchLog(LogType):
    def __init__(self, time: str, mode, initiator : str):
        self.time = time
        self.mode = mode
        self.initiator = initiator # heuristic, manual, llm, etc. 
        
    def log(self, logger : ExperimentLogger):
        log_message = f"{self.time}; Mode Switch; initiator: {self.initiator}, new_mode: {self.mode.command_mapping}"
        logger.log(log_message)
    
class TaskStateLog(LogType):
    def __init__(self, time: str, task_state : str):
        self.time = time
        self.task_state = task_state # task started, what state of the task, task ended, etc.
        
    def log(self, logger : ExperimentLogger):
        log_message = f"{self.time}; Task State; task_state: {self.task_state}"
        logger.log(log_message)
        
class MetricsLog(LogType):
    def __init__(self, time: str, metrics : dict):
        self.time = time
        self.metrics = metrics # number of mode switches, number of actions, time to finish task, etc.
        
    def log(self, logger : ExperimentLogger):
        log_message = f"{self.time}; Metrics; {self.metrics}"
        logger.log(log_message)