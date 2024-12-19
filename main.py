"""
main.py: Main script for the LLM-Driven Automatic Mode Switching (LAMS) system.

This script orchestrates the integration of various modules and hardware components to conduct 
experiments involving a Kinova robotic arm.  It dynamically generates and updates control modes 
using LLM calls, processes user joystick inputs for robot control, and logs experimental data, 
creating a real-time interactive system for shared autonomy.

Key Features:
1. **Hardware Control**:
   - Interfaces with a Kinova robotic arm and gripper using the `kinova_basics` module.
   - Supports real-time Cartesian and joint control of the robot.

2. **User Interface**:
   - Displays current control modes and robot states using the `ui` module.
   - Allows users to interact with the robot through joystick inputs.

3. **GPT Integration**:
   - Leverages the `gpt_api` module to generate and update control modes dynamically using a language model.

4. **Experimental Logging**:
   - Logs actions, mode switches, and system states for later analysis using the `experiment_logger` module.

5. **Workflow Management**:
   - Handles experimental configurations and dynamically adjusts control modes and robot behaviors based on task context and user inputs.
"""

from kinova_basics import *
import requests
import base64
from io import BytesIO
import os
import numpy as np
import time
from datetime import datetime
import random
import json
import copy
import concurrent.futures

# Initialize Pygame for joystick input
import pygame
pygame.init()
pygame.joystick.init()
# Check if a joystick is connected
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
else:
    print("No joystick found.")
    exit()


from ui import *
from actions import *
from gpt_api import *
from keyboard_listener import *
import experiment_logger
class Mode:
    def __init__(self, command_mapping):
        self.command_mapping = command_mapping

with open(uargs.config_path, 'r') as file:
    configs = json.load(file)
    participant_number = configs["participant_number"]
    





gripper_opening = 1


# Establish a connection with the Kinova robotic arm
with utilities.DeviceConnection.createTcpConnection(uargs) as router:       
    base = BaseClient(router)
    base_cyclic = BaseCyclicClient(router)
    GC = GripperCommandExample(router, base)  

    # Ensure necessary directories exist (logs, exmaples, rules)
    if not os.path.exists(f'logs/llm/{participant_number}/{configs["task_name"]}') or not os.path.exists(f'examples/{participant_number}/{configs["task_name"]}') or not os.path.exists(f'rules/{participant_number}/{configs["task_name"]}'):
        os.makedirs(f'logs/llm/{participant_number}/{configs["task_name"]}', exist_ok=True)
        os.makedirs(f'examples/{participant_number}/{configs["task_name"]}/txt', exist_ok=True)
        os.makedirs(f'examples/{participant_number}/{configs["task_name"]}/pkl', exist_ok=True)
        os.makedirs(f'rules/{participant_number}/{configs["task_name"]}/txt', exist_ok=True)
        os.makedirs(f'rules/{participant_number}/{configs["task_name"]}/pkl', exist_ok=True)


    # Initialize the GetActions thread for generating robot actions
    get_action_task = GetActions(configs, participant_number, gripper_opening, base_cyclic, GC)
    get_action_task.daemon = True
    get_action_task.start()

    # Function to save examples in a separate thread
    def add_example_in_background(output_to_save):
        get_action_task.add_example(output_to_save)

    # Start experiment logging thread
    experiment_logger_thread = experiment_logger.ExperimentLogger(f"{uargs.config_path.split('/')[1].split('.')[0]}_{configs['interact_index']}.csv", f"logs/llm/{participant_number}")
    experiment_logger_thread.daemon = True
    experiment_logger_thread.start()

    # Start keyboard listener thread for experimenter interaction
    listener_thread = KeyListenerThread(uargs.config_path, participant_number, configs, get_action_task, experiment_logger_thread, show_actions)
    listener_thread.daemon = True
    listener_thread.start()

    


     # Initialize state variables
    previous_action_indices = None
    no_movement_time = None
    mode_switch_happened = False
    move_start_time = None
    hat_pressed_time = None
    ui_color_changed_time = None
    gripper_open_time = gripper_close_time = None
    mode_switching_num = 0

    # Delay to ensure threads are initialized
    time.sleep(5)

    # Sync initial actions and display them on the UI
    listener_thread.generated_actions_ori, listener_thread.generated_action_names_ori = get_action_task.generated_actions, get_action_task.generated_action_names
    get_action_task.previous_executed_action_names = [None]*4
    listener_thread.generated_actions = copy.deepcopy(listener_thread.generated_actions_ori)
    listener_thread.generated_action_names = copy.deepcopy(listener_thread.generated_action_names_ori)
    show_actions(listener_thread.generated_action_names, colors = ['blue']*4)

    # Log the initial mode
    mode_switch_log = experiment_logger.ModeSwitchLog(
        time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
        mode= Mode({
            'joystick up': listener_thread.generated_action_names[0],
            'joystick down': listener_thread.generated_action_names[1],
            'joystick left': listener_thread.generated_action_names[2],
            'joystick right': listener_thread.generated_action_names[3],
            
        }),  
        initiator="LLM"
    )
    mode_switch_log.log(experiment_logger_thread)

    ui_color_changed_time = time.time()
    
    # Additional state variables for robot and UI control
    tried_indices = None
    moved_indices = []
    moved = False
    moved_directions = None
    adjustment_mode = False
    gripper_changed = False
    indices_to_adjust = None
    tested_action_indices = set()
    pos_to_save = None
    obj_pos_to_save = None

    """
    Main control loop
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        try:
            while True:
                # Check for keyboard backspace input (for user study)
                # if listener_thread.backspaced:
                #     show_actions(listener_thread.generated_action_names, colors = ['blue']*4)
                #     ui_color_changed_time = time.time()
                #     listener_thread.backspaced = False

                # Update the footer with the grasped object
                if get_action_task.grasped_object is None:
                    footer = 'None'
                else:
                    footer = get_action_task.grasped_object
                show_footer(footer)

                # Reset UI color after 1 second
                if time.time() - ui_color_changed_time >= 1:
                    show_actions(listener_thread.generated_action_names, colors = ['black']*4)

                # Capture joystick input
                pygame.event.pump()               
                left = -joystick.get_axis(0)
                forward = -joystick.get_axis(1)

                # For user manual mode switches
                hat = joystick.get_hat(0)
                hat_up = hat == (0,1)
                hat_down = hat == (0,-1)
                hat_left = hat == (-1,0)
                hat_right = hat == (1,0)
                assert hat_up+hat_down+hat_left+hat_right<=1
                
                if not (forward >= configs["xbox_threshold"] and np.all(listener_thread.generated_actions[0] == 1)):
                    gripper_open_time = None
                if not (forward <= -configs["xbox_threshold"] and np.all(listener_thread.generated_actions[1] == -1)):
                    gripper_close_time = None


                # User manual mode switching logic
                if hat_up+hat_down+hat_left+hat_right > 0:
                    mode_switch_happened = True
                    if listener_thread.start_time is None:
                        listener_thread.start_task() # Start task if not already running
                    
                    # Prevent mode switching if pressed too quickly in succession
                    if hat_pressed_time is None:
                        hat_pressed_time = time.time()
                    elif time.time() - hat_pressed_time < configs["hat_gap_time"]:
                        continue

                    # Determine which direction was pressed and find the corresponding group    
                    adjust_index = np.where(np.array([hat_up, hat_down, hat_left, hat_right]) == 1)[0][0]
                    group_name = f'Group {adjust_index+1}'
                    if not adjustment_mode:
                        sampled_action = ACTION_GROUPS[group_name].index(listener_thread.generated_action_names[adjust_index])
                    adjustment_mode = True
                    
                    all_modes = [0,1,2,3]             
                    
                    # Cycle to the next mode in the group
                    sampled_action = (sampled_action + 1)%len(all_modes)
                    listener_thread.generated_action_names[adjust_index] = ACTION_GROUPS[group_name][sampled_action]
                    listener_thread.generated_actions[adjust_index] = ACTION_CORRESPONDENCES[ACTION_GROUPS[group_name][sampled_action]]
                    
                    colors = ['black'] * 4
                    colors[adjust_index] = 'red'

                    # Display updated modes on the UI
                    show_actions(listener_thread.generated_action_names, colors)
                    mode_switch_log = experiment_logger.ModeSwitchLog(
                        time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                        mode= Mode({
                            'joystick up': listener_thread.generated_action_names[0],
                            'joystick down': listener_thread.generated_action_names[1],
                            'joystick left': listener_thread.generated_action_names[2],
                            'joystick right': listener_thread.generated_action_names[3],
                            
                        }),  
                        initiator="manual"
                        )
                    mode_switch_log.log(experiment_logger_thread)
                    ui_color_changed_time = time.time()
                                        
                else:
                    hat_pressed_time = None

            
                # Check if the joystick is in a neutral position (no movement)
                if sum([abs(left)>=configs["xbox_threshold"], abs(forward)>=configs["xbox_threshold"]]) == 0:
                    moved_directions = None
                    with get_action_task.lock:
                        base.Stop() # Stop the robotic arm
                    if no_movement_time is None:
                        no_movement_time = time.time()
                        
                    # Reset the move start time if no movement for a short duration
                    if time.time() - no_movement_time >= 0.1:
                        move_start_time = None               
                    
                    """
                    Perform an LLM mode switch if conditions are met (no movement and task completion)
                    """
                    if ((time.time() - no_movement_time >= configs["switch_time"]) and (moved or gripper_changed) and not adjustment_mode) or get_action_task.gripper_changed:                    
                        show_loading() # Indicate loading on the UI
                        get_action_task.run() # Trigger GPT call to regenerate modes
                        moved = False
                        
                        # Update generated actions and check for changes
                        listener_thread.generated_actions_ori, listener_thread.generated_action_names_ori = get_action_task.generated_actions, get_action_task.generated_action_names
                        if not np.array_equal(listener_thread.generated_actions_ori, listener_thread.generated_actions):
                            # Highlight changed actions on the UI
                            changed_indices = [i for i, (a, b) in enumerate(zip(listener_thread.generated_action_names_ori, listener_thread.generated_action_names)) if a != b]
                            colors = ['blue' if i in changed_indices else 'black' for i in range(4)]
                            for i in range(4):
                                if get_action_task.second[i]:
                                    colors[i] = 'yellow' # Highlight secondary choices
                            get_action_task.previous_executed_action_names = [None]*4
                            listener_thread.generated_actions = copy.deepcopy(listener_thread.generated_actions_ori)
                            listener_thread.generated_action_names = copy.deepcopy(listener_thread.generated_action_names_ori)
                            show_actions(listener_thread.generated_action_names, colors) # Display updates on UI

                            # Log the mode switch event
                            mode_switch_log = experiment_logger.ModeSwitchLog(
                                time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                mode= Mode({
                                    'joystick up': listener_thread.generated_action_names[0],
                                    'joystick down': listener_thread.generated_action_names[1],
                                    'joystick left': listener_thread.generated_action_names[2],
                                    'joystick right': listener_thread.generated_action_names[3],
                                    
                                }),  
                                initiator="LLM"
                            )
                            mode_switch_log.log(experiment_logger_thread)
                            mode_switch_happened = True
                            ui_color_changed_time = time.time()
                        get_action_task.gripper_changed = False     
                        show_loading(show=False) # Remove loading indicator
                            
    
                # if the joystick moves
                else:
                    moved_directions = [forward>=configs["xbox_threshold"], forward<=-configs["xbox_threshold"], left>=configs["xbox_threshold"], left<=-configs["xbox_threshold"]]
                    if listener_thread.start_time is None:
                        listener_thread.start_task()
                    moved = True # Set movement flag

                    # Update mode switch time when movement starts
                    if mode_switch_happened:
                        listener_thread.mode_switch_time += time.time() -  no_movement_time
                        mode_switch_happened = False
                    no_movement_time = None
                    hat_pressed_time = None
                    if move_start_time is None:
                        move_start_time = time.time() # Record movement start time
                        tested_action_indices.clear()              
                    
                    """
                    If the user manually switched mode before this movement 
                    (save examples and rules in this situation)
                    """
                    if adjustment_mode:
                        output_to_save = {}
                        for i in range(4):
                            if not np.array_equal(listener_thread.generated_actions[i], listener_thread.generated_actions_ori[i]):
                                for k, v in ACTION_CORRESPONDENCES.items():
                                    if np.array_equal(v,listener_thread.generated_actions[i]):                                       
                                        # Save the action mapping switched by the user
                                        action_letter = ['A', 'B', 'C', 'D'][ACTION_GROUPS[f'Group {i+1}'].index(k)]
                                        action_group, action_index = find_action(k, action_to_group_index)     
                                        assert action_group == i
                                        if configs["natural_languages"]:
                                            action_name = ACTION_GROUPS_NATURAL_LANGUAGES[f'Group {action_group+1}'][action_index]
                                        else:
                                            action_name = ACTION_GROUPS[f'Group {action_group+1}'][action_index]
                                                
                                        output_to_save[f'Group {action_group+1}'] = f'{action_letter}: {action_name}'
                        # Update mode switch count and save the example
                        mode_switching_num +=len(output_to_save)
                        listener_thread.mode_switching_num +=len(output_to_save)
                        show_ui_count(listener_thread.mode_switching_num)
                        if not np.array_equal(listener_thread.generated_actions, listener_thread.generated_actions_ori):
                            output_to_save = json.dumps(output_to_save, indent=4)
                            if configs["use_examples"]:                                
                                executor.submit(add_example_in_background, output_to_save)
                                print("add example submitted")
                                
                                pos_to_save = None
                                obj_pos_to_save = None

                        listener_thread.generated_actions_ori = listener_thread.generated_actions
                        listener_thread.generated_action_names_ori = listener_thread.generated_action_names

                    adjustment_mode = False # Exit (user) adjustment mode

                    moved_indices.clear()
        
                    # Calculate the combined action based on joystick inputs
                    action = np.array([0,0,0,0,0,0], dtype=np.float64)
                    
                    if forward >= configs["xbox_threshold"]:
                        action += listener_thread.generated_actions[0] * configs["speed"] * forward
                        moved_indices.append(0)
                        tested_action_indices.add(0)
                        get_action_task.previous_executed_action_names[0] = listener_thread.generated_action_names[0]
                    if forward <= -configs["xbox_threshold"]:
                        action += listener_thread.generated_actions[1] * configs["speed"] * (-forward)
                        moved_indices.append(1)
                        tested_action_indices.add(1)
                        get_action_task.previous_executed_action_names[1] = listener_thread.generated_action_names[1]
                    if left >= configs["xbox_threshold"]:
                        action += listener_thread.generated_actions[2] * configs["speed"] * left
                        moved_indices.append(2)
                        tested_action_indices.add(2)
                        get_action_task.previous_executed_action_names[2] = listener_thread.generated_action_names[2]
                    if left <= -configs["xbox_threshold"]:
                        action += listener_thread.generated_actions[3] * configs["speed"] * (-left)
                        moved_indices.append(3)
                        tested_action_indices.add(3)
                        get_action_task.previous_executed_action_names[3] = listener_thread.generated_action_names[3]
                    
                    # Send the calculated action or gripper command to the robot
                    with get_action_task.lock:            
                        if forward >= configs["xbox_threshold"] and np.all(listener_thread.generated_actions[0] == 1):
                            GC.SendGripperSpeed()
                            log_action = np.array([0,0,0,0,0,0,1])      
                            
                        elif forward <= -configs["xbox_threshold"] and np.all(listener_thread.generated_actions[1] == -1):
                            GC.SendGripperSpeed(open=False)
                            log_action = np.array([0,0,0,0,0,0,-1])      

                        else:
                            twist_command(base, action, mode = 'teleoperation')
                            log_action = np.array([action[0],action[1],action[2],action[3],action[4],action[5],0])      
                        
                    # Log the executed action
                    action_log = experiment_logger.ActionLog(time=datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f'), action=log_action, gripper=get_action_task.gripper_opening, joystick_directions=np.array([abs(forward), abs(left)]))
                    action_log.log(experiment_logger_thread)

                    
        except KeyboardInterrupt:
            base.Stop()
            pygame.quit()
            
        