"""
keyboard_listener.py

This module defines a thread-based keyboard listener that integrates experimenter input (via the keyboard) with a robotic control system. It enables interaction through specific keypresses, primarily for monitering the grasping and releasing of objects and managing experimental data.

Key Features:
1. **Key-Based Robot Interaction**:
   - Spacebar toggles between grasping and releasing predefined objects.
   - Enter key terminates the program and saves the session data if required.

2. **Data Logging and Saving**:
   - Saves interaction data (e.g., object states, mode switches) to text and pickle files for further analysis.
   - Supports rules and examples for shared autonomy tasks.

3. **Integration with Task Execution**:
   - Interfaces with a task execution thread (`get_action_task`) for real-time action updates.

4. **Experimental Workflow Management**:
   - Handles experimental session states, such as participant trials, and updates configuration files dynamically.
"""


import threading
import keyboard  
import time
import pygame
import json
import pickle
import os
import experiment_logger
from datetime import datetime

class KeyListenerThread(threading.Thread):
    def __init__(self, config_path, participant_number, configs, get_action_task, logger, show_actions):
        super().__init__()
        self.config_path = config_path
        self.participant_number = participant_number
        self.configs = configs
        self.get_action_task = get_action_task
        self.logger = logger
        

        self.to_be_grasped = configs["to_be_grasped"]
        self.count = -1
        self.running = True
        self.grasp = False

        self.mode_switching_num = 0
        self.start_time = None
        self.mode_switch_time = 0
        self.show_actions = show_actions
        self.backspaced = False


    def on_space_press(self):
        self.count += 1
        self.grasp = 1 - self.count % 2
        index = (self.count // 2) % len(self.to_be_grasped)
        if self.grasp:
            # update_tkinter(f"Object {self.to_be_grasped[index]} is grasped.")
            print(f"Object {self.to_be_grasped[index]} is grasped.")
            self.get_action_task.grasped_object = self.to_be_grasped[index]
            self.get_action_task.gripper_opening = 0
            self.get_action_task.gripper_changed = True
            # self.get_action_task.run()

        else:
            # self.update_tkinter(f"Object {self.to_be_grasped[index]} is released.")
            print(f"Object {self.to_be_grasped[index]} is released.")
            self.get_action_task.grasped_object = None
            self.get_action_task.dropped_object = self.to_be_grasped[index]
            self.get_action_task.gripper_opening = 1
            self.get_action_task.object_poses[self.to_be_grasped[index]][2] = 0
            self.get_action_task.gripper_changed = True
            # self.get_action_task.run()

        if index == len(self.to_be_grasped) - 1 and not self.grasp:
            self.count = -1

    def on_enter_press(self):
        print("Enter key pressed, exiting the program...")
        self.running = False  # Stop the loop
        # self.get_action_task.running = False
        end_time = time.time()
        end_date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if self.configs["participant_number"]!=0 and self.configs["participant_number"]!=5 and self.configs["participant_number"]!=51:     
            # print(f"mode switching times: {self.mode_switching_num}")
            save = input(f'Do you want to save data for this trial? (y/n): ')
            if save.lower() != 'y':
                print("Discarding data")
                # pygame.quit()
                # exit()
                os._exit(0)
            
        
            if self.configs["use_examples"]:
                # self.get_action_task.get_actions(final_call = True)
                if self.configs["summarize_examples"]:
                    self.get_action_task.summarize()
                    with open(f'rules/{self.participant_number}/{self.configs["task_name"]}/txt/{self.configs["example_file"]}', 'w') as file:
                        file.write(self.get_action_task.summarized_rules)

                    with open(f'rules/{self.participant_number}/{self.configs["task_name"]}/pkl/{self.configs["example_file"].replace("txt", "pkl")}', 'wb') as file:
                        pickle.dump(self.get_action_task.rule_list, file)
                    print(f'Rules saved.')


                if self.configs["inherit_rules"] and not self.configs["sample_all_examples"]:
                    writing_mode = 'a'
                else:
                    writing_mode = 'w'
                with open(f'examples/{self.participant_number}/{self.configs["task_name"]}/txt/{self.configs["example_file"]}', writing_mode) as file:
                    file.write(self.get_action_task.examples_prompt_all)
                    
                if not (self.configs["inherit_rules"] and not self.configs["sample_all_examples"]):
                    with open(f'examples/{self.participant_number}/{self.configs["task_name"]}/pkl/{self.configs["example_file"].replace("txt", "pkl")}', 'wb') as file:
                        pickle.dump(self.get_action_task.example_list, file)
                print(f'Examples saved. Next example index: {self.get_action_task.example_index}')
                


            
                self.configs["example_index"] = self.get_action_task.example_index
            self.configs["interact_index"] = self.configs["interact_index"] + 1
            with open(self.config_path, 'w') as file:
                json.dump(self.configs, file, indent=4)
            
            

        
        
        print(f"manual mode switching times: {self.mode_switching_num}")
        print(f"task completion time: {end_time - self.start_time}")
        print(f"mode switch time: {self.mode_switch_time}")
        print(f"proportion of mode switch time: {self.mode_switch_time / (end_time - self.start_time)}")

        
        
        metrics_log = experiment_logger.MetricsLog(
            time=end_date_time, 
            metrics={'manual switches': self.mode_switching_num, 'task completion time': end_time - self.start_time, \
                      'mode switch time': self.mode_switch_time, 'proportion of mode switch time': self.mode_switch_time / (end_time - self.start_time)  }
        )
        metrics_log.log(self.logger)

        task_state_log = experiment_logger.TaskStateLog(
            time=end_date_time, 
            task_state="Task Ended"
        )
        task_state_log.log(self.logger)
        
        # pygame.quit()
        # exit()  # Exit the program after saving data
        os._exit(0)


    def on_r_press(self):
        print("R key pressed, reloading actions...")
        self.generated_action_names = self.generated_action_names_ori
        self.generated_actions = self.generated_actions_ori
        self.backspaced = True
        
        


    def run(self):
        while self.running:
            if keyboard.is_pressed(' '):
                self.on_space_press()
                time.sleep(0.2)  # Debounce to avoid multiple triggers

            if keyboard.is_pressed('backspace'):
                self.on_r_press()
                time.sleep(0.2)  # Debounce to avoid multiple triggers

            if keyboard.is_pressed('enter'):
                self.on_enter_press()
                break  # Break the loop after pressing Enter

    def start_task(self):
        self.start_time = time.time()
        task_state_log = experiment_logger.TaskStateLog(
            time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
            task_state="Task Started"
        )
        task_state_log.log(self.logger)


