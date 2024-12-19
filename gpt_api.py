
"""
gpt_api.py

This file defines the `GetActions` class, a multithreaded implementation designed to interface with a GPT-based model 
to predict **robotic control modes** based on the current robot state, task instructions, and surrounding object information.

Key components and functionality:
1. **GPT API Integration**:
   - The class interacts with the OpenAI GPT API to generate predictions for control modes based on the robot's current state and task context.

2. **Control Mode Selection**:
   - Processes the GPT-generated response to select control modes. Besides the most likely modes, the second most likely control modes are tracked and evaluated for adaptive mode switching when necessary.

3. **Prompts Integration**:
   - The file relies on `prompts.py` to generate structured prompts for communicating task details, robot state, and object information to the model.
   
4. **Example and Rule Management**:
   - Examples and rules from past tasks are dynamically loaded, summarized, and integrated into the model's prompt.
   - Examples serve as input-output pairs for guiding the model, while rules summarize patterns derived from examples to aid predictions.

5. **Dynamic Context Updating**:
   - Tracks and updates the robot's position, orientation, gripper state, and surrounding object poses in real-time.
   - Supports scenarios where objects are being grasped, dropped, or moved.

6. **Threaded Execution**:
   - The `GetActions` class is implemented as a thread, allowing continuous prediction and real-time updates without blocking other processes.

7. **Utility Functions**:
   - Includes helper methods for generating prompts, normalizing model probabilities, and mapping predicted control modes to actionable commands.

"""


import threading
import os
import re
import json
import time
import pickle
import random
from openai import OpenAI
MODEL="gpt-4o"

from actions import *

from prompts import *

class GetActions(threading.Thread):
    def __init__(self, configs, participant_number, gripper_opening, base_cyclic, GC):
        super().__init__()
        self.client = OpenAI(api_key="your_api_key")
        self.gripper_opening = gripper_opening
        self.base_cyclic = base_cyclic
        self.GC = GC
        self.configs = configs
        self.shuffle = configs["shuffle"]

        self.switch_previous_threshold = configs["switch_previous_threshold"]

        self.natural_languages = configs["natural_languages"]
        self.use_robot_location = configs["use_robot_location"]
        self.relative = configs["relative"]
        self.holding_prompt = configs["holding_prompt"]
        self.always_opposite = configs["always_opposite"]
        self.gripper_mode = configs["gripper_mode"]
        self.summarize_examples = configs["summarize_examples"]
        
        self.binary_gripper = configs["binary_gripper"]
        self.gripper_discrete_nums = configs["gripper_discrete_nums"]
        self.use_orientation_examples = configs["use_orientation_examples"]

        self.use_coordinate_system_information = configs["use_coordinate_system_information"]


        self.task = configs["task"]
        
        # Initialize prompts based on configurations
        if self.natural_languages:
            self.get_natural_languages_prompts()
        else:
            self.natural_languages_prompts = None
        self.prompt_objective = prompt_objective
        self.prompt_data_structures = generate_prompt_data_structures(self.natural_languages_prompts, self.natural_languages, self.use_robot_location, self.relative, self.holding_prompt, self.use_orientation_examples, self.use_coordinate_system_information, self.binary_gripper, self.gripper_discrete_nums)
        self.prompt_task_specification = generate_prompt_task_specifications(self.always_opposite, self.natural_languages, self.gripper_mode)
        if self.summarize_examples:
            self.prompt_objective_summarize = prompt_objective_summarize
            self.prompt_task_specification_summarize = generate_prompt_task_specifications(self.always_opposite, self.natural_languages, self.gripper_mode, analyze_examples = True)
        
        # State tracking variables
        self.running = True
        self.generated_actions = None
        self.generated_action_names = None
        self.pos = None

        self.position_approximate = configs["position_approximate"]
        self.orientation_approximate = configs["orientation_approximate"]
        self.decimal = configs["decimal"]
        if self.decimal > 0:
            assert self.position_approximate == self.orientation_approximate == 1

        # Initialize object poses and rules
        self.generate_object_poses(configs)
        self.grasped_object = None
        self.dropped_object = None
        self.lock = threading.Lock()
        self.use_examples = configs["use_examples"]
        self.update_rules = configs["update_rules"]
        self.inherit_rules = configs["inherit_rules"]
        self.sample_all_examples = configs["sample_all_examples"]
        self.one_rule_per_example = configs["one_rule_per_example"]
        self.latest_example = None


        # Load existing examples and rules if available
        if os.path.exists(f'examples/{participant_number}/{configs["task_name"]}/txt/{configs["example_file"]}'):
            assert configs["example_index"] != 0
            if not (self.configs["inherit_rules"] and not self.configs["sample_all_examples"]):
                assert os.path.exists(f'examples/{participant_number}/{configs["task_name"]}/pkl/{configs["example_file"].replace("txt", "pkl")}')

            if not configs["inherit_rules"]:
                with open(f'examples/{participant_number}/{configs["task_name"]}/txt/{configs["example_file"]}', 'r') as file:
                    self.examples_prompt_all = file.read()
                with open(f'examples/{participant_number}/{configs["task_name"]}/pkl/{configs["example_file"].replace("txt", "pkl")}', 'rb') as file:
                    self.example_list = pickle.load(file)
                self.inherited_rules = ''''''
                self.inherited_rule_list = []
                if self.use_examples:
                    if self.summarize_examples:
                        self.summarize()
                    else:
                        self.summarized_rules = ''''''

            else:
                if configs["sample_all_examples"]:
                    with open(f'examples/{participant_number}/{configs["task_name"]}/pkl/{configs["example_file"].replace("txt", "pkl")}', 'rb') as file:
                        self.example_list = pickle.load(file)
                        self.examples_prompt_all = ''.join(random.sample(self.example_list, min(len(self.example_list), configs["sample_all_examples"])))
                else:
                    self.examples_prompt_all = ''''''
                    self.example_list = []
                

                with open(f'rules/{participant_number}/{configs["task_name"]}/txt/{configs["example_file"]}', 'r') as file:
                    self.inherited_rules = file.read()
                with open(f'rules/{participant_number}/{configs["task_name"]}/pkl/{configs["example_file"].replace("txt", "pkl")}', 'rb') as file:
                    self.inherited_rule_list = pickle.load(file)
                self.summarized_rules = self.inherited_rules
        else:
            assert configs["example_index"] == 0
            self.examples_prompt_all = ''''''
            self.example_list = []
            self.inherited_rules = ''''''
            self.inherited_rule_list = []
            self.summarized_rules = ''''''
        
        self.all_rules = configs["all_rules"]
        if self.all_rules:
            self.rule_list = self.inherited_rule_list
        self.example_index = configs["example_index"]

        # Previous state tracking 
        self.previous_executed_action_names = [None]*4
        self.second = [False]*4

        self.previous_pos = None
        self.previous_object_poses = None
        self.gripper_changed = False
                
    def run(self):
        # Main loop for the thread, executes the mode switching process
        # self.generated_actions, self.generated_action_names, self.pos, self.object_poses = self.get_actions()
        # while self.running:
        #     self.get_actions()
        #     time.sleep(0.5)  
        self.get_actions()

    def stop(self):
        # Stops the thread by setting the running flag to False
        self.running = False

    
    def get_actions(self):
        with self.lock:    
            # Refresh feedback to retrieve the current state of the robot  
            feedback = self.base_cyclic.RefreshFeedback()
        
        assert self.binary_gripper
        gripper_opening = self.gripper_opening           

        pos = np.array([round(feedback.base.tool_pose_x * 100, 2) , 
            round(feedback.base.tool_pose_y * 100, 2), 
            round(feedback.base.tool_pose_z * 100, 2),
            round(feedback.base.tool_pose_theta_x, 2), 
            round(feedback.base.tool_pose_theta_y, 2), 
            round(feedback.base.tool_pose_theta_z, 2),
            round(gripper_opening)
            ])
        # print("current arm position:", pos)
        # Normalize orientation angles to fall within the range [0, 360)
        pos[3:-1][pos[3:-1] >= 360] -= 360
        pos[3:-1][pos[3:-1] < 0] += 360
        assert np.all((pos[3:-1] >= 0) & (pos[3:-1] < 360))
        
        self.pos = pos
        
        if self.grasped_object is not None:
            assert self.grasped_object in self.object_poses.keys()
            self.object_poses[self.grasped_object] = self.pos
        
        print("running")
        print("object_poses")
        print(self.object_poses)
        print("pos")
        print(self.pos)
        self.previous_pos = self.pos
        self.previous_object_poses = self.object_poses

        # Generate the current task prompt based on the updated states
        prompt_current_task = generate_prompt_current_task(self.task, self.always_opposite, self.object_poses, self.pos, self.position_approximate, self.orientation_approximate, self.decimal, self.natural_languages, self.use_robot_location, self.relative, self.holding_prompt, self.grasped_object, self.dropped_object, self.binary_gripper, self.natural_languages_prompts, self.gripper_mode)
        
        while True:
            messages = [{"role": "user", "content": self.prompt_objective},
                        {"role": "user", "content": self.prompt_data_structures},
                        {"role": "user", "content": self.prompt_task_specification},
                    ]
            
            # Add examples or summarized rules if applicable
            if self.use_examples and (len(self.summarized_rules) > 10 or len(self.examples_prompt_all) > 10):
                print("using examples")
                if self.summarize_examples:                   
                   messages.append({"role": "user", "content": generate_provided_rules_prompt(self.summarized_rules)})
  
                else:
                    prompt_examples = generate_examples_prompt(self.examples_prompt_all, analyze_examples = False)
                    messages.append({"role": "user", "content": prompt_examples})
        

            messages.append({"role": "user", "content": prompt_current_task})
            # print(prompt_current_task)
            # os._exit(0)        
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                logprobs=True,
                top_logprobs = 4,
                temperature=0.0,
                # logit_bias={14711:-100},
            )

            # Parse the JSON response to extract actions
            gpt_response_ori = response.choices[0].message.content
            try:
                action_dict = json.loads(gpt_response_ori)
            except json.JSONDecodeError:
                try:
                    gpt_response = gpt_response_ori.strip('```json').strip('```').strip()
                    action_dict = json.loads(gpt_response)
                except json.JSONDecodeError:
                    
                    try:
                        gpt_response = gpt_response_ori.split('```json')[-1].split('```')[0]
                        action_dict = json.loads(gpt_response)
                        assert set(action_dict.keys()) == set(ACTION_GROUPS.keys())
                        
                    except:
                        print("format error, retrying")
                        continue

           # Validate the response format and content
            try:   
                if self.natural_languages:
                    break
                if self.always_opposite:
                    if self.natural_languages:
                        act_groups = ACTION_GROUPS_ALWAYS_OPPOSITE_NATURAL_LANGUAGES
                    else:
                        act_groups = ACTION_GROUPS_ALWAYS_OPPOSITE
                else:
                    if self.natural_languages:
                        act_groups = ACTION_GROUPS_NATURAL_LANGUAGES
                    else:
                        act_groups = ACTION_GROUPS
                
                assert set(key.lower() for key in action_dict.keys()) == set(key.lower() for key in act_groups.keys()), ("error 0", set(action_dict.keys()), set(act_groups.keys()))
                

                for key in act_groups:
                    assert action_dict[key].split(': ')[1].lower() in [action.lower() for action in act_groups[key]], ("error 1", key, action_dict[key].split(': ')[1], act_groups[key])
                    assert act_groups[key].index(action_dict[key].split(': ')[1]) == ['A', 'B', "C", 'D'].index(action_dict[key].split(': ')[0]), ("error 2", key, act_groups[key].index(action_dict[key].split(': ')[1]), ['A', 'B', "C", 'D'].index(action_dict[key].split(': ')[0]))

                break
            except:
                print("retrying")
                continue   
        # Extract probabilities for the actions and generate results
        probs = self.extract_top_logprobs_from_choice(response.choices[0])    
        self.generated_actions, self.generated_action_names, self.second = self.generate_actions_from_probs(probs)
    

    
    def normalize_logprobs(self, logprobs_dict):
        # Normalize the log probabilities to calculate a proper probability distribution
        probs = {k: np.exp(v) for k, v in logprobs_dict.items()}
        total_prob = sum(probs.values())
        normalized_probs = {k: v / total_prob for k, v in probs.items()}

        return normalized_probs

    
    
    def extract_top_logprobs_from_choice(self, choice, top_n=4):
        # Extract the top N log probabilities from the model's response
        result = []

        for logprob_info in choice.logprobs.content:
            if logprob_info.token in ['A', 'B', 'C', 'D'] and logprob_info.top_logprobs:
                token_logprob_dict = {}
                for top_logprob in logprob_info.top_logprobs[:top_n]:
                    token = top_logprob.token.decode('utf-8') if isinstance(top_logprob.token, bytes) else top_logprob.token
                    logprob = top_logprob.logprob
                
                    token_logprob_dict[token] = logprob
                result.append(token_logprob_dict)
        if self.always_opposite:
            assert len(result) == 2
        else:
            assert len(result)==4
        return [self.normalize_logprobs(d) for d in result]


    def generate_actions_from_probs(self, probs):
        # Generate the most likely control modes (actions) based on probabilities
        actions = []
        action_names = []
        selected_keys = []
        if self.always_opposite:
            assert len(probs)==2
        else:
            assert len(probs)==4

        second = [False, False, False, False] # Tracks if the second most likely action was chosen
        for i in range(4):
            if i%2==1 and self.always_opposite:
                action_name = ACTION_GROUPS[f'Group {i+1}'][['A', 'B', 'C', 'D'].index(selected_keys[i//2])]
                actions.append(np.array(ACTION_CORRESPONDENCES[action_name]))
                action_names.append(action_name)
                continue
            else:
                if self.always_opposite:
                    group_probs = probs[i//2]
                else:
                    group_probs = probs[i]
            print(f"probs for group {i}")
            print(group_probs)

            # Sort probabilities to find the top two
            sorted_items = sorted(group_probs.items(), key=lambda item: item[1], reverse=True)
            max_key, max_value = sorted_items[0]
            second_max_key, second_max_value = sorted_items[1]
            
            # Handle previously executed actions and switching conditions
            if self.previous_executed_action_names[i] is not None:
                previous_key = ['A', 'B', 'C', 'D'][ACTION_GROUPS[f'Group {i+1}'].index(self.previous_executed_action_names[i])]
                
                if previous_key == max_key and second_max_value > self.switch_previous_threshold:
                    selected_key = second_max_key # Switch to second most likely action
                    second[i] = True
                else:
                     selected_key = max_key
            else:
                selected_key = max_key

            selected_keys.append(selected_key)
            
            action_name = ACTION_GROUPS[f'Group {i+1}'][['A', 'B', 'C', 'D'].index(selected_key)]
            actions.append(np.array(ACTION_CORRESPONDENCES[action_name])) # Map to motion command
            action_names.append(action_name)
        self.previous_executed_action_names = [None] * 4
        return actions, action_names, second

    def get_natural_languages_prompts(self):
        # Define natural language descriptions for object relations and orientations
        self.natural_languages_prompts = {}
        self.natural_languages_prompts["x_relation"] = ["to the forward of the robot arm", "to the backward of the robot arm", "close to the robot arm along the x-axis"]
        self.natural_languages_prompts["y_relation"] = ["to the left of the robot arm", "to the right of the robot arm", "close to the robot arm along the y-axis"]
        self.natural_languages_prompts["z_relation"] = ["above the robot arm", "below the robot arm", "close to the robot arm along the z-axis"]

        self.natural_languages_prompts["pitch_relation"] = ["pitched more up compared to the robot arm", "pitched more down compared to the robot arm", "pitch orientation is close to the robot arm's pitch orientation"]
        self.natural_languages_prompts["roll_relation"] = ["rolled more left compared to the robot arm", "rolled more right compared to the robot arm", "roll orientation is close to the robot arm's roll orientation"]
        self.natural_languages_prompts["yaw_relation"] = ["yawed more left compared to the robot arm", "yawed more right compared to the robot arm", "yaw orientation is close to the robot arm's roll orientation"]
        if self.holding_prompt:
            self.natural_languages_prompts["holding_prompt"] = "The robot arm is holding the object."
        
    


    def add_example(self, output):
        # Add a new example to the examples list and update the prompt
        example_prompt = generate_prompt_current_task(self.task, self.always_opposite, self.previous_object_poses, self.previous_pos, self.position_approximate, self.orientation_approximate, self.decimal, self.natural_languages, self.use_robot_location, self.relative, self.holding_prompt, self.grasped_object, self.dropped_object, self.binary_gripper, self.natural_languages_prompts, self.gripper_mode, output, self.example_index)
        print(example_prompt)
        self.latest_example = example_prompt

        # Add the new example to the list and shuffle if required
        self.example_list.append(example_prompt)
        if self.shuffle:
            random.shuffle(self.example_list)
        self.examples_prompt_all = ''.join(self.example_list)

        self.example_index += 1
        # Update rules if needed or if this is the first interaction
        if self.update_rules or (self.configs["interact_index"] == 0):
            self.summarize()
        
        
    def summarize(self):
        # Summarize examples into general rules
        if self.one_rule_per_example:
            rule_prompt = prompt_summarize_singel_example
            use_example = self.latest_example
        elif self.sample_all_examples:
            rule_prompt = prompt_summary_rules
            use_example = ''.join(random.sample(self.example_list, min(len(self.example_list), self.sample_all_examples)))
        else:
            rule_prompt = prompt_summary_rules
            use_example = self.examples_prompt_all
        summarize_examples_messages = [
            {"role": "system", "content": self.prompt_objective_summarize},
            {"role": "user", "content": self.prompt_data_structures},
            {"role": "user", "content": self.prompt_task_specification_summarize},
            {"role": "user", "content": generate_examples_prompt(use_example, analyze_examples = True)},
            {"role": "user", "content": rule_prompt}
        ]

        # Generate summarized rules using the language model
        summarize_examples_response = self.client.chat.completions.create(
        model=MODEL,
        messages=summarize_examples_messages,
        temperature=0.0,
        )
        summarized_rules = summarize_examples_response.choices[0].message.content
        print(summarized_rules)

        # Update rule list and format the rules
        if self.one_rule_per_example:
            self.rule_list = self.rule_list + [summarized_rules]
            random.shuffle(self.rule_list)
            print("rule list length")
            print(len(self.rule_list))
            shuffled_rules = ""
            for i, rule in enumerate(self.rule_list, 1):
                shuffled_rules += f'{i}. '+ rule + '\n\n'
            self.summarized_rules = shuffled_rules
        else:
            if self.shuffle and self.inherit_rules:
                rule_pattern = r'(\d+\..*?)(?=\d+\.|\Z)'
                rules = re.findall(rule_pattern, summarized_rules, re.DOTALL)
                if self.all_rules:
                    self.rule_list = self.rule_list + rules
                else:
                    self.rule_list = self.inherited_rule_list + rules

                random.shuffle(self.rule_list)
                print("rule list length")
                print(len(self.rule_list))

                shuffled_rules = ""
                for i, rule in enumerate(self.rule_list, 1):
                    shuffled_rules += re.sub(r'^\d+', str(i), rule.strip()) + '\n\n'
                self.summarized_rules = shuffled_rules
            else:
                if len(self.inherited_rules) > 10:
                    summarized_rules = self.inherited_rules + '\n \n \n \n' + summarized_rules
            
                self.summarized_rules = summarized_rules
        

    def generate_object_poses(self, configs):
        # Generate initial object poses based on configuration
        self.object_poses = {}
        self.object_locations = configs["object_locations"][configs["interact_index"]]
        assert len(configs["objects"]) == len(self.object_locations)
        for j in range(len(configs["objects"])):
            obj_name = configs["objects"][j]
            self.object_poses[obj_name] = self.object_locations[j]
            