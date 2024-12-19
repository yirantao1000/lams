"""
prompts.py

This file defines functions and constants used to generate structured prompts for GPT-based task execution 
in robotic manipulation scenarios. The prompts are designed to communicate the current state of the robot, 
the surrounding objects, and task-specific information to the language model (e.g., GPT-4), enabling it to 
predict the most likely control modes.

Key functionalities:
1. **Objective Prompt**: Specifies the high-level goal for the language model.
2. **Data Structure Definitions**: Constructs descriptions of the robot's state, object positions, and orientations 
   based on natural language or numerical formats.
3. **Task Specifications**: Defines action groups and task-specific requirements for the language model.
4. **Examples and Rules Prompt**: Prompts for summarizing rules from examples, integrating rules or examples into 
   action prediction prompts, and converting generated examples into natural languages.
5. **Current Task Prompt**: Combines all components (robot state, object information, and task details) into a 
   single prompt for the language model.

The file is modularized to support flexibility, enabling prompts to be customized based on:
- Whether to use natural language or numeric descriptions.
- Task-specific requirements, such as gripper modes or relative vs absolute positioning.
- Whether examples and rules should be directly included or summarized.

"""



import numpy as np

prompt_objective = '''
    **Objective:**  
    You will be given task instructions, the current state of the robot arm, and information of objects around. Your goal is to predict the most likely actions out of the specified groups of actions.  
    '''

def generate_prompt_data_structures(natural_languages_prompts, natural_languages, use_robot_location, relative, holding_prompt, use_orientation_examples, use_coordinate_system_information, binary_gripper, gripper_discrete_nums):
    if use_coordinate_system_information:
        coordinate_system_prompt = '''**Coordinate System:**

    - **Frame of Reference:** All poses are in the robot frame, relative to the base position `base_pose = {{ 'position': [0,0,0], 'orientation': [0,0,0] }}`.
        '''
    else:
        coordinate_system_prompt = ''

    if use_robot_location:
        robot_pos_prompt = '''
            - `position`: A dictionary indicating the coordinates of the robot arm's position in centimeters.
                - `x`: The position along the x-axis, an integer value in centimeters.
                - `y`: The position along the y-axis, an integer value in centimeters.
                - `z`: The position along the z-axis, an integer value in centimeters.
            - `orientation`: A dictionary indicating the orientation of the robot arm in degrees.
                - `theta_x`: The rotation around the x-axis, an integer value in degrees ranging from 0 to 360.
                - `theta_y`: The rotation around the y-axis, an integer value in degrees ranging from 0 to 360.
                - `theta_z`: The rotation around the z-axis, an integer value in degrees ranging from 0 to 360.'''
    else:
        robot_pos_prompt = ''
    
    if natural_languages:
        
        if holding_prompt:
            object_prompt = f'''
            - `relative_pos`: Either a natural language string "{list(natural_languages_prompts.values())[-1]}" or "has been dropped", or a dictionary with two keys `relative_position` and `relative_orientation`. 

                For `relative_position`, the dictionary should have three keys `{list(natural_languages_prompts.keys())[0]}`, `{list(natural_languages_prompts.keys())[1]}`, and `{list(natural_languages_prompts.keys())[2]}`, each containing a natural language string describing the object's position relative to the robot arm in the respective direction. For example:
                
                - `{list(natural_languages_prompts.keys())[0]}`: "{list(natural_languages_prompts.values())[0][0]}" or "{list(natural_languages_prompts.values())[0][1]}" or "{list(natural_languages_prompts.values())[0][2]}"
                - `{list(natural_languages_prompts.keys())[1]}`: "{list(natural_languages_prompts.values())[1][0]}" or "{list(natural_languages_prompts.values())[1][1]}" or "{list(natural_languages_prompts.values())[1][2]}"
                - `{list(natural_languages_prompts.keys())[2]}`: "{list(natural_languages_prompts.values())[2][0]}" or "{list(natural_languages_prompts.values())[2][1]}" or "{list(natural_languages_prompts.values())[2][2]}"
                
                For `relative_orientation`, the dictionary should have three keys `{list(natural_languages_prompts.keys())[3]}`, `{list(natural_languages_prompts.keys())[4]}`, and `{list(natural_languages_prompts.keys())[5]}`, each containing a natural language string describing the object's orientation relative to the robot arm in the respective axis. For example:

                - `{list(natural_languages_prompts.keys())[3]}`: "{list(natural_languages_prompts.values())[3][0]}" or "{list(natural_languages_prompts.values())[3][1]}" or "{list(natural_languages_prompts.values())[3][2]}"
                - `{list(natural_languages_prompts.keys())[4]}`: "{list(natural_languages_prompts.values())[4][0]}" or "{list(natural_languages_prompts.values())[4][1]}" or "{list(natural_languages_prompts.values())[4][2]}"
                - `{list(natural_languages_prompts.keys())[5]}`: "{list(natural_languages_prompts.values())[5][0]}" or "{list(natural_languages_prompts.values())[5][1]}" or "{list(natural_languages_prompts.values())[5][2]}"'''
        else:
            object_prompt = f'''
            - `relative_pos`: A dictionary with two keys `relative_position` and `relative_orientation`. 

                For `relative_position`, the dictionary should have three keys `{list(natural_languages_prompts.keys())[0]}`, `{list(natural_languages_prompts.keys())[1]}`, and `{list(natural_languages_prompts.keys())[2]}`, each containing a natural language string describing the object's position relative to the robot arm in the respective direction. For example:
                
                - `{list(natural_languages_prompts.keys())[0]}`: "{list(natural_languages_prompts.values())[0][0]}" or "{list(natural_languages_prompts.values())[0][1]}" or "{list(natural_languages_prompts.values())[0][2]}"
                - `{list(natural_languages_prompts.keys())[1]}`: "{list(natural_languages_prompts.values())[1][0]}" or "{list(natural_languages_prompts.values())[1][1]}" or "{list(natural_languages_prompts.values())[1][2]}"
                - `{list(natural_languages_prompts.keys())[2]}`: "{list(natural_languages_prompts.values())[2][0]}" or "{list(natural_languages_prompts.values())[2][1]}" or "{list(natural_languages_prompts.values())[2][2]}"
                
                For `relative_orientation`, the dictionary should have three keys `{list(natural_languages_prompts.keys())[3]}`, `{list(natural_languages_prompts.keys())[4]}`, and `{list(natural_languages_prompts.keys())[5]}`, each containing a natural language string describing the object's orientation relative to the robot arm in the respective axis. For example:

                - `{list(natural_languages_prompts.keys())[3]}`: "{list(natural_languages_prompts.values())[3][0]}" or "{list(natural_languages_prompts.values())[3][1]}" or "{list(natural_languages_prompts.values())[3][2]}"
                - `{list(natural_languages_prompts.keys())[4]}`: "{list(natural_languages_prompts.values())[4][0]}" or "{list(natural_languages_prompts.values())[4][1]}" or "{list(natural_languages_prompts.values())[4][2]}"
                - `{list(natural_languages_prompts.keys())[5]}`: "{list(natural_languages_prompts.values())[5][0]}" or "{list(natural_languages_prompts.values())[5][1]}" or "{list(natural_languages_prompts.values())[5][2]}"'''
        
        directional_information_prompt = ''

    else:
        if relative:
            object_prompt = '''
            - `relative_position`: A dictionary indicating the coordinates of the object's centroid relative to the robot arm in centimeters.
                - `x`: The offset position along the x-axis, an integer value in centimeters.
                - `y`: The offset position along the y-axis, an integer value in centimeters.
                - `z`: The offset position along the z-axis, an integer value in centimeters.
            - `relative_orientation`: A dictionary indicating the orientation of the object relative to the robot arm in degrees.
                - `theta_x`: The offset rotation around the x-axis, an integer value in degrees.
                - `theta_y`: The offset rotation around the y-axis, an integer value in degrees.
                - `theta_z`: The offset rotation around the z-axis, an integer value in degrees.'''
        else:
            
            object_prompt = '''
            - `position`: A dictionary indicating the coordinates of the object's position in centimeters.
                - `x`: The position along the x-axis, an integer value in centimeters.
                - `y`: The position along the y-axis, an integer value in centimeters.
                - `z`: The position along the z-axis, an integer value in centimeters.
            - `orientation`: A dictionary indicating the orientation of the object in degrees.
                - `theta_x`: The rotation around the x-axis, an integer value in degrees ranging from 0 to 360.
                - `theta_y`: The rotation around the y-axis, an integer value in degrees ranging from 0 to 360.
                - `theta_z`: The rotation around the z-axis, an integer value in degrees ranging from 0 to 360.'''
        directional_information_prompt = '''**Directional Information:**

    - **X-Axis:**  
    - Positive x: Forward (away from the robot)  
    - Negative x: Backward (towards the robot)  

    - **Y-Axis:**  
    - Positive y: To the left of the robot  
    - Negative y: To the right of the robot  

    - **Z-Axis:**  
    - Positive z: Up  
    - Negative z: Down 

    - **Theta X:**  
    - Positive x: Pitch down
    - Negative x: Pitch up

    - **Theta Y:**  
    - Positive y: Roll left
    - Negative y: Roll right

    - **Theta Z:**  
    - Positive z: Yaw left
    - Negative z: Yaw right  
        '''

    if natural_languages:
        gripper_prompt = 'A string `open` or `closed` indicating whether the gripper is open or closed.' 
        gripper_key = 'gripper'
    else:
        gripper_key = 'gripper opening'
        if binary_gripper:
            gripper_prompt = 'A boolean value indicating whether the gripper is open or closed. `1` means the gripper is open and `0` means the gripper is closed.'
        else:
            gripper_prompt = f'An integer value ranging from 0 to {gripper_discrete_nums - 1}, indicating the state of the gripper. `{gripper_discrete_nums - 1}` means the gripper is fully open, and `0` means the gripper is fully closed.'
        
    
    prompt_data_structures = f'''
    **Data Structures:**
    1. **Current State of the Robot Arm:**
    - **Type:** Dictionary
    - **Keys:**{robot_pos_prompt}
        - `{gripper_key}`: {gripper_prompt}

    2. **Object Information:**
    - **Type:** Dictionary
    - **Keys:** The object type as a string.
    - **Values:** 
        - A dictionary containing:{object_prompt}
        
    {coordinate_system_prompt}
    {directional_information_prompt}  
    '''

    prompt_orientation_examples = '''
    **Orientation Examples:**

    1. **Gripper pointing towards positive x (Approaching from front, away from the robot):**
    - `[90, 0, 90]`: Fingers' opening/closing direction aligned with y-axis.
    - `[90, 90/-90, 90]`: Fingers' opening/closing direction aligned with z-axis.

    2. **Gripper pointing towards negative z (Approaching from the top):**
    - `[180, 0, 90]`: Fingers' opening/closing direction aligned with y-axis.
    - `[180, 0, 0/180]`: Fingers' opening/closing direction aligned with x-axis.

    3. **Gripper pointing towards positive y (Approaching from the right side):**
    - `[90, 0, 180]`: Fingers' opening/closing direction aligned with x-axis.
    - `[90, 90/-90, 180]`: Fingers' opening/closing direction aligned with z-axis.

    4. **Gripper pointing towards negative y (Approaching from the left side):**
    - `[90, 0, 0]`: Fingers' opening/closing direction aligned with x-axis.
    - `[90, 90/-90, 0]`: Fingers' opening/closing direction aligned with z-axis.

    '''

    if use_orientation_examples:
        prompt_data_structures += prompt_orientation_examples

    return prompt_data_structures


prompt_distance_adjustment_rules = '''
    **Distance Adjustment Rules:**

    - **When you need to increase the distance between two coordinates:**
    - If you are adjusting the smaller coordinate, you should subtract a delta from the smaller coordinate.
    - If you are adjusting the larger coordinate, you should add a delta to the larger coordinate.

    - **When you need to decrease the distance between two coordinates:**
    - If you are adjusting the smaller coordinate, you should add a delta to the smaller coordinate.
    - If you are adjusting the larger coordinate, you should substract a delta from the larger coordinate.
    ''' 


def generate_output_format(always_opposite, gripper_mode):
    if gripper_mode:
        first_group_identifiers_prompt = 'A/B/C/D'
    else:
        first_group_identifiers_prompt = 'A/B/C'

    if always_opposite:
        output_format = f'''{{
    "Group 1": "{first_group_identifiers_prompt}: {{corresponding most likely action from group 1}}",
    "Group 2": "A/B/C: {{corresponding most likely action from group 2}}",
    }}'''

    else:
        output_format = f'''{{
    "Group 1": "{first_group_identifiers_prompt}: {{corresponding most likely action from group 1}}",
    "Group 2": "{first_group_identifiers_prompt}: {{corresponding most likely action from group 2}}",
    "Group 3": "A/B/C: {{corresponding most likely action from group 3}}",
    "Group 4": "A/B/C: {{corresponding most likely action from group 4}}",
    }}'''
    
    return output_format



def generate_prompt_task_specifications(always_opposite, natural_languages, gripper_mode, analyze_examples = False):
    if always_opposite:
        if natural_languages:
            action_groups = [
            '''
    **Group 1:**
    - A: Either move forward or move backward
    - B: Either move up or move down
    - C: Either rotate up or rotate down
    ''',

    '''
    **Group 2:**
    - A: Either move left or move right
    - B: Either tilt left or tilt right
    - C: Either rotate left or rotate right
    '''
            ]
        else:
            action_groups = [
            '''
    **Group 1:**
    - A: Adjust x (either increase or decrease)
    - B: Adjust z (either increase or decrease)
    - C: Adjust theta x (either increase or decrease)
    ''',

    '''
    **Group 2:**
    - A: Adjust y (either increase or decrease)
    - B: Adjust theta y (either increase or decrease)
    - C: Adjust theta z (either increase or decrease)
    '''
            ]

        if gripper_mode:
            if natural_languages:
                action_groups[0]+= '''- D: Either open or close gripper
            '''
            else:
                action_groups[0]+= '''- D: Adjust gripper (either open or close)
            '''
            additional_identifier_prompt = 'Group 1 also includes an additional action labeled as D.'
            identifiers_prompt = 'A, B, C, or D for group 1, and A, B, or C for group 2'
        else:
            additional_identifier_prompt = ''
            identifiers_prompt = 'A, B, or C'




    else:
        if natural_languages:
            action_groups = [
            '''
    **Group 1:**
    - A: Move forward
    - B: Move up
    - C: Rotate up
    ''',

    '''
    **Group 2:**
    - A: Move backward
    - B: Move down
    - C: Rotate down
    ''',

    '''
    **Group 3:**
    - A: Move left
    - B: Tilt left
    - C: Rotate left
    ''',

    '''
    **Group 4:**
    - A: Move right
    - B: Tilt right
    - C: Rotate right
    '''

            ]
        else:
            action_groups = [
            '''
    **Group 1:**
    - A: Increase x
    - B: Increase z
    - C: Increase theta x
    ''',

    '''
    **Group 2:**
    - A: Decrease x
    - B: Decrease z
    - C: Decrease theta x
    ''',

    '''
    **Group 3:**
    - A: Increase y
    - B: Increase theta y
    - C: Increase theta z
    ''',

    '''
    **Group 4:**
    - A: Decrease y
    - B: Decrease theta y
    - C: Decrease theta z
    '''
            ]
   
        if gripper_mode:
            action_groups[0]+= '''- D: Open gripper
            '''
            action_groups[1]+='''- D: Close gripper
            '''
            additional_identifier_prompt = 'Groups 1 and 2 also include an additional action labeled as D.'
            identifiers_prompt = 'A, B, C, or D for groups 1 and 2, and A, B, or C for groups 3 and 4'
        else:
            additional_identifier_prompt = ''
            identifiers_prompt = 'A, B, or C'


    if analyze_examples:
        output_prompt = ''
        head_prompt = f'''
    The task of the agent you are trying to help is to determine the most likely actions from each of the following groups, based on the provided current robot state and object information:'''
    else:
        head_prompt = f'''
    **Task:**

    Based on the provided information and the current task, robot state, and object information, determine the most likely actions from each of the following groups. For each group, the actions are labeled with identifiers A, B, and C for clarity. {additional_identifier_prompt}'''


        output_format = generate_output_format(always_opposite, gripper_mode)
        output_prompt = f'''**Output Requirements:**  
    Your output should be a dictionary where each key represents a group, and the corresponding value is the most likely action's letter identifier ({identifiers_prompt}) followed by the corresponding action description. The output should look like this, do not output any additional analysis:
    
    {output_format}
    {''.join(action_groups)}
    ---

    '''
    #output
    prompt_task_specification = f'''
    {head_prompt}
    ---

    **Definition of Most Likely Actions:**  
    Most likely actions refer to the actions that have the highest probability of successfully achieving the task objectives based on the current state of the robot arm, information of objects around, and the specified action groups. These actions should be determined by evaluating the robot's ability to manipulate objects effectively and efficiently according to the given criteria.

    ---

    {output_prompt}'''
    return prompt_task_specification


def get_spatial_prompt(obj_pos_component, pos_component, prompts, approximate):
    if abs(obj_pos_component - pos_component) <= approximate:
        return prompts[2]
    else:
        if obj_pos_component > pos_component:
            return prompts[0]
        else:
            return prompts[1]
   
        
def get_approximate_num(num, approximate, decimal):
    if decimal == 0:
        approximate_num = round(num/approximate) * approximate
        # if approximate_num == -360:
        #     approximate_num = 0
        if abs(approximate_num) >= 350:
            approximate_num = 0   
    else:
        approximate_num = round(num/approximate, decimal) * approximate
        # if approximate_num == format(-360, f".{decimal}f"):
        #     approximate_num = format(0, f".{decimal}f")
        if abs(approximate_num) >= 350:
            approximate_num = format(0, f".{decimal}f")

    return approximate_num


def generate_prompt_current_task(task, always_opposite, object_poses, pos, position_approximate, orientation_approximate, decimal, natural_languages, use_robot_location, relative, holding_prompt, holding_object, dropped_object, binary_gripper, natural_languages_prompts, gripper_mode, output = None, example_index = None):
    print("poses in prompt")
    print(pos)
    print("object poses in prompt")
    print(object_poses)
    if output is not None:
        assert example_index is not None
    
    if use_robot_location:
        robot_pos_prompt = f'''
        "position": {{
            "x": {get_approximate_num(pos[0], position_approximate, decimal)},         
            "y": {get_approximate_num(pos[1], position_approximate, decimal)},
            "z": {get_approximate_num(pos[2], position_approximate, decimal)}
        }},
        "orientation": {{
            "theta x": {get_approximate_num(pos[3], orientation_approximate, decimal)},
            "theta y": {get_approximate_num(pos[4], orientation_approximate, decimal)},
            "theta z": {get_approximate_num(pos[5], orientation_approximate, decimal)}
        }}'''
    else:
        robot_pos_prompt = ''
      
    
    obj_information_prompt = ""
    if natural_languages:
        for obj, obj_pos in object_poses.items():
            if holding_prompt and obj == holding_object:
                obj_information_prompt += f'''
        "{obj}": {{
            "relative_pos":"{list(natural_languages_prompts.values())[-1].replace("object", obj)}",    
        }},'''
            elif holding_prompt and obj == dropped_object:
                obj_information_prompt += f'''
        "{obj}": {{
            "relative_pos":"has been dropped",    
        }},'''
            else:
                spatial_prompt_x = get_spatial_prompt(obj_pos[0], pos[0], list(natural_languages_prompts.values())[0], position_approximate)
                spatial_prompt_y = get_spatial_prompt(obj_pos[1], pos[1], list(natural_languages_prompts.values())[1], position_approximate)
                spatial_prompt_z = get_spatial_prompt(obj_pos[2], pos[2], list(natural_languages_prompts.values())[2], position_approximate)
                spatial_prompt_thetax = get_spatial_prompt(obj_pos[3], pos[3], list(natural_languages_prompts.values())[3], orientation_approximate)
                spatial_prompt_thetay = get_spatial_prompt(obj_pos[4], pos[4], list(natural_languages_prompts.values())[4], orientation_approximate)
                spatial_prompt_thetaz = get_spatial_prompt(obj_pos[5], pos[5], list(natural_languages_prompts.values())[5], orientation_approximate)

                obj_information_prompt += f'''
        "{obj}": {{
            "relative_pos":{{
                "relative_position":{{
                    "{list(natural_languages_prompts.keys())[0]}": "{spatial_prompt_x}",
                    "{list(natural_languages_prompts.keys())[1]}": "{spatial_prompt_y}",
                    "{list(natural_languages_prompts.keys())[2]}": "{spatial_prompt_z}",
                }},
                "relative_orientation":{{
                    "{list(natural_languages_prompts.keys())[3]}": "{spatial_prompt_thetax}",
                    "{list(natural_languages_prompts.keys())[4]}": "{spatial_prompt_thetay}",
                    "{list(natural_languages_prompts.keys())[5]}": "{spatial_prompt_thetaz}",
                }},
            }}
        }},'''
        
    else:
        if relative:
            for obj, obj_pos in object_poses.items():
                obj_information_prompt += f'''
        "{obj}": {{
            "relative_position": {{
                "x": {get_approximate_num(obj_pos[0] - pos[0], position_approximate, decimal)},
                "y": {get_approximate_num(obj_pos[1] - pos[1], position_approximate, decimal)},
                "z": {get_approximate_num(obj_pos[2] - pos[2], position_approximate, decimal)}
            }},
            "relative_orientation": {{
                "theta x": {get_approximate_num(obj_pos[3] - pos[3], orientation_approximate, decimal)},
                "theta y": {get_approximate_num(obj_pos[4] - pos[4], orientation_approximate, decimal)},
                "theta z": {get_approximate_num(obj_pos[5] - pos[5], orientation_approximate, decimal)}
            }}
        }},'''
        else:
            for obj, obj_pos in object_poses.items():
                obj_information_prompt += f'''
        "{obj}": {{
            "position": {{
                "x": {get_approximate_num(obj_pos[0], position_approximate, decimal)},
                "y": {get_approximate_num(obj_pos[1], position_approximate, decimal)},
                "z": {get_approximate_num(obj_pos[2], position_approximate, decimal)}
            }},
            "orientation": {{
                "theta x": {get_approximate_num(obj_pos[3], orientation_approximate, decimal)},
                "theta y": {get_approximate_num(obj_pos[4], orientation_approximate, decimal)},
                "theta z": {get_approximate_num(obj_pos[5], orientation_approximate, decimal)}
            }}
        }},'''


    if output is None:
        first_line_prompt = '### Current Task, Robot Arm State, and Object Information:  '
        output_format = generate_output_format(always_opposite, gripper_mode)
        final_prompt = f'''- **Output (do not output any additional analysis):**  
    {output_format}'''
    
    else:
        first_line_prompt = f'**Example {example_index}:**  '
        final_prompt = f'''- **Most Likely Action(s):**  
    {output}'''

    if natural_languages:
        gripper_key = 'gripper'
        if int(pos[6]) == 0:
            gripper_state = 'closed'
        else:
            assert int(pos[6]) == 1
            gripper_state = 'open'
    else:
        gripper_key = 'gripper opening'
        gripper_state = int(pos[6])

    prompt_current_task = f'''
    {first_line_prompt} 

    - **Current Task:** {task}

    - **Current State of the Robot Arm:**  
    {{{robot_pos_prompt}
        "{gripper_key}": {gripper_state}
    }}

    - **Current Object Information:**  
    {{{obj_information_prompt}
    }}

    {final_prompt}
            
    '''
    return prompt_current_task



def generate_examples_prompt(examples_prompt_all, analyze_examples = False):
    if analyze_examples:
        pre_examples_prompt = ''
    else:
        pre_examples_prompt = '''
    Each example below will only provide the most likely action(s) for one or some of the groups. You should use this format to understand how to predict actions for all groups. Your output should always include the most likely action for ALL four groups.'''
    
    examples_prompt = f'''
    ### Examples:{pre_examples_prompt}
    {examples_prompt_all}
    '''
    return examples_prompt


prompt_objective_summarize = '''
    **Objective:**  
    You will be given examples of task instructions, poses of a robot arm, and information of objects around it. 
    Your goal is to analyze the examples and summarize the patterns or rules, which will be used to assist another agent to predict the most likely actions out of the specified groups of actions in similar scenarios of the same task.    
    '''



prompt_summary_rules = '''
    **Summary Task:**

    You have been provided with several examples that contain robot arm states, object information, and the corresponding actions that were determined as the most likely. Your goal is to analyze these examples and summarize the underlying patterns or rules that can be applied to predict the most likely actions in similar situations. These rules should take into account the relative positions and orientations of the robot arm and objects.

    **Instructions:**

    1. **Analyze the Examples:**
    - Review each example carefully.
    - Identify the relationship between the task information, robot arm's states, object information, and the chosen actions.
    - Avoid referring to examples by number (e.g., "Example x"). Focus on describing the relationships between the objects and the robot arm in terms of position, orientation, and gripper state.

    2. **Identify Patterns:**
    - Determine the common factors that influence the selection of each action group.
    - Specify the conditions under which a particular action is preferred, referencing specific object names rather than using broad terms like "object" or "target object".
    - Describe how each object's position and orientation relative to the robot arm influences the chosen action. If necessary, include the gripper state of the robot arm in these rules.   

    3. **Summarize the Rules:**
    - Formulate clear and concise rules that capture the patterns you've identified.
    - Ensure that the rules are specific, mentioning the relationships between the robot's position, orientation, gripper state, and the (relative) positions and orientations of all relevant objects.
    - Avoid vague language; all terms should be well-defined based on the examples provided. Do not reference examples by number, instead use specific positional and orientational details to justify the rules.

    **Output Format:**

    - Your output should be a list of summarized rules.
    - Each rule should be clearly stated and should include references to the specific objects involved, the relationships between the robot's position, orientation, gripper state, and the (relative) positions and orientations of all relevant objects. The rules should be actionable and applicable to similar scenarios.

    **Note:**
    The rules you generate will be used to inform the AI agent's decision-making process, so they should be both comprehensive and actionable.
    '''


def generate_provided_rules_prompt(rules):
    prompt_provide_rules = f'''
    Below are a set of rules derived from previous examples. These rules summarize the patterns identified between task information, robot arm's state, object information, and the chosen actions. Your task is to apply these rules to predict the most likely actions out of the specified groups for the current situation.

    {rules}
    '''
    return prompt_provide_rules



prompt_summarize_singel_example = '''
    **Summary Task:**

    You have been provided with an example that contain robot arm states, object information, and the corresponding action(s) that were determined as the most likely. Your goal is to analyze the example and summarize an underlying pattern or rule that can be applied to predict the most likely actions in similar situations. The rule should take into account the relative positions and orientations of the robot arm and objects.

    **Instructions:**

    1. **Analyze the Example:**
    - Review the example carefully.
    - Identify the relationship between the task information, robot arm's states, object information, and the chosen actions.
    - Avoid referring to examples by number (e.g., "Example x"). Focus on describing the relationships between the objects and the robot arm in terms of position, orientation, and gripper state.

    2. **Identify Patterns:**
    - Determine the factors that influence the selection of each action group.
    - Specify the conditions under which a particular action is preferred, referencing specific object names rather than using broad terms like "object" or "target object".
    - Describe how each object's position and orientation relative to the robot arm influences the chosen action. If necessary, include the gripper state of the robot arm in these rules.   

    3. **Summarize the Rule:**
    - Formulate a clear and concise rule that capture the patterns you've identified.
    - Ensure that the rule is specific, mentioning the relationships between the robot's position, orientation, gripper state, and the (relative) positions and orientations of all relevant objects.
    - Avoid vague language; all terms should be well-defined based on the example provided. Do not reference the example by number, instead use specific positional and orientational details to justify the rules.

    **Output Format:**

    - Your output should be a single summarized rules.
    - The rule should be clearly stated and should include references to the specific objects involved, the relationships between the robot's position, orientation, gripper state, and the (relative) positions and orientations of all relevant objects. The rule should be actionable and applicable to similar scenarios.

    **Note:**
    The rule you generate will be used to inform the AI agent's decision-making process, so they should be both comprehensive and actionable.
    '''