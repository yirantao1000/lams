"""
actions.py

This file defines mappings related to robotic actions (control modes). 

Key components:
1. **Action Groups**: Represents the mapping between joystick inputs and possible robot actions for 
   each mode. Each group contains a set of actions associated with a particular joystick movement.
2. **Action Control Commands**: Maps each action to a specific motion command represented as an array, 
   directly used to control the robot's movement or gripper state.
3. **Natural Language Support**: Provides natural language equivalents for action descriptions to enhance readability (these descriptions are shown to users).
4. **Utility Functions**: Includes helper functions like `find_action` to retrieve action details from mappings.
"""


import numpy as np
ACTION_GROUPS = {
    'Group 1': ['Increase x', 'Increase z', 'Increase theta x', 'Open gripper'],
    'Group 2': ['Decrease x', 'Decrease z', 'Decrease theta x', 'Close gripper'],
    'Group 3': ['Increase y', 'Increase theta y', 'Increase theta z'],
    'Group 4': ['Decrease y', 'Decrease theta y', 'Decrease theta z']
}

# Map actions to their corresponding group and index
action_to_group_index = {}

for group_index, (group, actions) in enumerate(ACTION_GROUPS.items()):
    for action_index, action in enumerate(actions):
        action_to_group_index[action] = (group_index, action_index)


# Find the group index and action index for a given action.
def find_action(action, action_map):
    return action_map.get(action, (None, None))


# Motion commands for each action, used for controlling the robot
ACTION_CORRESPONDENCES = {
    'Increase x': np.array([1,0,0,0,0,0]), 
    'Decrease x': np.array([-1,0,0,0,0,0]), 
    'Increase y': np.array([0,1,0,0,0,0]), 
    'Decrease y': np.array([0,-1,0,0,0,0]), 
    'Increase z': np.array([0,0,1,0,0,0]), 
    'Decrease z': np.array([0,0,-1,0,0,0]), 
    'Increase theta x': np.array([0,0,0,100,0,0]),
    'Decrease theta x': np.array([0,0,0,-100,0,0]),    
    'Increase theta y': np.array([0,0,0,0,0,-200]), 
    'Decrease theta y': np.array([0,0,0,0,0,200]), 
    'Increase theta z': np.array([0,0,0,0,100,0]),  
    'Decrease theta z': np.array([0,0,0,0,-100,0]),
    'Open gripper': np.array([1,1,1,1,1,1]),
    'Close gripper': np.array([-1,-1,-1,-1,-1,-1]),
}


ACTION_GROUPS_NATURAL_LANGUAGES = {
    'Group 1': ['Move forward', 'Move up', 'Pitch up', 'Open gripper'],
    'Group 2': ['Move backward', 'Move down', 'Pitch down', 'Close gripper'],
    'Group 3': ['Move left', 'Roll left', 'Yaw left'],
    'Group 4': ['Move right', 'Roll right', 'Yaw right']
}

ACTION_GROUPS_ALWAYS_OPPOSITE = {
    'Group 1': ['Adjust x (either increase or decrease)', 'Adjust z (either increase or decrease)', 'Adjust theta x (either increase or decrease)', 'Adjust gripper (either open or close)'],
    'Group 2': ['Adjust y (either increase or decrease)', 'Adjust theta y (either increase or decrease)', 'Adjust theta z (either increase or decrease)']
}

ACTION_GROUPS_ALWAYS_OPPOSITE_NATURAL_LANGUAGES = {
    'Group 1': ['Either move forward or move backward', 'Either move up or move down', 'Either pitch up or pitch down', 'Either open or close gripper'],
    'Group 2': ['Either move left or move right', 'Either roll left or roll right', 'Either yaw left or yaw right']
}