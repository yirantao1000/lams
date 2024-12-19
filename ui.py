"""
ui.py: UI Module for Real-Time Robot Action Visualization

This module provides a graphical user interface (GUI) for displaying real-time updates of robot control modes and associated states. Key functionalities include:

1. **Real-Time Arrow Display**:
   - Visualizes control modes with directional arrows and descriptive text.
   - Updates arrow labels and colors to reflect the robot's current control modes dynamically.

2. **Footer and Status Updates**:
   - Displays additional information, such as the currently grasped object, loading status, and the count of mode switches.

3. **Integration with Multithreading**:
   - Runs the GUI in a separate thread to ensure responsiveness and compatibility with the main robotic control processes.

4. **Utility Functions**:
   - Provides functions to update GUI elements, such as the displayed control modes, footer text, loading status, and mode switch count.
"""

import threading
import tkinter as tk
from tkinter import font

class RealtimeArrowDisplay:
    """
    Class for creating and managing the real-time arrow display UI.
    Visualizes control modes and related information using arrows and text.
    """
    def __init__(self, root):
        self.root = root
        n = 2.7

        self.canvas = tk.Canvas(root, width=int(300 * n), height=int(320 * n))
        self.canvas.pack()

        # Draw arrows for each joystick movement direction
        self.arrow_up = self.canvas.create_line(int(150 * n), int(50 * n), int(150 * n), int(10 * n), arrow=tk.LAST, width=int(5 * n), arrowshape=(20 * n, 25 * n, 10 * n))
        self.arrow_down = self.canvas.create_line(int(150 * n), int(250 * n), int(150 * n), int(290 * n), arrow=tk.LAST, width=int(5 * n), arrowshape=(20 * n, 25 * n, 10 * n))
        self.arrow_left = self.canvas.create_line(int(50 * n), int(150 * n), int(10 * n), int(150 * n), arrow=tk.LAST, width=int(5 * n), arrowshape=(20 * n, 25 * n, 10 * n))
        self.arrow_right = self.canvas.create_line(int(250 * n), int(150 * n), int(290 * n), int(150 * n), arrow=tk.LAST, width=int(5 * n), arrowshape=(20 * n, 25 * n, 10 * n))
        
        # Add text labels for arrows
        self.text_up = self.canvas.create_text(int(150 * n), int(60 * n), text="Var1", font=font.Font(family='Helvetica', size=int(12 * n), weight='bold'))
        self.text_down = self.canvas.create_text(int(150 * n), int(240 * n), text="Var2", font=font.Font(family='Helvetica', size=int(12 * n), weight='bold'))
        self.text_left = self.canvas.create_text(int(50 * n), int(170 * n), text="Var3", font=font.Font(family='Helvetica', size=int(12 * n), weight='bold'))
        self.text_right = self.canvas.create_text(int(250 * n), int(170 * n), text="Var4", font=font.Font(family='Helvetica', size=int(12 * n), weight='bold'))
        
        # Add footer text for additional information
        self.footer_text = self.canvas.create_text(int(150 * n), int(310 * n), text="Grasped Object: None", font=font.Font(family='Helvetica', size=int(12 * n), weight='bold'))

        # Add placeholders for loading status and mode switch count
        self.loading = self.canvas.create_text(int(150 * n), int(130 * n), text="", font=font.Font(family='Helvetica', size=int(12 * n), weight='bold'))
        self.ui_count = self.canvas.create_text(int(150 * n), int(150 * n), text=f"# mode switches: 0", font=font.Font(family='Helvetica', size=int(12 * n), weight='bold'))

        

    def update_values(self, var1, var2, var3, var4, colors):
        """
        Updates the text and colors for the arrows based on current control modes.
        """
        self.canvas.itemconfig(self.text_up, text=var1, fill = colors[0])
        self.canvas.itemconfig(self.text_down, text=var2, fill = colors[1])
        self.canvas.itemconfig(self.text_left, text=var3, fill = colors[2])
        self.canvas.itemconfig(self.text_right, text=var4, fill = colors[3])

    def update_footer(self, footer):
        self.canvas.itemconfig(self.footer_text, text=footer)

    def update_loading(self, show=True):
        if show:
            self.canvas.itemconfig(self.loading, text="loading...", fill = 'red')
        else:
            self.canvas.itemconfig(self.loading, text=f"")

    def update_count(self, count):
        self.canvas.itemconfig(self.ui_count, text=f"# mode switches: {count}")
        
    def run(self):
        self.root.mainloop()

def start_tkinter(app):
    app.run()

root = tk.Tk()
app = RealtimeArrowDisplay(root)


tkinter_thread = threading.Thread(target=start_tkinter, args=(app,))
tkinter_thread.daemon = True
tkinter_thread.start()


def show_actions(generated_action_names, colors):
    Readable_actions = {
        'Increase x': 'move forward', 
        'Decrease x': 'move backward', 
        'Increase y': 'move left', 
        'Decrease y': 'move right', 
        'Increase z': 'move up', 
        'Decrease z': 'move down', 
        'Increase theta x': 'rotate up',
        'Decrease theta x': 'rotate down',    
        'Increase theta y': 'roll left', 
        'Decrease theta y': 'roll right', 
        'Increase theta z': 'rotate left',  
        'Decrease theta z': 'rotate right',
        "Open gripper": "open gripper",
        "Close gripper": "close gripper",
        "Do nothing": "do nothing"

    }

    app.update_values(
        Readable_actions[generated_action_names[0]], 
        Readable_actions[generated_action_names[1]], 
        Readable_actions[generated_action_names[2]], 
        Readable_actions[generated_action_names[3]],
        colors
    )

    root.update_idletasks()
    root.update()

def show_footer(text):
    app.update_footer(f"Grasped Object: {text}")

    root.update_idletasks()
    root.update()


def show_loading(show=True):

    app.update_loading(show)

    root.update_idletasks()
    root.update()


def show_ui_count(count):

    app.update_count(count)

    root.update_idletasks()
    root.update()