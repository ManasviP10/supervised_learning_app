"""
------------------------------------------------------
Supervised Learning App
main.py

Starts the GUI application.
------------------------------------------------------
"""

import tkinter as tk
from gui import SupervisedLearningGUI


def main():
    # Create main window
    root = tk.Tk()

    # Initialize GUI
    app = SupervisedLearningGUI(root)

    # Start application
    root.mainloop()


if __name__ == "__main__":
    main()