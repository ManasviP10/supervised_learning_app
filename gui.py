"""
------------------------------------------------------
Supervised Learning App
gui.py

GUI using Tkinter
------------------------------------------------------
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ml_engine import run_ml_pipeline


class SupervisedLearningGUI:

    def __init__(self, root):

        self.root = root
        self.root.title("Supervised Learning App")
        self.root.geometry("700x500")
        self.root.resizable(False, False)

        self.train_file = ""
        self.test_file = ""

        self.problem_type = tk.StringVar(value="classification")

        self.build_gui()

    # -------------------------------------------------
    # GUI Layout
    # -------------------------------------------------

    def build_gui(self):

        title = tk.Label(
            self.root,
            text="SUPERVISED LEARNING APP",
            font=("Arial", 18, "bold")
        )

        title.pack(pady=15)

        #################################################
        # Training CSV
        #################################################

        train_frame = tk.Frame(self.root)
        train_frame.pack(fill="x", padx=20, pady=10)

        tk.Label(
            train_frame,
            text="Training CSV",
            width=18,
            anchor="w"
        ).pack(side="left")

        self.train_label = tk.Label(
            train_frame,
            text="No file selected",
            anchor="w",
            fg="blue"
        )

        self.train_label.pack(side="left", padx=5)

        tk.Button(
            train_frame,
            text="Browse",
            command=self.browse_train
        ).pack(side="right")

        #################################################
        # Test CSV
        #################################################

        test_frame = tk.Frame(self.root)
        test_frame.pack(fill="x", padx=20, pady=10)

        tk.Label(
            test_frame,
            text="Test CSV",
            width=18,
            anchor="w"
        ).pack(side="left")

        self.test_label = tk.Label(
            test_frame,
            text="No file selected",
            anchor="w",
            fg="blue"
        )

        self.test_label.pack(side="left", padx=5)

        tk.Button(
            test_frame,
            text="Browse",
            command=self.browse_test
        ).pack(side="right")

        #################################################
        # Problem Type
        #################################################

        option_frame = tk.LabelFrame(
            self.root,
            text="Problem Type",
            padx=10,
            pady=10
        )

        option_frame.pack(fill="x", padx=20, pady=15)

        tk.Radiobutton(
            option_frame,
            text="Classification",
            variable=self.problem_type,
            value="classification"
        ).pack(anchor="w")

        tk.Radiobutton(
            option_frame,
            text="Regression",
            variable=self.problem_type,
            value="regression"
        ).pack(anchor="w")

        #################################################
        # Start Button
        #################################################

        self.start_button = tk.Button(
            self.root,
            text="START",
            bg="#4CAF50",
            fg="white",
            width=20,
            height=2,
            font=("Arial", 11, "bold"),
            command=self.start_process
        )

        self.start_button.pack(pady=20)

        #################################################
        # Progress Bar
        #################################################

        self.progress = ttk.Progressbar(
            self.root,
            orient="horizontal",
            mode="indeterminate",
            length=500
        )

        self.progress.pack(pady=10)

        #################################################
        # Status
        #################################################

        self.status = tk.Label(
            self.root,
            text="Status : Waiting...",
            font=("Arial", 10),
            fg="darkgreen"
        )

        self.status.pack(pady=10)

    # -------------------------------------------------
    # Browse Functions
    # -------------------------------------------------

    def browse_train(self):

        filename = filedialog.askopenfilename(
            title="Select Training CSV",
            filetypes=[("CSV Files", "*.csv")]
        )

        if filename:
            self.train_file = filename
            self.train_label.config(text=os.path.basename(filename))

    def browse_test(self):

        filename = filedialog.askopenfilename(
            title="Select Test CSV",
            filetypes=[("CSV Files", "*.csv")]
        )

        if filename:
            self.test_file = filename
            self.test_label.config(text=os.path.basename(filename))

    # -------------------------------------------------
    # Start ML
    # -------------------------------------------------

    def start_process(self):

        if self.train_file == "":
            messagebox.showerror(
                "Error",
                "Please select training CSV."
            )
            return

        if self.test_file == "":
            messagebox.showerror(
                "Error",
                "Please select test CSV."
            )
            return

        self.start_button.config(state="disabled")

        self.progress.start()

        self.status.config(
            text="Status : Training models..."
        )

        thread = threading.Thread(
            target=self.run_pipeline
        )

        thread.daemon = True
        thread.start()

    # -------------------------------------------------
    # Run Pipeline
    # -------------------------------------------------

    def run_pipeline(self):

        try:

            run_ml_pipeline(
                train_path=self.train_file,
                test_path=self.test_file,
                problem_type=self.problem_type.get()
            )

            self.root.after(
                0,
                self.training_success
            )

        except Exception as e:
            self.root.after(
                0,
                lambda err=e: self.training_failed(err)
            )

    # -------------------------------------------------
    # Success
    # -------------------------------------------------

    def training_success(self):

        self.progress.stop()

        self.start_button.config(state="normal")

        self.status.config(
            text="Status : Completed Successfully!"
        )

        messagebox.showinfo(
            "Success",
            "Training completed!\n\n"
            "Files generated:\n"
            "output/output.csv\n"
            "output/report.txt"
        )

    # -------------------------------------------------
    # Failed
    # -------------------------------------------------

    def training_failed(self, error):

        self.progress.stop()

        self.start_button.config(state="normal")

        self.status.config(
            text="Status : Failed"
        )

        messagebox.showerror(
            "Error",
            str(error)
        )