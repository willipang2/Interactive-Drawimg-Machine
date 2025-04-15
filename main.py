#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 18:57:06 2025

@author: callerw

This is the all_in_one run script.
"""

import subprocess
import os
import time
import threading


def monitor_for_detection():
    """Monitor for the creation of paper_detection.jpg file"""
    while True:
        if os.path.exists("paper_detection.jpg"):
            time.sleep(0.5)
            print("\nPaper detection completed, running code1_genImage.py...")

            # Run genImage.py and wait for it to complete
            main_process = subprocess.run(
                ["python", "code1_genImage.py"], check=False)

            if main_process.returncode == 0:
                print("code1_genImage.py completed successfully")

                os.remove("paper_detection.jpg")
                print("paper_detection.jpg has been deleted")

                print("Running code2_drawItOut...")
                additional_process = subprocess.run(
                    ["python", "code2_drawItOut.py"], check=False)

                if additional_process.returncode == 0:
                    print("code2_drawItOut.py completed successfully")
                else:
                    print("Error: drawItOut.py did not execute successfully")

                print("Waiting for new paper detection...")
            else:
                print("Error: genImage.py did not execute successfully")

        time.sleep(0.1)  # Check every 100ms


def run_scripts():
    print("Starting code0_cam.py...")

    # Start the monitoring thread
    monitor_thread = threading.Thread(target=monitor_for_detection)
    monitor_thread.daemon = True
    monitor_thread.start()

    # Run cam.py (this will block until cam.py exits)
    cam_process = subprocess.run(["python", "code0_cam.py"], check=False)

    if cam_process.returncode != 0:
        print("Error: code0_cam.py did not execute successfully")


if __name__ == "__main__":
    run_scripts()
