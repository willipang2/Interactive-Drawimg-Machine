#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 17:45:06 2025

@author: callerw


the following code as the starting point, the start the cam, detect the ticket, ticket's size and location. 
ANd save the location as json.
"""

import cv2
import numpy as np
import threading
import time
import json

points = []
a4_transform = None
a4_reference = None
small_paper_mode = False
process_request = False
quit_program = False
current_frame = None
frame_lock = threading.Lock()

detection_result = None
detection_info = None
show_detection_result = False


def webcam_reader_thread(cap):
    global current_frame, quit_program, frame_lock
    while not quit_program:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                current_frame = frame.copy()
        else:
            time.sleep(1)  # Wait before retrying on error


def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            cv2.circle(param['frame'], (x, y), 5, (0, 255, 0), -1)
            if len(points) >= 2:
                cv2.line(param['frame'], points[-2],
                         points[-1], (0, 255, 0), 2)
            if len(points) == 4:
                cv2.line(param['frame'], points[3], points[0], (0, 255, 0), 2)
            cv2.imshow(param['window_name'], param['frame'])


def detect_paper_by_color(image, reference):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_image, gray_reference)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    valid_contours = [c for c in contours if cv2.contourArea(c) > 5000]
    if not valid_contours:
        return None, None, None
    largest_contour = max(valid_contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    width = rect[1][0]
    height = rect[1][1]
    if width > height:
        width, height = height, width
    result_image = image.copy()
    cv2.drawContours(result_image, [box], 0, (0, 0, 255), 2)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return result_image, (x, y, w, h), (width, height, area)


def process_paper_detection():
    global process_request, quit_program, current_frame, frame_lock
    global detection_result, detection_info, show_detection_result
    while not quit_program:
        if process_request:
            with frame_lock:
                if current_frame is not None:
                    frame_to_process = current_frame.copy()
                else:
                    process_request = False
                    continue
            current_warped = cv2.warpPerspective(
                frame_to_process, a4_transform, a4_size)
            result_image, rect, rotated_dims = detect_paper_by_color(
                current_warped, a4_reference)
            if rect is not None and rotated_dims is not None:
                x, y, w, h = rect
                width, height, area_px = rotated_dims
                # Calculate dimensions in mm
                width_ratio = width / a4_width
                height_ratio = height / a4_height
                small_paper_width_mm = 210 * width_ratio
                small_paper_height_mm = 297 * height_ratio
                # Save the image for displaying in main thread
                cv2.imwrite("paper_detection.jpg", result_image)
                info = {
                    "width_px": width,
                    "height_px": height,
                    "width_mm": small_paper_width_mm,
                    "height_mm": small_paper_height_mm,
                    "area_mm2": small_paper_width_mm * small_paper_height_mm,
                    "position": {
                        "x": x,
                        "y": y,
                        "center_x": x + width/2,
                        "center_y": y + height/2
                    }
                }
                position_data = {
                    "width_mm": small_paper_width_mm,
                    "height_mm": small_paper_height_mm,
                    "position_x_mm": (x / a4_width) * 210,  # Convert to mm
                    "position_y_mm": (y / a4_height) * 297,  # Convert to mm
                    "center_x_mm": ((x + width/2) / a4_width) * 210,
                    "center_y_mm": ((y + height/2) / a4_height) * 297
                }
                with open("small_canvas_info.json", "w") as f:
                    json.dump(position_data, f)
                paper_sizes = {
                    "A6": (105, 148.5),
                    "A5": (148.5, 210),
                    "A4": (210, 297),
                    "B6": (125, 176),
                    "B5": (176, 250)
                }
                closest_size = "Custom"
                min_diff = float('inf')
                for size_name, (w_mm, h_mm) in paper_sizes.items():
                    w_diff = abs(small_paper_width_mm - w_mm)
                    h_diff = abs(small_paper_height_mm - h_mm)
                    total_diff = w_diff + h_diff
                    if total_diff < min_diff and total_diff < 30:  # Threshold for matching
                        min_diff = total_diff
                        closest_size = size_name
                info["closest_size"] = closest_size
                with frame_lock:
                    detection_result = result_image.copy()
                    detection_info = info
                    show_detection_result = True
            else:
                print(
                    "No paper detected. Make sure the paper is clearly visible and different from the background.")
                with frame_lock:
                    detection_result = None
                    detection_info = None
                    show_detection_result = True
            process_request = False
        time.sleep(0.1)


a4_width = 794
a4_height = 1123
a4_size = (a4_width, a4_height)

# Initialize cam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

webcam_thread = threading.Thread(target=webcam_reader_thread, args=(cap,))
webcam_thread.daemon = True
webcam_thread.start()

window_name = "Webcam Feed"
cv2.namedWindow(window_name)

callback_params = {'window_name': window_name, 'frame': None}
cv2.setMouseCallback(window_name, mouse_callback, callback_params)

print("Click 4 points to define the A4 area.")
print("Press 'r' to reset points.")
print("Press 'p' to process the selected area.")
print("After pressing 'p', place a smaller paper on the A4 reference.")
print("Press 'o' to detect and calculate the smaller paper size.")
print("Press 'q' to quit.")

# Start the processing thread
detection_thread = threading.Thread(target=process_paper_detection)
detection_thread.daemon = True
detection_thread.start()

# Wait for the first frame
while current_frame is None and not quit_program:
    print("Waiting for first frame...")
    time.sleep(0.5)

while not quit_program:
    with frame_lock:
        if current_frame is None:
            print("Webcam disconnected. Trying to reconnect...")
            time.sleep(1)
            continue
        frame = current_frame.copy()

    need_to_show_detection = show_detection_result
    if need_to_show_detection:
        result_to_show = detection_result
        info_to_show = detection_info
        show_detection_result = False

    display_frame = frame.copy()
    callback_params['frame'] = display_frame

    # Draw the points and lines
    for i, point in enumerate(points):
        cv2.circle(display_frame, point, 5, (0, 255, 0), -1)
        if i > 0:
            cv2.line(display_frame, points[i-1], point, (0, 255, 0), 2)
    if len(points) == 4:
        cv2.line(display_frame, points[3], points[0], (0, 255, 0), 2)

    # Add status text
    status_text = "Define A4 corners"
    if small_paper_mode:
        status_text = "Place paper & press 'o' to measure"
    cv2.putText(display_frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the main webcam feed
    cv2.imshow(window_name, display_frame)

    # If in small paper mode, also show the warped view
    if small_paper_mode:
        warped = cv2.warpPerspective(frame, a4_transform, a4_size)
        cv2.imshow("A4 Reference", warped)

    # Display detection results in the main thread
    if need_to_show_detection:
        if result_to_show is not None and info_to_show is not None:
            cv2.imshow("Paper Detection", result_to_show)
            # Print the detection information
            print(f"\nSmaller paper dimensions:")
            print(
                f"Width: {info_to_show['width_px']:.1f} pixels ({info_to_show['width_mm']:.1f} mm)")
            print(
                f"Height: {info_to_show['height_px']:.1f} pixels ({info_to_show['height_mm']:.1f} mm)")
            print(f"\nClosest standard size: {info_to_show['closest_size']}")
            if info_to_show['closest_size'] == "Custom":
                print(
                    f"Custom size: {info_to_show['width_mm']:.1f}mm x {info_to_show['height_mm']:.1f}mm")
            print(f"Area: {info_to_show['area_mm2']:.1f} square mm")
        else:
            print(
                "No paper detected. Make sure the paper is clearly visible and different from the background.")

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        quit_program = True
        break
    elif key == ord('r'):
        points = []
    elif key == ord('p') and len(points) == 4 and not small_paper_mode:
        # Create perspective transform
        src_pts = np.array(points, dtype=np.float32)
        dst_pts = np.array([
            [0, a4_height],
            [a4_width, a4_height],
            [a4_width, 0],
            [0, 0]
        ], dtype=np.float32)
        a4_transform = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_a4 = cv2.warpPerspective(frame, a4_transform, a4_size)
        a4_reference = warped_a4.copy()
        cv2.imshow("A4 Reference", warped_a4)
        cv2.imwrite("a4_reference.jpg", warped_a4)
        print("A4 reference saved as 'a4_reference.jpg'")
        small_paper_mode = True
        print("A4 reference created. Now place a smaller paper on it and press 'o' to measure.")
    elif key == ord('o') and small_paper_mode:
        process_request = True

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Program terminated.")
