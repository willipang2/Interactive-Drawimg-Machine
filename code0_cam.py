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

# Global variables
points = []
a4_transform = None
a4_reference = None
small_paper_mode = False
quit_program = False
current_frame = None
frame_lock = threading.Lock()
detection_result = None
detection_info = None
show_detection_result = False

# New global variables for empty area
empty_area_points = []
empty_area_transform = None
empty_area_reference = None
empty_area_defined = False
empty_area_size = (400, 600)  # Default size, will adjust based on actual area
empty_area_width_mm = 210  # Default values, will be updated when defined
empty_area_height_mm = 297
empty_area_x_mm = 0
empty_area_y_mm = 0


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
    global points, empty_area_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if small_paper_mode and not empty_area_defined and len(empty_area_points) < 4:
            # Converting click coordinates to A4 space
            src_pts = np.array([[[x, y]]], dtype=np.float32)
            dst_pts = cv2.perspectiveTransform(src_pts, a4_transform)
            a4_x, a4_y = dst_pts[0][0]
            # Guide the user on point ordering
            if len(empty_area_points) == 0:
                print("Defining top-left corner of empty area")
            elif len(empty_area_points) == 1:
                print("Defining top-right corner of empty area")
            elif len(empty_area_points) == 2:
                print("Defining bottom-right corner of empty area")
            elif len(empty_area_points) == 3:
                print("Defining bottom-left corner of empty area")
            empty_area_points.append((int(a4_x), int(a4_y)))
            # Draw on the transformed view
            warped = cv2.warpPerspective(param['frame'], a4_transform, a4_size)
            cv2.circle(warped, (int(a4_x), int(a4_y)), 5, (0, 0, 255), -1)
            if len(empty_area_points) >= 2:
                cv2.line(
                    warped, empty_area_points[-2], empty_area_points[-1], (0, 0, 255), 2)
            if len(empty_area_points) == 4:
                cv2.line(
                    warped, empty_area_points[3], empty_area_points[0], (0, 0, 255), 2)
            cv2.imshow("A4 Reference", warped)
        elif not small_paper_mode and len(points) < 4:
            # Original A4 definition behavior
            points.append((x, y))
            cv2.circle(param['frame'], (x, y), 5, (0, 255, 0), -1)
            if len(points) >= 2:
                cv2.line(param['frame'], points[-2],
                         points[-1], (0, 255, 0), 2)
            if len(points) == 4:
                cv2.line(param['frame'], points[3], points[0], (0, 255, 0), 2)
            cv2.imshow(param['window_name'], param['frame'])


def define_empty_area():
    global empty_area_defined, empty_area_transform, empty_area_reference
    global empty_area_size, empty_area_width_mm, empty_area_height_mm
    global empty_area_x_mm, empty_area_y_mm
    if len(empty_area_points) == 4:
        # Calculate width and height of the empty area
        # Using basic distance formula between points for estimation
        width1 = np.sqrt((empty_area_points[1][0] - empty_area_points[0][0])**2 +
                         (empty_area_points[1][1] - empty_area_points[0][1])**2)
        width2 = np.sqrt((empty_area_points[2][0] - empty_area_points[3][0])**2 +
                         (empty_area_points[2][1] - empty_area_points[3][1])**2)
        height1 = np.sqrt((empty_area_points[3][0] - empty_area_points[0][0])**2 +
                          (empty_area_points[3][1] - empty_area_points[0][1])**2)
        height2 = np.sqrt((empty_area_points[2][0] - empty_area_points[1][0])**2 +
                          (empty_area_points[2][1] - empty_area_points[1][1])**2)

        # Average to handle potential perspective distortion
        empty_width = int((width1 + width2) / 2)
        empty_height = int((height1 + height2) / 2)
        empty_area_size = (empty_width, empty_height)

        # Calculate dimensions in mm based on A4 proportions (A4 is 210mm × 297mm)
        ratio_width = empty_width / a4_width
        ratio_height = empty_height / a4_height
        empty_area_width_mm = 210 * ratio_width
        empty_area_height_mm = 297 * ratio_height

        # Calculate position of empty area's top-left corner in A4 space (in mm)
        empty_area_x_mm = (empty_area_points[0][0] / a4_width) * 210
        empty_area_y_mm = (empty_area_points[0][1] / a4_height) * 297

        # Add 1cm margin to the empty area (inward)
        margin_mm = 10  # 1cm = 10mm
        empty_area_with_margin_width_mm = empty_area_width_mm - (2 * margin_mm)
        empty_area_with_margin_height_mm = empty_area_height_mm - \
            (2 * margin_mm)
        empty_area_with_margin_x_mm = empty_area_x_mm + margin_mm
        empty_area_with_margin_y_mm = empty_area_y_mm + margin_mm

        # Update the JSON file with the location of the empty area (with margin)
        position_data = {
            "width_mm": empty_area_with_margin_width_mm,
            "height_mm": empty_area_with_margin_height_mm,
            "position_x_mm": empty_area_with_margin_x_mm,
            # Fixed: was incorrectly using empty_area_y_mm
            "position_y_mm": empty_area_with_margin_y_mm,
            "center_x_mm": empty_area_with_margin_x_mm + empty_area_with_margin_width_mm / 2,
            "center_y_mm": empty_area_with_margin_y_mm + empty_area_with_margin_height_mm / 2
        }

        with open("small_canvas_info.json", "w") as f:
            json.dump(position_data, f, indent=4)

        # Create perspective transform for empty area
        src_pts = np.array(empty_area_points, dtype=np.float32)
        dst_pts = np.array([
            [0, 0],
            [empty_width, 0],
            [empty_width, empty_height],
            [0, empty_height]
        ], dtype=np.float32)
        empty_area_transform = cv2.getPerspectiveTransform(src_pts, dst_pts)

        empty_area_defined = True
        return True
    return False


def detect_paper_by_color(image, reference):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_image, gray_reference)
    # Reduced threshold for better sensitivity
    _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)  # Reduced kernel size
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    valid_contours = [c for c in contours if cv2.contourArea(
        c) > 3000]  # Reduced min area
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


def detect_paper_in_empty_area(image, reference):
    result_image, rect, rotated_dims = detect_paper_by_color(image, reference)
    if rect is not None and rotated_dims is not None:
        x, y, w, h = rect
        width, height, area_px = rotated_dims
        # Convert to mm based on empty area dimensions
        width_ratio = width / empty_area_size[0]
        height_ratio = height / empty_area_size[1]
        paper_width_mm = empty_area_width_mm * width_ratio
        paper_height_mm = empty_area_height_mm * height_ratio

        # Save detection image - moved from process_paper_detection
        cv2.imwrite("paper_detection.jpg", result_image)

        return result_image, rect, (paper_width_mm, paper_height_mm, paper_width_mm * paper_height_mm)
    return None, None, None


# Main program execution
a4_width = 794
a4_height = 1123
a4_size = (a4_width, a4_height)

# Initialize camera
cap = cv2.VideoCapture(2)
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
print("After pressing 'p', you can define an empty area within A4:")
print("- Press 'e' to begin defining empty area")
print("- Click 4 points within the A4 reference (in order: top-left, top-right, bottom-right, bottom-left)")
print("- Press 'd' to confirm the empty area - This will create the JSON with 1cm margin")
print("Then place a paper inside the empty area and press 'o' to detect.")
print("Press 'q' to quit.")

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
        if empty_area_defined:
            status_text = "Place paper in empty area & press 'o' to measure"
        elif len(empty_area_points) > 0:
            status_text = f"Define empty area: {len(empty_area_points)}/4 points"
        else:
            status_text = "Press 'e' to define empty area or 'o' for whole A4"

    cv2.putText(display_frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the main webcam feed
    cv2.imshow(window_name, display_frame)

    # If in small paper mode, also show the warped view
    if small_paper_mode:
        warped = cv2.warpPerspective(frame, a4_transform, a4_size)
        cv2.imshow("A4 Reference", warped)
        # If empty area is defined, show it as well
        if empty_area_defined:
            empty = cv2.warpPerspective(
                warped, empty_area_transform, empty_area_size)
            cv2.imshow("Empty Area Reference", empty)

    # Display detection results in the main thread
    if need_to_show_detection:
        if result_to_show is not None and info_to_show is not None:
            cv2.imshow("Paper Detection", result_to_show)
            # Print the detection information
            print(f"\nPaper dimensions:")
            if "width_px" in info_to_show:
                print(
                    f"Width: {info_to_show['width_px']:.1f} pixels ({info_to_show['width_mm']:.1f} mm)")
                print(
                    f"Height: {info_to_show['height_px']:.1f} pixels ({info_to_show['height_mm']:.1f} mm)")
            else:
                print(f"Width: {info_to_show['width_mm']:.1f} mm")
                print(f"Height: {info_to_show['height_mm']:.1f} mm")
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
        empty_area_points = []
        empty_area_defined = False
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
        print("A4 reference created. Press 'e' to define empty area or 'o' to detect in full A4.")
    elif key == ord('e') and small_paper_mode and not empty_area_defined:
        print("Click 4 points in A4 Reference to define empty area (top-left, top-right, bottom-right, bottom-left)")
        empty_area_points = []
    elif key == ord('d') and small_paper_mode and len(empty_area_points) == 4:
        if define_empty_area():
            print(
                "Empty area defined with 1cm margin. JSON file updated for drawing machine.")
            print(
                "The JSON file contains the location of the empty area with 1cm margin based on A4 reference.")
            # Initialize empty area reference
            warped_a4 = cv2.warpPerspective(frame, a4_transform, a4_size)
            empty = cv2.warpPerspective(
                warped_a4, empty_area_transform, empty_area_size)
            empty_area_reference = empty.copy()
        else:
            print("Failed to define empty area. Try again.")
    elif key == ord('o') and small_paper_mode:
        with frame_lock:
            frame_to_process = current_frame.copy()

        if empty_area_defined:
            # First transform to A4 space
            warped_a4 = cv2.warpPerspective(
                frame_to_process, a4_transform, a4_size)
            # Then transform to empty area
            warped_empty = cv2.warpPerspective(
                warped_a4, empty_area_transform, empty_area_size)
            result_image, rect, dims = detect_paper_in_empty_area(
                warped_empty, empty_area_reference)

            if result_image is not None:
                paper_width_mm, paper_height_mm, area_mm2 = dims
                # Create detection info for display
                info = {
                    "width_mm": paper_width_mm,
                    "height_mm": paper_height_mm,
                    "area_mm2": area_mm2,
                    "position": "Inside empty area"
                }

                # Determine closest paper size
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
                    w_diff = abs(paper_width_mm - w_mm)
                    h_diff = abs(paper_height_mm - h_mm)
                    total_diff = w_diff + h_diff
                    if total_diff < min_diff and total_diff < 30:  # Threshold for matching
                        min_diff = total_diff
                        closest_size = size_name

                info["closest_size"] = closest_size
                detection_result = result_image.copy()
                detection_info = info
                show_detection_result = True

                # Display results immediately
                cv2.imshow("Paper Detection", detection_result)
                print(f"\nPaper dimensions:")
                print(f"Width: {info['width_mm']:.1f} mm")
                print(f"Height: {info['height_mm']:.1f} mm")
                print(f"\nClosest standard size: {info['closest_size']}")
                if info['closest_size'] == "Custom":
                    print(
                        f"Custom size: {info['width_mm']:.1f}mm x {info['height_mm']:.1f}mm")
                print(f"Area: {info['area_mm2']:.1f} square mm")
            else:
                print(
                    "No paper detected in empty area. Make sure the paper is clearly visible.")
                detection_result = None
                detection_info = None
                show_detection_result = True
        else:
            # Original A4 detection code
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

                # Save detection image
                cv2.imwrite("paper_detection.jpg", result_image)

                # Prepare detection info for display
                info = {
                    "width_px": width,
                    "height_px": height,
                    "width_mm": small_paper_width_mm,
                    "height_mm": small_paper_height_mm,
                    "area_mm2": small_paper_width_mm * small_paper_height_mm,
                }

                # Determine closest paper size
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
                detection_result = result_image.copy()
                detection_info = info
                show_detection_result = True

                # Display results immediately
                cv2.imshow("Paper Detection", detection_result)
                print(f"\nPaper dimensions:")
                print(
                    f"Width: {info['width_px']:.1f} pixels ({info['width_mm']:.1f} mm)")
                print(
                    f"Height: {info['height_px']:.1f} pixels ({info['height_mm']:.1f} mm)")
                print(f"\nClosest standard size: {info['closest_size']}")
                if info['closest_size'] == "Custom":
                    print(
                        f"Custom size: {info['width_mm']:.1f}mm x {info['height_mm']:.1f}mm")
                print(f"Area: {info['area_mm2']:.1f} square mm")
            else:
                print(
                    "No paper detected. Make sure the paper is clearly visible and different from the background.")
                detection_result = None
                detection_info = None
                show_detection_result = True

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Program terminated.")
