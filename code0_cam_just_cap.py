"""
Like cam.py, but this one only for deceted the ticket

"""

import cv2
import numpy as np
import threading
import time

# Global variables
points = []
empty_area_transform = None
empty_area_size = (400, 600)  # Default, will be set after 4 points
current_frame = None
frame_lock = threading.Lock()
quit_program = False


def webcam_reader_thread(cap):
    global current_frame, quit_program, frame_lock
    while not quit_program:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                current_frame = frame.copy()
        else:
            time.sleep(1)


def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(param['frame'], (x, y), 5, (0, 255, 0), -1)
        if len(points) >= 2:
            cv2.line(param['frame'], points[-2], points[-1], (0, 255, 0), 2)
        if len(points) == 4:
            cv2.line(param['frame'], points[3], points[0], (0, 255, 0), 2)
        cv2.imshow(param['window_name'], param['frame'])


def define_empty_area():
    global empty_area_transform, empty_area_size
    if len(points) == 4:
        # Calculate width and height
        width1 = np.linalg.norm(np.array(points[1]) - np.array(points[0]))
        width2 = np.linalg.norm(np.array(points[2]) - np.array(points[3]))
        height1 = np.linalg.norm(np.array(points[3]) - np.array(points[0]))
        height2 = np.linalg.norm(np.array(points[2]) - np.array(points[1]))
        empty_width = int((width1 + width2) / 2)
        empty_height = int((height1 + height2) / 2)
        empty_area_size = (empty_width, empty_height)
        src_pts = np.array(points, dtype=np.float32)
        dst_pts = np.array([
            [0, 0],
            [empty_width, 0],
            [empty_width, empty_height],
            [0, empty_height]
        ], dtype=np.float32)
        empty_area_transform = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return True
    return False


def detect_paper_by_color(image, reference):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_image, gray_reference)
    _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 3000:
        return None
    result_image = image.copy()
    cv2.drawContours(result_image, [largest_contour], 0, (0, 0, 255), 2)
    return result_image


# Main program
cap = cv2.VideoCapture(0)
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

print("Click 4 points to define the empty area (top-left, top-right, bottom-right, bottom-left).")
print("Press 'r' to reset points.")
print("Press 'd' to confirm the area and save a reference image.")
print("Press 'o' to detect paper in the area and save the result.")
print("Press 'q' to quit.")

empty_area_reference = None

while current_frame is None and not quit_program:
    print("Waiting for first frame...")
    time.sleep(0.5)

while not quit_program:
    with frame_lock:
        if current_frame is None:
            continue
        frame = current_frame.copy()
    display_frame = frame.copy()
    callback_params['frame'] = display_frame

    # Draw points and lines
    for i, point in enumerate(points):
        cv2.circle(display_frame, point, 5, (0, 255, 0), -1)
        if i > 0:
            cv2.line(display_frame, points[i-1], point, (0, 255, 0), 2)
    if len(points) == 4:
        cv2.line(display_frame, points[3], points[0], (0, 255, 0), 2)

    cv2.imshow(window_name, display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        quit_program = True
        break
    elif key == ord('r'):
        points = []
        empty_area_transform = None
        empty_area_reference = None
    elif key == ord('d') and len(points) == 4:
        if define_empty_area():
            warped = cv2.warpPerspective(
                frame, empty_area_transform, empty_area_size)
            empty_area_reference = warped.copy()
            cv2.imshow("Empty Area Reference", empty_area_reference)
            cv2.imwrite("empty_area_reference.jpg", empty_area_reference)
            print("Empty area reference saved as 'empty_area_reference.jpg'")
        else:
            print("Failed to define empty area. Try again.")
    elif key == ord('o') and empty_area_reference is not None:
        with frame_lock:
            frame_to_process = current_frame.copy()
        warped = cv2.warpPerspective(
            frame_to_process, empty_area_transform, empty_area_size)
        result_image = detect_paper_by_color(warped, empty_area_reference)
        if result_image is not None:
            cv2.imshow("Paper Detection", result_image)
            cv2.imwrite("paper_detection.jpg", result_image)
            print("Paper detection result saved as 'paper_detection.jpg'")
        else:
            print("No paper detected in the empty area.")

cap.release()
cv2.destroyAllWindows()
print("Program terminated.")
