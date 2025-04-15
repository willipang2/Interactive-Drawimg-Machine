import cv2
import numpy as np
import json
import serial
import time
import os


def find_latest_image():
    """Find the most recently modified image file"""
    files = [f for f in os.listdir('.') if f.endswith(
        '.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
    if not files:
        print("No suitable image files found.")
        return None

    latest_file = max(files, key=os.path.getmtime)
    print(f"Found image: {latest_file}")
    return latest_file


class DrawingProcessor:
    def __init__(self, canvas_info_path, margin_mm=10, feed_rate=1000):
        self.margin_mm = margin_mm
        self.feed_rate = feed_rate
        self.A4_WIDTH = 210
        self.A4_HEIGHT = 297

        # Load canvas information from JSON
        with open(canvas_info_path, 'r') as f:
            self.canvas_info = json.load(f)

        # Extract canvas dimensions and position
        self.canvas_width = self.canvas_info["width_mm"]
        self.canvas_height = self.canvas_info["height_mm"]
        self.canvas_position_x = self.canvas_info["position_x_mm"]
        self.canvas_position_y = self.canvas_info["position_y_mm"]

    def create_smooth_edges(self, input_path, output_path):
        print(f"Processing image: {input_path}")
        # Read the input image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError("Could not load input image")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

        # Apply Canny edge detection with automatic thresholding
        # Compute median of image to determine thresholds
        v = np.median(blurred)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(blurred, lower, upper)

        # Dilate edges slightly to connect nearby edges
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Save the processed edge image
        cv2.imwrite(output_path, dilated)
        print(f"Edge image saved to: {output_path}")
        return dilated

    def generate_gcode(self, edge_image_path, output_path):
        print(f"Generating G-code from: {edge_image_path}")
        # Read the edge image
        edges = cv2.imread(edge_image_path, cv2.IMREAD_GRAYSCALE)
        if edges is None:
            raise ValueError("Could not load edge image")

        # Find contours in the edge image
        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate usable canvas dimensions (accounting for margin)
        usable_width = self.canvas_width - (2 * self.margin_mm)
        usable_height = self.canvas_height - (2 * self.margin_mm)

        # Get image dimensions
        img_height, img_width = edges.shape

        # Calculate scaling factor to fit image to canvas
        scale_x = usable_width / img_width
        scale_y = usable_height / img_height
        scale = min(scale_x, scale_y)

        # Calculate the offset to center the image on the canvas
        offset_x = self.canvas_position_x + self.margin_mm + \
            (usable_width - img_width * scale) / 2
        offset_y = self.canvas_position_y + self.margin_mm + \
            (usable_height - img_height * scale) / 2

        # Generate G-code header
        gcode = [
            "G21 ; Set units to millimeters",
            "G90 ; Set absolute positioning",
            "G0 Z0 ; Pen up",
            "G0 X0 Y0 ; Move to home position"
        ]

        # Process each contour
        print(f"Processing {len(contours)} contours...")
        for contour in contours:
            # Skip very small contours (likely noise)
            if cv2.contourArea(contour) < 10:
                continue

            # Simplify contour to reduce points (smoother lines)
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Further smooth the contour using a spline interpolation
            if len(approx) > 2:
                # Convert to scaled coordinates
                points = []
                for point in approx:
                    x, y = point[0]
                    # Scale and offset to fit canvas
                    x_mm = x * scale + offset_x
                    y_mm = (img_height - y) * scale + \
                        offset_y  # Invert Y coordinate
                    points.append((x_mm, y_mm))

                # Start drawing this contour
                x, y = points[0]
                gcode.extend([
                    f"G0 X{x:.3f} Y{y:.3f}",
                    "G0 Z13 ; Pen down"
                ])

                # Draw the contour with smooth lines
                for x, y in points[1:]:
                    gcode.append(f"G1 X{x:.3f} Y{y:.3f} F{self.feed_rate}")

                # Lift pen after drawing this contour
                gcode.append("G0 Z0 ; Pen up")

        # End G-code
        gcode.extend([
            "G0 Z0 ; Pen up",
            "G0 X0 Y0 ; Return home"
        ])

        # Write G-code to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(gcode))

        print(f"G-code saved to: {output_path}")


def send_gcode_to_machine(gcode_file, port, baud_rate):
    """
    Reads a G-code file and sends commands to the machine

    Args:
        gcode_file (str): Path to the G-code file
        port (str): Serial port for the machine
        baud_rate (int): Baud rate for serial communication
    """
    try:
        # Open serial connection
        print(f"Opening connection to {port} at {baud_rate} baud...")
        ser = serial.Serial(port, baud_rate, timeout=5)
        time.sleep(2)  # Wait for connection to establish

        # Check if connection is open
        if not ser.is_open:
            print("Failed to open serial connection.")
            return False

        print("Connection established.")

        # Wake up the controller
        ser.write("\r\n\r\n".encode())
        time.sleep(2)

        # Clear any startup messages in buffer
        ser.flushInput()

        # Read G-code file
        with open(gcode_file, 'r') as f:
            gcode_lines = f.readlines()

        total_lines = len(gcode_lines)
        print(f"Sending {total_lines} lines of G-code...")

        # Send each line of G-code
        for i, line in enumerate(gcode_lines):
            # Strip comments and whitespace
            l = line.split(';')[0].strip()
            if not l:
                continue  # Skip empty lines

            # Send the command
            ser.write((l + '\n').encode())

            # Wait for response
            response = ser.readline().decode().strip()

            # Print progress
            if i % 10 == 0 or i == total_lines - 1:
                print(
                    f"Progress: {i+1}/{total_lines} ({(i+1)/total_lines*100:.1f}%)")

            # Safety check - look for error messages
            if 'error' in response.lower():
                print(f"Error received: {response}")
                choice = input("Continue sending commands? (y/n): ")
                if choice.lower() != 'y':
                    print("Aborting.")
                    break

        # Close connection
        ser.close()
        print("G-code sending completed successfully.")
        return True

    except serial.SerialException as e:
        print(f"Serial connection error: {e}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def main():
    # Set default values
    canvas_info_path = "small_canvas_info.json"
    edge_output = "edges.png"
    output_gcode = "output.gcode"
    PORT = "/dev/cu.usbserial-2130"  # Your specified port
    BAUD_RATE = 115200

    # Check if small_canvas_info.json exists
    if not os.path.exists(canvas_info_path):
        print("Canvas info file not found. Creating example file...")
        example_canvas = {
            "width_mm": 100,
            "height_mm": 150,
            "position_x_mm": 55,
            "position_y_mm": 75,
            "center_x_mm": 105,
            "center_y_mm": 150
        }
        with open(canvas_info_path, 'w') as f:
            json.dump(example_canvas, f, indent=4)
        print(f"Created example canvas info file: {canvas_info_path}")
        print("Please edit this file with your actual canvas dimensions and position.")
        return

    # Find the latest image file
    input_path = find_latest_image()
    if not input_path:
        print("No image file found. Please add an image file to the directory.")
        return

    # Process the image
    processor = DrawingProcessor(canvas_info_path)
    processor.create_smooth_edges(input_path, edge_output)
    processor.generate_gcode(edge_output, output_gcode)

    # Send to machine automatically
    print("\nSending commands to machine...")
    send_gcode_to_machine(output_gcode, PORT, BAUD_RATE)
    print("✓ Drawing complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
