from flask import Flask, request
from flask_cors import CORS
import cv2
import depthai as dai
import numpy as np


# Function to detect the color of a region of interest (ROI)
def detect_color(roi):
    # verde albe negru si rosu
    # Convert ROI to HSV color space
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the color (in HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])

    sensitivity = 15
    lower_white = np.array([0, 0, 255 - sensitivity])
    upper_white = np.array([255, sensitivity, 255])

    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Check which color is predominant in the ROI
    colors = ""
    if cv2.countNonZero(red_mask) > 0:
        colors = 'red'
        # colors.append('red')
    if cv2.countNonZero(green_mask) > 0:
        colors = 'green'
        # colors.append('green')
    if cv2.countNonZero(black_mask) > 0:
        colors = 'black'
        # colors.append('black')
    if cv2.countNonZero(white_mask) > 0:
        colors = 'white'
        # colors.append('white')

    return colors


app = Flask(__name__)
CORS(app)


@app.route('/test', methods=['POST'], endpoint="http://localhost:3000/")
def receive_post():
    if request.method == 'POST':

        sigla = request.json['sigla']

        # Create pipeline
        pipeline = dai.Pipeline()

        # Create DepthAI node for color camera
        colorCam = pipeline.createColorCamera()
        colorCam.setPreviewSize(1000, 1000)  # Set preview size (width, height)

        # Define output stream
        xout = pipeline.createXLinkOut()
        xout.setStreamName("video")

        # Link nodes
        colorCam.preview.link(xout.input)

        # Start the pipeline
        with dai.Device(pipeline) as device:
            # Output stream
            videoQueue = device.getOutputQueue(name="video", maxSize=1, blocking=False)

            # Define rectangles coordinates (x1, y1, x2, y2)
            rectangles = [
                (50, 50, 150, 150),  # Rectangle 1
                (200, 50, 300, 150),  # Rectangle 2
                (350, 50, 450, 150),  # Rectangle 2
                (500, 50, 600, 150),  # Rectangle 2
            ]

            while True:
                inFrame = videoQueue.get()  # Get the next frame

                # Convert to BGR format
                frame = inFrame.getCvFrame()
                index = 0
                # Draw rectangles on the frame
                for (x1, y1, x2, y2) in rectangles:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # Extract ROI and detect color
                    roi = frame[y1:y2, x1:x2]
                    colors = detect_color(roi)

                    # print(colors)
                    # Draw text indicating the detected color(s)
                    text = ', '.join(colors) if colors else 'None'
                    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    index += 1

                    sigle = {
                        'rsp':'red',
                        'fiir':'black',
                        'spd':'white',
                        'fiirbots':'green'
                    }

                    if colors == sigle[sigla]:
                        print(index)
                        # return colors

                # Display the frame
                cv2.imshow("DepthAI Camera", frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) == ord('q'):
                    break

            cv2.destroyAllWindows()

        return "POST request received successfully"
    else:
        return "Only POST requests are allowed"


if __name__ == '__main__':
    app.run(debug=True)
