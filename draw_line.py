import cv2

def draw_line(video_path):
    # Open the video file4
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read the first frame
    ret, frame = cap.read()

    # If frame read is successful
    if ret:
       # frame = cv2.resize(frame, (1020, 500))

        # Display the frame
        cv2.imshow('Frame', frame)

        # Function to draw line
        def draw_line(event, x, y, flags, param):
            global start_point, end_point, drawing

            if event == cv2.EVENT_LBUTTONDOWN:
                start_point = (x, y)
                drawing = True

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    temp_frame = frame.copy()
                    cv2.line(temp_frame, start_point, (x, y), (0, 255, 0), 2)
                    cv2.imshow('Frame', temp_frame)

            elif event == cv2.EVENT_LBUTTONUP:
                end_point = (x, y)
                drawing = False
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
                cv2.imshow('Frame', frame)

        # Initialize global variables
        global start_point, end_point, drawing
        start_point, end_point = (0, 0), (0, 0)
        drawing = False

        # Set the mouse callback function to draw line
        cv2.setMouseCallback('Frame', draw_line)

        # Wait until user presses Enter key
        while True:
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(1)
            if key == 13:  # Enter key
                break

        # Release the video capture object
        cap.release()

        # Close all OpenCV windows
        cv2.destroyAllWindows()

        # Return the starting and ending coordinates of the line
        return (start_point[0], start_point[1], end_point[0], end_point[1])
    else:
        print("Error: Could not read frame.")


# coordinates = draw_line('sample/line-count.mp4')
# print(f"Line coordinates: {coordinates}")
