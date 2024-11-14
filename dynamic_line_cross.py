import cv2
from collections import defaultdict
import supervision as sv
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')

def cross_line(input_video, output_video, start_x, start_y, end_x, end_y):
    # Set up video capture
    cap = cv2.VideoCapture(input_video)

    # Define the line coordinates
    START = sv.Point(start_x, start_y)
    END = sv.Point(end_x, end_y)

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Create dictionaries to keep track of objects that have crossed the line
    crossed_objects_lr = {}  # Left to Right
    crossed_objects_rl = {}  # Right to Left

    # Get the original video resolution
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Open a video sink for the output video
    video_info = sv.VideoInfo.from_video_path(input_video)
    with sv.VideoSink(output_video, video_info) as sink:

        while cap.isOpened():
            success, frame = cap.read()

            if success:
                # frame = cv2.resize(frame, (1020, 500))

                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.track(frame, classes=[2,6,7], persist=True, save=True, tracker="bytetrack.yaml")

                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id

                if track_ids is not None:
                    track_ids = track_ids.cpu().numpy().astype(int)
                else:
                    track_ids = []

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Plot the tracks and count objects crossing the line
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)

                    # print("Track:", track_id, ": ",track)
                    # Check if the object crosses the line
                    if START.x < x < END.x and abs(y - START.y) < abs(END.y - START.y):
                        if len(track) > 1:
                        # Use the first recorded position in the track as the previous position
                            prev_x, prev_y = track[0]
                            print(f"Track ID: {track_id}, Prev_x: {prev_x}, Current_x: {x}, Start_x: {START.x}, End_x: {END.x}")

                            if (prev_x < START.x and x > START.x) or (prev_x < END.x and x > END.x):
                                if track_id not in crossed_objects_lr:
                                    print(f"Object {track_id} crossed L to R")
                                    crossed_objects_lr[track_id] = True

                            if (prev_x > START.x and x < START.x) or (prev_x > END.x and x < END.x):
                                if track_id not in crossed_objects_rl:
                                    print(f"Object {track_id} crossed R to L")
                                    crossed_objects_rl[track_id] = True

                        # Annotate the object as it crosses the line
                        cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)),
                                      (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

                # Draw the line on the frame
                cv2.line(annotated_frame, (START.x, START.y), (END.x, END.y), (0, 0, 255), 2)

                # Write the count of objects on each frame
                count_text_lr = f"Objects crossed (L to R): {len(crossed_objects_lr)}"
                count_text_rl = f"Objects crossed (R to L): {len(crossed_objects_rl)}"
                cv2.putText(annotated_frame, count_text_lr, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(annotated_frame, count_text_rl, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

                # Resize the annotated frame back to the original size before writing
                output_frame = cv2.resize(annotated_frame, (original_width, original_height))

                # Write the frame with annotations to the output video
                sink.write_frame(output_frame)

                cv2.imshow("frame", annotated_frame)

                if cv2.waitKey(10) & 0xFF == 27:
                    # print(track_history)
                    break
            else:
                break

    # Release the video capture
    cap.release()

# Example usage:
# cross_line('input_video.mp4', 'output_video.mp4', start_x=100, start_y=200, end_x=500, end_y=200)
