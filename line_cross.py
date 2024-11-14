import cv2
from collections import defaultdict
import supervision as sv
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

def cross_line(input_video, output_video, start_x, start_y, end_x, end_y):
    # Set up video capture
    cap = cv2.VideoCapture(input_video)

    # Define the line coordinates
    START = sv.Point(start_x, start_y)
    END = sv.Point(end_x, end_y)

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Create a dictionary to keep track of objects that have crossed the line
    crossed_objects = {}

    # Open a video sink for the output video
    video_info = sv.VideoInfo.from_video_path(input_video)#input_video# )
    with sv.VideoSink(output_video, video_info) as sink:

        while cap.isOpened():
            success, frame = cap.read()

           # frame = cv2.resize(frame, (1020, 500))

            if success:
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

                # detections = sv.Detections.from_yolov8(results[0])

                # Plot the tracks and count objects crossing the line
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)

                    # Check if the object crosses the line
                    # if START.y < y < END.y and abs(x - START.x) < 5:  # Assuming objects cross horizontally
                    if START.y < y < END.y and abs(x - START.x) < abs(END.x - START.x):
                        if track_id not in crossed_objects:
                            crossed_objects[track_id] = True

                        # Annotate the object as it crosses the line
                        cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)),
                                      (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

                # Draw the line on the frame
                cv2.line(annotated_frame, (START.x, START.y), (END.x, END.y), (0, 0, 255), 2)

                # Write the count of objects on each frame
                count_text = f"Objects crossed: {len(crossed_objects)}"
                cv2.putText(annotated_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Write the frame with annotations to the output video
                sink.write_frame(annotated_frame)

                cv2.imshow("frame", annotated_frame)

                if cv2.waitKey(10)&0xFF==27:
                    break
            else:
                break

    # Release the video capture
    cap.release()