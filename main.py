import cv2
from collections import defaultdict
import supervision as sv
from ultralytics import YOLO
from draw_line import draw_line
# from angular_line_cross2 import cross_line
from dynamic_line_cross import cross_line

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')

start_x, start_y = 1124, 335
end_x, end_y = 1144, 900


if __name__ == "__main__":
    input_video = 'balcony.mp4'
    output_video = 'output-balcony1.mp4'
    start_x, start_y, end_x, end_y = draw_line(input_video)
    cross_line(input_video, output_video, start_x, start_y, end_x, end_y)