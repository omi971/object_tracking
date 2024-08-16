import cv2
import numpy as np
import math
from object_detection import ObjectDetection


cap = cv2.VideoCapture("C:\\Users\\Nazmul Haque Omi\\Desktop\\codes\\machine_vision\\object_tracking\\source_code\\los_angeles.mp4")

# Initialize Object Detection
od = ObjectDetection()

# Center point current frame
c_pt_curr_frame = []

# Center point prev frame
c_pt_prev_frame = []

# Tracking Object Dictionary
tracking_object = {}

# Define Track id
track_id = 0

# Frame Counter
count = 0
while True:
    tracking_object = {}
    ret, frame = cap.read()
    count += 1
    # r_frame = cv2.resize(frame, (640, 480))
    # if we stop getting frame than break
    if not ret:
        break
    
    # detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)
    c_pt_curr_frame = []
    print("\n"+ "--------------------------------------"+" \n")
    print(f"Length: {len(c_pt_curr_frame)}")
    
    for box in boxes:
        (x, y, w, h) = box

        # Getting the center of the bounding box``
        cx = int((x+x+w)/2)
        cy = int((y+y+h)/2)
        c_pt_curr_frame.append((cx, cy))
        print(f"Frame No. {count}, Box: {box}")
        # print(f"Center Point: {center_points}") 
        print(f"length: {len(c_pt_curr_frame)}")
        # cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # Only at the beginning we compare previous and current frame
    if count <= 2:
        for pt in c_pt_curr_frame:
            for pt2 in c_pt_curr_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 20:
                    tracking_object[track_id] = pt
                    track_id += 1
    else:
        tracking_objects_copy = tracking_object.copy()
        center_points_cur_frame_copy = c_pt_curr_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # Update IDs position
                if distance < 20:
                    tracking_object[object_id] = pt
                    object_exists = True
                    if pt in c_pt_curr_frame:
                        c_pt_curr_frame.remove(pt)
                    continue

            # Remove IDs lost
            if not object_exists:
                tracking_object.pop(object_id)

        # Add new IDs found
        for pt in c_pt_curr_frame:
            tracking_object[track_id] = pt
            track_id += 1

    for object_id, pt in tracking_object.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

    print("")
    print(f"Tracking Object: {tracking_object}")
    print(f"count: {len(tracking_object)}")

    print("")
    print(f"Curr Frame: {c_pt_curr_frame}")

    print("")
    print(f"Prev Frame: {c_pt_prev_frame}")

    cv2.imshow('Frame', frame) 
    
    # make a copy of the points
    c_pt_prev_frame = c_pt_curr_frame.copy()

    # print("ret: ", ret)
    # print("frame: ", frame)

    
    key = cv2.waitKey(0)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()