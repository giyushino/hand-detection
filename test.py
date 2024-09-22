import os
from ultralytics import YOLO
import cv2


def detect_directions():
    detected = []
    up = 0
    down = 0
    forward = 0
    backward = 0
    left = 0
    right = 0

    # Set up webcam capture
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Load your custom YOLO model
    model_path = os.path.join('.', 'runs', 'detect', 'train9', 'weights', 'best.pt')
    model = YOLO(model_path)
    model = YOLO(model_path)
    model.to(device)

    threshold = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO model inference on the frame
        results = model(frame)[0]

        # Process the results
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                current_direction = results.names[int(class_id)]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            print(results.names[int(class_id)])
            if results.names[int(class_id)] == "up":
                up += 1
            elif results.names[int(class_id)] == "down":
                down += 1
            elif results.names[int(class_id)] == "right":
                right += 1
            elif results.names[int(class_id)] == "left":
                left += 1
            elif results.names[int(class_id)] == "forward":
                forward += 1
            elif results.names[int(class_id)] == "backward":
                backward += 1


        # Display the frame with detections
        cv2.imshow('Webcam', frame)


        # Exit on pressing the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    print("Up: {0}, Down: {1}, Right: {2}, Left: {3}, Forward: {4}, Backward: {5}".format(up, down, left, right, forward, backward))
    return up, down, left, right, forward, backward
