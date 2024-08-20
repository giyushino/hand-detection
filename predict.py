import os
from ultralytics import YOLO
import cv2
import pyautogui

def detect_directions():
    detected = []
    up = 0
    down = 0
    forward = 0
    backward = 0
    left = 0
    right = 0

    
    cap = cv2.VideoCapture(0)  

    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None  # Exit if webcam cannot be opened

    
    ret, frame = cap.read()
    H, W, _ = frame.shape
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    model_path = os.path.join('.', 'runs', 'detect', 'train5', 'weights', 'last.pt')
    model = YOLO(model_path)

    threshold = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        results = model(frame)[0]

       
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
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


        
        cv2.imshow('Webcam', frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if up > 100 or down > 100 or left > 100 or right > 100 or forward > 100 or backward > 100:
            pyautogui.press("q")
            break


    
    cap.release()
    cv2.destroyAllWindows()

    print("Up: {0}, Down: {1}, Right: {2}, Left: {3}, Forward: {4}, Backward: {5}".format(up, down, left, right, forward, backward))
    return up, down, left, right, forward, backward
