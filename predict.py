import os
from ultralytics import YOLO
import cv2
import torch
import MultiWii
from predict import detect_directions

def detect_directions():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    H, W, _ = frame.shape

    model_path = os.path.join('.', 'runs', 'detect', 'train9', 'weights', 'best.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    model.to(device)
    print(model.device)
    threshold = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        current_direction = None

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                current_direction = results.names[int(class_id)]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                print(current_direction)

        cv2.imshow('Webcam', frame)

        yield current_direction

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def interpret(direction):
    board = MultiWii("COM3")
    board.connect()

    try:
        while True:
            command = [1500, 1500, 1500, 1000]
            if direction == "up":
                command = [1500, 1500, 1500, 1200]
            elif direction == "down":
                command = [1500, 1500, 1500, 800]
            elif direction == "right":
                command = [1700, 1500, 1500, 1000]
            elif direction == "left":
                command = [1300, 1500, 1500, 1000]
            elif direction == "backwards":
                command = [1500, 1700, 1500, 1000]
            elif direction == "forward":
                command = [1500, 1300, 1500, 1000]

            board.sendCMD(8, MultiWii.SET_RAW_RC, command)
    except KeyboardInterrupt:
        print("Disconnecting")
    finally:
        board.disconnect()


def fly():
    seen = []
    current_direction = None

    for direction in detect_directions():
        seen.append(direction)
        if len(seen) > 10:
            seen.pop(0)
        if len(seen) == 10 and seen.count(seen[0]) == 10:
            current_direction = seen[0]
            interpret(current_direction)
        else:
            current_direction = None


