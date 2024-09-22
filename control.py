import os
from ultralytics import YOLO
import cv2
import torch
import MultiWii
from predict import detect_directions


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


