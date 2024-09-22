from ultralytics import YOLO
import torch

def main():
    model = YOLO(r"C:\Users\allan\PycharmProjects\yolov8\runs\detect\train8\weights\best.pt")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = model.train(
        data=r"C:\Users\allan\PycharmProjects\yolov8\config.yaml",
        epochs=300,
        patience=100,
        project=r"C:\Users\allan\PycharmProjects\yolov8\runs\detect",
        device=device,
        cache=False  # Force cache regeneration
    )

if __name__ == '__main__':
    main()
