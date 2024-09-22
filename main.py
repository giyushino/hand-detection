from ultralytics import YOLO
import torch

def main():
    model = YOLO(r"path to best.pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = model.train(
        data=r"path to config.yaml",
        epochs=300,
        patience=100,
        project=r"path to \yolov8\runs\detect",
        device=device,
        cache=False  # Force cache regeneration
    )

if __name__ == '__main__':
    main()
