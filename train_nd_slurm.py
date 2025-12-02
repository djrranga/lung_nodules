import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    fold = args.fold

    model = YOLO("yolov8l.pt")

    results = model.train(
        data=f"data_yaml/fold{fold}.yaml",
        epochs=100,
        imgsz=1024,
        optimizer="SGD",
        lr0=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        batch=16,               # tune based on your GPU memory
        device=0                # this GPU assigned by SLURM
    )

    print(f"Finished fold {fold}")

if __name__ == "__main__":
    main()
