from ultralytics.models.yolo import YOLO

fold_metrics = []

for i in range(10):
    model = YOLO('yolov8l.pt')
    results = model.train(
        data=f'data_yaml/fold{i}.yaml',
        epochs=5,
        optimizer='SGD',
        lr0=0.001,
        momentum=0.937,
        weight_decay=0.0005
    )
    fold_metrics.append(results)