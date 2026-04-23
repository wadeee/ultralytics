from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)
    model.to("cpu")

    # Train the model
    results = model.train(
        data="custom/african-wildlife.yaml",
        epochs=100,
        imgsz=640,
        device="cpu"
    )
