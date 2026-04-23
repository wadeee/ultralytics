from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolo26m.pt")  # load a pretrained model (recommended for training)
    # model.to("cuda")

    # Train the model
    results = model.train(
        data="datasets/african-wildlife.yaml",
        epochs=100,
        imgsz=640,
        device=0,
        batch=12,        # 8GB 显存 + n 系列模型，16 没问题
        workers=2,       # 数据加载线程数，Windows 建议 4
        cache='disk',      # 缓存图片到内存，加快训练速度
        amp=True,        # 混合精度训练，省显存又快
    )
