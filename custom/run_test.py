from ultralytics import YOLO
import os
import glob

# 加载预训练的 YOLO26n 模型
model = YOLO("../runs/detect/train8/weights/best.pt")

# 设置图片路径 - 支持单个文件或文件夹
image_path = "../../datasets/african-wildlife/images/test"  # 可以是单个文件或文件夹路径

# 获取所有图片文件
if os.path.isfile(image_path):
    # 单个文件
    image_files = [image_path]
else:
    # 文件夹 - 支持常见图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_path, ext)))
    image_files.sort()  # 按名称排序

print(f"找到 {len(image_files)} 个图片文件")

# 批量处理图片
results = model(image_files)  # 一次性处理所有图片

# 输出 YOLO 格式的标记 txt 文件
def save_yolo_labels(results, output_dir="labels"):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    for i, result in enumerate(results):
        # 获取图像路径和文件名
        img_path = result.path
        img_name = os.path.basename(img_path)
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_name)
        
        # 获取图像尺寸
        height, width = result.orig_shape
        
        # 打开 txt 文件
        with open(txt_path, "w") as f:
            # 遍历每个检测到的目标
            for box in result.boxes:
                # 获取类别 ID
                class_id = int(box.cls[0])
                
                # 获取边界框坐标 (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0]
                
                # 计算 YOLO 格式的坐标 (归一化)
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                
                # 写入 txt 文件
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
        
        print(f"YOLO labels saved to: {txt_path}")

# 批量显示和保存结果
for i, result in enumerate(results):
    # 显示结果（可选，注释掉可以避免弹出多个窗口）
    # result.show()
    
    # 保存结果图片
    img_name = os.path.basename(result.path)
    output_path = f"labels/{img_name}"
    result.save(output_path)
    print(f"处理完成: {img_name}")

# 保存 YOLO 格式的标记文件
save_yolo_labels(results)