import os
import random
import shutil


def sample_yolo(img_dir, txt_dir, output_dir, num_samples=2000):
    # Get all image files in the directory
    img_files = [
        f for f in os.listdir(img_dir) if f.endswith(".jpg") or f.endswith(".png")
    ]
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    # Randomly select num_samples images
    selected_files = random.sample(img_files, num_samples)

    # Copy the selected images and their corresponding txt files to the output directory
    for file in selected_files:
        img_path = os.path.join(img_dir, file)
        txt_path = os.path.join(
            txt_dir, file.replace(".jpg", ".txt").replace(".png", ".txt")
        )
        output_img_path = os.path.join(output_dir, "images", file)
        output_txt_path = os.path.join(
            output_dir, "labels", file.replace(".jpg", ".txt").replace(".png", ".txt")
        )
        shutil.copy(img_path, output_img_path)
        shutil.copy(txt_path, output_txt_path)


if __name__ == "__main__":
    sample_yolo(
        img_dir="Z:/Shanghai/广告牌数据集/all-imgs",
        txt_dir="Z:/Shanghai/广告牌数据集/all-labs-yolo",
        output_dir="F:/上海项目/accTestData/data/店招数据集",
        num_samples=2000,
    )
