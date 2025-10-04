import os
import shutil
import matplotlib.pyplot as plt
import cv2


def draw_one_box(img, x1, y1, x2, y2, x3, y3, x4, y4):
    X, Y = img.shape[1], img.shape[0]

    x1 = int(x1 * X)
    y1 = int(y1 * Y)
    x2 = int(x2 * X)
    y2 = int(y2 * Y)
    x3 = int(x3 * X)
    y3 = int(y3 * Y)
    x4 = int(x4 * X)
    y4 = int(y4 * Y)

    img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    img = cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 2)
    img = cv2.line(img, (x3, y3), (x4, y4), (0, 0, 255), 2)
    img = cv2.line(img, (x4, y4), (x1, y1), (0, 0, 255), 2)

    return img


def read_img_txt(img_path, txt_path):
    img = cv2.imread(img_path)
    with open(txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == "":
                continue
            line = line.split(" ")
            x1, y1, x2, y2, x3, y3, x4, y4 = tuple(map(float, line[1:]))
            # print(x1, y1, x2, y2, x3, y3, x4, y4)
            img = draw_one_box(img, x1, y1, x2, y2, x3, y3, x4, y4)
    return img


def move_rename_files(src_dir, dst_dir, prefix, file_extension):
    for file_name in os.listdir(src_dir):
        if file_name.endswith(file_extension):
            src_path = os.path.join(src_dir, file_name)
            dst_path = os.path.join(dst_dir, f"{prefix}_{file_name}")
            shutil.move(src_path, dst_path)


def split_train_val(root):
    image_path = os.path.join(root, "images")
    label_path = os.path.join(root, "labels")
    train_image_path = os.path.join(root, "train/images")
    train_label_path = os.path.join(root, "train/labels")
    val_image_path = os.path.join(root, "val/images")
    val_label_path = os.path.join(root, "val/labels")

    os.makedirs(train_image_path, exist_ok=True)
    os.makedirs(train_label_path, exist_ok=True)
    os.makedirs(val_image_path, exist_ok=True)
    os.makedirs(val_label_path, exist_ok=True)

    files = [
        f for f in os.listdir(image_path) if f.endswith(".tif") or f.endswith(".jpg")
    ]
    import random

    random.shuffle(files)
    train_files = files[: int(len(files) * 0.8)]
    val_files = files[int(len(files) * 0.8) :]

    for file in train_files:
        src_image_path = os.path.join(image_path, file)
        src_label_path = os.path.join(label_path, file.split(".")[0] + ".txt")
        dst_image_path = os.path.join(train_image_path, file)
        dst_label_path = os.path.join(train_label_path, file.split(".")[0] + ".txt")
        shutil.copy(src_image_path, dst_image_path)
        shutil.copy(src_label_path, dst_label_path)

    for file in val_files:
        src_image_path = os.path.join(image_path, file)
        src_label_path = os.path.join(label_path, file.split(".")[0] + ".txt")
        dst_image_path = os.path.join(val_image_path, file)
        dst_label_path = os.path.join(val_label_path, file.split(".")[0] + ".txt")
        shutil.copy(src_image_path, dst_image_path)
        shutil.copy(src_label_path, dst_label_path)


def run_move_merged_parking():
    os.makedirs("F:/停车场-小区口标注/merged/images", exist_ok=True)
    os.makedirs("F:/停车场-小区口标注/merged/labels", exist_ok=True)
    move_rename_files(
        "F:/停车场-小区口标注/tiles_Production_domdsm_1_ortho_merge",
        "F:/停车场-小区口标注/merged/images",
        "1",
        ".tif",
    )
    move_rename_files(
        "F:/停车场-小区口标注/tiles_Production_domdsm_2_ortho_merge",
        "F:/停车场-小区口标注/merged/images",
        "2",
        ".tif",
    )
    move_rename_files(
        "F:/停车场-小区口标注/tiles_Production_domdsm_3_ortho_merge",
        "F:/停车场-小区口标注/merged/images",
        "3",
        ".tif",
    )
    move_rename_files(
        "F:/停车场-小区口标注/tiles_Production_domdsm_4_ortho_merge",
        "F:/停车场-小区口标注/merged/images",
        "4",
        ".tif",
    )
    move_rename_files(
        "F:/停车场-小区口标注/tiles_Production_dsm_ortho_merge_32651",
        "F:/停车场-小区口标注/merged/images",
        "5",
        ".tif",
    )

    move_rename_files(
        "F:/停车场-小区口标注/tiles_Production_domdsm_1_ortho_merge",
        "F:/停车场-小区口标注/merged/labels",
        "1",
        ".txt",
    )
    move_rename_files(
        "F:/停车场-小区口标注/tiles_Production_domdsm_2_ortho_merge",
        "F:/停车场-小区口标注/merged/labels",
        "2",
        ".txt",
    )
    move_rename_files(
        "F:/停车场-小区口标注/tiles_Production_domdsm_3_ortho_merge",
        "F:/停车场-小区口标注/merged/labels",
        "3",
        ".txt",
    )
    move_rename_files(
        "F:/停车场-小区口标注/tiles_Production_domdsm_4_ortho_merge",
        "F:/停车场-小区口标注/merged/labels",
        "4",
        ".txt",
    )
    move_rename_files(
        "F:/停车场-小区口标注/tiles_Production_dsm_ortho_merge_32651",
        "F:/停车场-小区口标注/merged/labels",
        "5",
        ".txt",
    )


def run_move_merged_door():
    os.makedirs("F:/停车场-小区口标注/mergeddoor/images", exist_ok=True)
    os.makedirs("F:/停车场-小区口标注/mergeddoor/labels", exist_ok=True)
    move_rename_files(
        "F:/停车场-小区口标注/tiles_Production_domdsm_1_ortho_merge",
        "F:/停车场-小区口标注/mergeddoor/images",
        "1",
        ".tif",
    )
    move_rename_files(
        "F:/停车场-小区口标注/tiles_Production_domdsm_2_ortho_merge",
        "F:/停车场-小区口标注/mergeddoor/images",
        "2",
        ".tif",
    )
    move_rename_files(
        "F:/停车场-小区口标注/tiles_Production_domdsm_3_ortho_merge",
        "F:/停车场-小区口标注/mergeddoor/images",
        "3",
        ".tif",
    )
    move_rename_files(
        "F:/停车场-小区口标注/tiles_Production_domdsm_4_ortho_merge",
        "F:/停车场-小区口标注/mergeddoor/images",
        "4",
        ".tif",
    )
    move_rename_files(
        "F:/停车场-小区口标注/tiles_Production_dsm_ortho_merge_32651",
        "F:/停车场-小区口标注/mergeddoor/images",
        "5",
        ".tif",
    )

    move_rename_files(
        "F:/停车场-小区口标注/tiles_Production_domdsm_1_ortho_merge",
        "F:/停车场-小区口标注/mergeddoor/labels",
        "1",
        ".txt",
    )
    move_rename_files(
        "F:/停车场-小区口标注/tiles_Production_domdsm_2_ortho_merge",
        "F:/停车场-小区口标注/mergeddoor/labels",
        "2",
        ".txt",
    )
    move_rename_files(
        "F:/停车场-小区口标注/tiles_Production_domdsm_3_ortho_merge",
        "F:/停车场-小区口标注/mergeddoor/labels",
        "3",
        ".txt",
    )
    move_rename_files(
        "F:/停车场-小区口标注/tiles_Production_domdsm_4_ortho_merge",
        "F:/停车场-小区口标注/mergeddoor/labels",
        "4",
        ".txt",
    )
    move_rename_files(
        "F:/停车场-小区口标注/tiles_Production_dsm_ortho_merge_32651",
        "F:/停车场-小区口标注/mergeddoor/labels",
        "5",
        ".txt",
    )


if __name__ == "__main__":
    # run_move_merged_door()
    split_train_val("F:/label_exit_yolo")
