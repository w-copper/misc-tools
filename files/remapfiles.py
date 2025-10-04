import os
import pandas as pd
import shutil
import zipfile
import py7zr


def read_excel_file(excel_path, sheet_name=0):
    # 读取Excel文件
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    return df


def create_student_folders(df, root_path):
    # 根据学生学号和姓名创建文件夹
    for _, row in df.iterrows():
        student_id = row["学号"]
        student_name = row["姓名"]
        folder_name = f"{student_id}-{student_name}"
        folder_path = os.path.join(root_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


def find_and_organize_files(student_info, source_dir, target_root_dir):
    # 在指定目录下查找文件并归类
    for file_name in os.listdir(source_dir):
        for student_id, student_name in student_info:
            if (str(student_id) in file_name) or (student_name in file_name):
                target_dir = os.path.join(
                    target_root_dir, f"{student_id}-{student_name}"
                )
                file_path = os.path.join(source_dir, file_name)
                # 判断文件类型并做相应处理
                if file_name.endswith(".zip"):
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        zip_ref.extractall(target_dir)
                elif file_name.endswith(".7z"):
                    with py7zr.SevenZipFile(file_path, mode="r") as z:
                        z.extractall(target_dir)
                else:
                    # 非压缩文件，直接移动
                    shutil.move(file_path, target_dir)
                break


if __name__ == "__main__":
    excel_path = r"D:\CVEO\2022-02-网络程序设计助教\分数-2024.xlsx"  # Excel文件路径
    sheet_name = 0  # or the name of the sheet
    source_dir = (
        r"D:\CVEO\2022-02-网络程序设计助教\2024大作业"  # 包含学号或姓名文件的源目录
    )
    target_root_dir = (
        r"D:\CVEO\2022-02-网络程序设计助教\2024大作业整理"  # 学号-姓名文件夹的根目录
    )

    df = read_excel_file(excel_path, sheet_name)
    create_student_folders(df, target_root_dir)

    student_info = df[["学号", "姓名"]].values.tolist()
    find_and_organize_files(student_info, source_dir, target_root_dir)
