import PySide6.QtWidgets as QtWidgets
import os


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("文件重命名工具")
        self.setGeometry(100, 100, 400, 300)

        self._layout = QtWidgets.QVBoxLayout()

        floder_layout = QtWidgets.QHBoxLayout()

        self.label = QtWidgets.QLabel("请选择文件夹")
        floder_layout.addWidget(self.label)

        self.folder_path = QtWidgets.QLineEdit()
        floder_layout.addWidget(self.folder_path)

        self.browse_button = QtWidgets.QPushButton("浏览")
        self.browse_button.clicked.connect(self.browse_folder)
        floder_layout.addWidget(self.browse_button)

        self._layout.addLayout(floder_layout)

        csv_layout = QtWidgets.QHBoxLayout()

        self.csv_label = QtWidgets.QLabel("请选择CSV文件")
        csv_layout.addWidget(self.csv_label)

        self.csv_path = QtWidgets.QLineEdit()
        csv_layout.addWidget(self.csv_path)

        self.csv_browse_button = QtWidgets.QPushButton("浏览")
        self.csv_browse_button.clicked.connect(self.browse_csv)
        csv_layout.addWidget(self.csv_browse_button)

        self._layout.addLayout(csv_layout)

        self.rename_button = QtWidgets.QPushButton("重命名")
        self.rename_button.clicked.connect(self.rename_files)
        self._layout.addWidget(self.rename_button)

        self.setLayout(self._layout)

    def browse_folder(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择文件夹")
        self.folder_path.setText(folder_path)

    def browse_csv(self):
        csv_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择CSV文件")
        self.csv_path.setText(csv_path)

    def rename_files(self):
        folder_path = self.folder_path.text()
        csv_path = self.csv_path.text()

        if not folder_path or not csv_path:
            QtWidgets.QMessageBox.warning(self, "警告", "请选择文件夹和CSV文件")
            return

        try:
            with open(csv_path, "r", encoding="utf8") as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                old_name, new_name = parts[1], parts[0]
                old_path = os.path.join(folder_path, old_name)
                new_path = os.path.join(folder_path, new_name)
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                    print(f"重命名文件: {old_path} -> {new_path}")
                else:
                    print(f"文件不存在: {old_path}")

            QtWidgets.QMessageBox.information(self, "完成", "文件重命名完成")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"文件重命名失败: {str(e)}")


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
