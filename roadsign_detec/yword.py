import ultralytics
import ultralytics.data.annotator as annotator


model = ultralytics.YOLOWorld("yolov8x-worldv2.pt")
print(model.names)
results = model.predict(
    source="F:/percar_yolo/images/20241209114545_7h02043.JPG", stream=False
)
print(results[0].names)
