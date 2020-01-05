from Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("/content/DLAssignment/custom_model.h5")
detector.setJsonPath("/content/DLAssignment/data/json/detection_config.json")
detector.loadModel()

for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])