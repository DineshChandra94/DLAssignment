from Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("/content/DLAssignment/data/models/detection_model-ex-029--loss-0038.714.h5")
detector.setJsonPath("/content/DLAssignment/data/json/detection_config.json")
detector.loadModel()

from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath('/content/DLAssignment/data/models/detection_model-ex-029--loss-0038.714.h5')
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"), minimum_percentage_probability=30)


detections = detector.detectObjectsFromImage(input_image="/content/DLAssignment/data/validation/images/27.jpg", output_image_path="/content/DLAssignment/27.jpg")