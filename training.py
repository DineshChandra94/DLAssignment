import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
from Custom import DetectionModelTrainer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory=os.path.join(BASE_DIR, "data"))
trainer.setTrainConfig(class_json_path = os.path.join(BASE_DIR,'number_to_type.json'), batch_size=8, num_experiments=30, train_from_pretrained_model=os.path.join(BASE_DIR,"pretrained-yolov3.h5"))
trainer.trainModel()
    