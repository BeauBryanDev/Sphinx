# pyrefly: ignore [missing-import]
from ultralytics import YOLO

model = YOLO("yolo11l.pt")

model.train(
    data    = "/content/SphinxEyes_v1/dataset.yaml",
    epochs  = 80,
    imgsz   = 224,
    batch   = 128,      
    cache   = True,
    lr0     = 0.01,
    lrf     = 0.01,
    warmup_epochs = 5,
    momentum      = 0.937,
    weight_decay  = 0.0005,
    patience      = 20,
    optimizer     = "SGD",
    augment       = True,
    degrees       = 3.0,    
    flipud        = 0.0,  
    fliplr        = 0.0,   
    mosaic        = 0.5,    #
    close_mosaic  = 20,   
    save_period   = 10,
    project       = "SphinxEyes",
    name          = "v1"
)


# VERSION V1 was already trained on 2026-05-06 
# It was YOLOV11 LARGE MODEL ,  TOTAL IMAGES : 25125 images 
# Total Clases 150 :  148 Gardiner Signs + Cartouches + Unknown
# They all make up 150 classes 
# First training successful
# We Reacht mAP : 0.98 , Presion : 0.97, Recall : 0.96 , mAP@95 : 0.94
# We are pending second traning v2  within  best.pt from v1
# Second Training will be multiple signs per images. 
# Dataset will come from Roboflow .print