# pyrefly: ignore [missing-import]
from email.mime import image
# pyrefly: ignore [missing-import]
from sahi import AutoDetectionModel
# pyrefly: ignore [missing-import]
from sahi.predict import get_sliced_prediction

model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="best_v4.pt",
    confidence_threshold=0.25,
    device="cuda"
)

result = get_sliced_prediction(
    image,
    model,
    slice_height=640,      # tamaño del tile
    slice_width=640,
    overlap_height_ratio=0.25,   # 25% overlap entre tiles
    overlap_width_ratio=0.25
)

# TODO: step 1 , after image enhancement (CLAHE) and image resizing (640x640) 
# this script will be called by the backend to detect the hieroglyphs in the image.
# It will be app/utils/sahi.py
# when backend is built in FastAPI

