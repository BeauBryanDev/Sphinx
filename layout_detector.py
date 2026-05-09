# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np


# TODO:  This script will turn out to be app/utils/layout_detector.py
# When backend is built in FastAPI , it will come after step 0 . 
# It will be called by the backend to detect the layout of the image.
# Ancient Egyptian hieroglyphs are written in rows or columns.
# This script will detect the layout of the image.


def detect_layout(img):
    """
    Detect layout of the image
    """
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # adaptive thresholding
    binary  = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )
    
 
    h, w    = binary.shape
   
    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (w // 3, 1)
    )
    h_lines  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    h_score  = np.sum(h_lines) / 255  
    
   

    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, h // 3)
    )
    v_lines  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    v_score  = np.sum(v_lines) / 255
    

    MIN_SCORE = 500  
    
    if h_score > MIN_SCORE and h_score > v_score * 1.5:

        return "rows"       

    elif v_score > MIN_SCORE and v_score > h_score * 1.5:

        return "columns"    

    else:

        return "rows"      