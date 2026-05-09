# pyrefly: ignore [missing-import]
import cv2

def apply_clahe(img):
    """
    Apply CLAHE to an image
    """
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(
        clipLimit=2.0,     
        tileGridSize=(8,8)  
    )
    l_clahe = clahe.apply(l)
    
    lab_clahe = cv2.merge([l_clahe, a, b])
    
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)