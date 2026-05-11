# pyrefly: ignore [missing-import]
import cv2
import os
from pathlib import Path
from tqdm import tqdm

def resize_dataset(source_dirs, target_size=(224, 224)):
    """
    Scale all images in the source directories to target_size.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    for s_dir in source_dirs:
        print(f"\nProcessing directory: {s_dir}")
        path = Path(s_dir)
        
        # Get list of files
        files = [f for f in path.rglob('*') if f.suffix.lower() in valid_extensions]
        
        for img_path in tqdm(files, desc="Resizing"):
            try:
                # Read image
                img = cv2.imread(str(img_path))
                
                if img is None:
                    continue
                
                # Resize (we use INTER_AREA to reduce, it is better to avoid aliasing)
                resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                
                # Overwrite or save (here we overwrite to save space in the localhost)
                cv2.imwrite(str(img_path), resized_img)
                
            except Exception as e:
                print(f"Error in {img_path}: {e}")


if __name__ == "__main__":
    
    directories = ['./SphinxEyes_Final', './SphinxEyes_Others_Class200']
    
    resize_dataset(directories)
    print("\nDone ,all images turned out to 224x224.")


# TODO:  This script will turn out to be app/utils/pre_processing.py
# When backend is built in FastAPI , it will come after step 0 . CLAHE . 
# It will be called by the backend to resize the images.
# Ancient Egyptian hieroglyphs are written in rows or columns.
# This script will resize the images.
# We need all image resize to 640x640. 
# We will use TILE SAHI  for multiple signs per image.

