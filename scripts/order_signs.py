def order_signs(detections, layout, direction):
    """
    detections: lista de bboxes con (x_center, y_center, class_id, conf)
    layout: 'rows' | 'columns'
    direction: 'ltr' | 'rtl'
    """
    if layout == "rows":
      
        detections.sort(key=lambda d: d.y_center)
        rows = []
        current_row = [detections[0]]
        ROW_THRESHOLD = 0.05   
        
        for det in detections[1:]:
            if abs(det.y_center - current_row[-1].y_center) < ROW_THRESHOLD:
                current_row.append(det)
            else:
                rows.append(current_row)
                current_row = [det]
        rows.append(current_row)
        
  
        reverse = (direction == "rtl")
        for row in rows:
            row.sort(key=lambda d: d.x_center, reverse=reverse)
        
        return [det for row in rows for det in row]
    
    else:  # columns
        
        detections.sort(key=lambda d: d.x_center,
                       reverse=(direction == "rtl"))
        cols = []
        current_col = [detections[0]]
        COL_THRESHOLD = 0.05
        
        for det in detections[1:]:
            if abs(det.x_center - current_col[-1].x_center) < COL_THRESHOLD:
                current_col.append(det)
            else:
                cols.append(current_col)
                current_col = [det]
        cols.append(current_col)
        
       
        for col in cols:

            col.sort(key=lambda d: d.y_center)
        
        return [det for col in cols for det in col]


# TODO:  This script will turn out to be app/utils/order_signs.py
# When backend is built in FastAPI , it will come after step 0 . CLAHE . 
# It will be called by the backend to order the signs in the image.
# Ancient Egyptian hieroglyphs are written in rows or columns.
# This script will order the signs in the image.
