import cv2
import numpy as np
import os
import pandas as pd

# Define folders and labels
folders = {
    "Gallopavonis": r"D:/Eimeria_Classification_TFG/Eimeria_Classification_TFG/imatges/Especies_sanititzades/Eimeria gallopavonis 079",  
    "Meleagrimitis": r"D:/Eimeria_Classification_TFG/Eimeria_Classification_TFG/imatges/Especies_sanititzades/Eimeria meleagrimitis 068",  
    "Dispersa": r"D:/Eimeria_Classification_TFG/Eimeria_Classification_TFG/imatges/Especies_sanititzades/Eimeria dispersa 060",       
    "Innocua": r"D:/Eimeria_Classification_TFG/Eimeria_Classification_TFG/imatges/Especies_sanititzades/Eimeria innocua 088"        
}

# List to store all results
results = []

# Parameters
MIN_AREA = 10000
MAX_AREA = 30000
ECCENTRICITY_THRESHOLD = 0.75
BORDER_MARGIN = 2


# === IMAGE PROCESSING LOOP ===
for label, folder_path in folders.items():
    print(f"\nProcessing folder: {folder_path}")

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
            continue
        img_file = os.path.join(folder_path, filename)
        print(f"  -> Processing {filename}...")

        img = cv2.imread(img_file)
        if img is None:
            print(f"Error reading {img_file}. Skipping...")
            continue

        # === PREPROCESSING ===
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Morphological gradient to enhance edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph_gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)
        
        # Threshold the image
        thresh = cv2.adaptiveThreshold(morph_gradient, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
        
        # Morphological closing to fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # === FEATURE EXTRACTION ===
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if contours:
            for contour in contours:
                area = cv2.contourArea(contour)
                if MIN_AREA <= area <= MAX_AREA and len(contour) >= 5:
                    

                    ellipse = cv2.fitEllipse(contour)
                    (center_x, center_y), (axis1, axis2), angle = ellipse

                    length = max(axis1, axis2)
                    width = min(axis1, axis2)

                    perimeter = cv2.arcLength(contour, True)
                    aspect_ratio = width / length if length != 0 else 0

                    IC = length / width if width != 0 else 0

                    x, y, w, h = cv2.boundingRect(contour)
                    extent = area / (w * h) if w * h != 0 else 0
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area != 0 else 0

                    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

                    minor_axis = min(axis1, axis2)
                    major_axis = max(axis1, axis2)

                    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis != 0 else 0
                    
                    # FILTERS
                    if eccentricity >= ECCENTRICITY_THRESHOLD:
                        continue

                    
                    img_height, img_width = gray.shape[:2]

                    if (
                        x <= BORDER_MARGIN or
                        y <= BORDER_MARGIN or
                        x + w >= img_width - BORDER_MARGIN or
                        y + h >= img_height - BORDER_MARGIN
                    ):
                        continue

                    # All checks passed
                    results.append({
                        "Filename": filename,
                        "Species": label,
                        "Length": round(length, 2),
                        "Width": round(width, 2),
                        "Area": round(area, 2),
                        "Perimeter": round(perimeter, 2),
                        "Aspect_Ratio": round(aspect_ratio, 2),
                        "IC": round(IC, 2),
                        "Extent": round(extent, 2),
                        "Solidity": round(solidity, 2),
                        "Circularity": round(circularity, 2),
                        "Eccentricity": round(eccentricity, 2)
                    })
                    
        else:
            print("No contours found for", filename)
        


# === SAVE RESULTS ===
df = pd.DataFrame(results)
output_excel = r"D:/Eimeria_Classification_TFG/Eimeria_Classification_TFG/data/auto_measurements.xlsx"

df.to_excel(output_excel, index=False)
print(f"\n✅ Results saved to {output_excel}")

