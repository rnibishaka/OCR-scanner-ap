"""
ocr_core.py
Preprocessing + PyTesseract wrapper used by the GUI.

Provides:
- preprocess_for_ocr(cv_img)
- ocr_image(cv_img, config="--psm 6")
- draw_ocr_boxes(img, data, offset=(0,0))
"""

from PIL import Image
import cv2
import numpy as np
import pytesseract

# If Tesseract is not on PATH, edit the line below:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_for_ocr(cv_img):
    """Return a binary image suitable for OCR (grayscale -> denoise -> adaptive thresh)."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    th = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 25, 10)
    return th

def ocr_image(cv_img, config="--psm 6", lang="eng"):
    """
    Run pytesseract OCR on cv_img.
    cv_img may be BGR or grayscale (numpy array).
    Returns: (text, data_dict) where data_dict is pytesseract Output.DICT from image_to_data.
    """
    if len(cv_img.shape) == 3:
        pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    else:
        pil = Image.fromarray(cv_img)
    data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT, config=config, lang=lang)
    text = pytesseract.image_to_string(pil, config=config, lang=lang)
    return text, data

def draw_ocr_boxes(img, data, offset=(0,0), min_confidence=0):
    """
    Draw bounding boxes and short labels onto img (in-place copy recommended by caller).
    data: pytesseract Output.DICT
    offset: (x_off, y_off) added to box coords (useful when OCR ran on ROI).
    min_confidence: integer - ignore words with confidence < min_confidence (use -1 to keep all)
    """
    n = len(data.get('level', []))
    x_off, y_off = offset
    for i in range(n):
        text = data['text'][i].strip()
        conf = -1
        try:
            conf = int(float(data['conf'][i]))
        except Exception:
            conf = -1
        if text == "" or conf < min_confidence:
            continue
        x = int(data['left'][i]) + x_off
        y = int(data['top'][i]) + y_off
        w = int(data['width'][i])
        h = int(data['height'][i])
        # rectangle
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 200, 0), 2)
        # small filled background for label
        label_w = min(w, 200)
        cv2.rectangle(img, (x, max(0, y-18)), (x + label_w, y), (0,200,0), -1)
        cv2.putText(img, text[:30], (x+2, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
    return img
