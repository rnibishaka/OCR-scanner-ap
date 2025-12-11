"""
ocr_gui.py
Main GUI app (PyQt5) that uses ocr_core.py functions.

Features:
- Load image
- Live camera preview + capture
- ROI selection (click & drag)
- Run OCR (on ROI or full image) with optional preprocessing
- Overlay preview (bounding boxes) and display extracted text
"""

import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ocr_core import preprocess_for_ocr, ocr_image, draw_ocr_boxes

# If Tesseract binary not found, set path here:
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def cv2_to_qimage(cv_img):
    """Convert BGR cv image to QImage"""
    h, w = cv_img.shape[:2]
    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    bytes_per_line = 3 * w
    return QtGui.QImage(cv_img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

class OCRScanner(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR Scanner (separate files)")
        self.setGeometry(120, 80, 1150, 680)

        # state
        self.cv_image = None
        self.display_image = None
        self.roi = None
        self.capture = None
        self.capturing = False

        # UI
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout(central)

        left = QtWidgets.QVBoxLayout()
        self.image_label = QtWidgets.QLabel("No image")
        self.image_label.setFixedSize(780, 520)
        self.image_label.setStyleSheet("background:#111; border:1px solid #333;")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        left.addWidget(self.image_label)

        # buttons row
        row = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        row.addWidget(self.load_btn)

        self.cam_btn = QtWidgets.QPushButton("Open Camera")
        self.cam_btn.clicked.connect(self.toggle_camera)
        row.addWidget(self.cam_btn)

        self.cap_btn = QtWidgets.QPushButton("Capture Frame")
        self.cap_btn.clicked.connect(self.capture_frame)
        self.cap_btn.setEnabled(False)
        row.addWidget(self.cap_btn)

        self.ocr_btn = QtWidgets.QPushButton("Run OCR")
        self.ocr_btn.clicked.connect(self.run_ocr)
        self.ocr_btn.setEnabled(False)
        row.addWidget(self.ocr_btn)

        left.addLayout(row)

        self.overlay_cb = QtWidgets.QCheckBox("Overlay preview")
        self.overlay_cb.setChecked(True)
        left.addWidget(self.overlay_cb)

        h.addLayout(left)

        right = QtWidgets.QVBoxLayout()
        self.text_edit = QtWidgets.QTextEdit()
        right.addWidget(self.text_edit)

        self.preproc_cb = QtWidgets.QCheckBox("Use preprocessing (recommended)")
        self.preproc_cb.setChecked(True)
        right.addWidget(self.preproc_cb)

        right.addWidget(QtWidgets.QLabel("Tesseract PSM:"))
        self.psm = QtWidgets.QComboBox()
        self.psm.addItems(["3 - Full page","6 - Single block","11 - Sparse text","12 - Sparse+OSD"])
        self.psm.setCurrentIndex(1)
        right.addWidget(self.psm)

        save_row = QtWidgets.QHBoxLayout()
        self.save_btn = QtWidgets.QPushButton("Save text")
        self.save_btn.clicked.connect(self.save_text)
        self.save_btn.setEnabled(False)
        save_row.addWidget(self.save_btn)
        right.addLayout(save_row)

        h.addLayout(right)

        # mouse ROI
        self._mouse_down = False
        self._start_pos = None
        self.image_label.installEventFilter(self)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._grab_frame)

        self.status = self.statusBar()
        self.status.showMessage("Ready")

    # image/camera
    def load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Failed to load image.")
            return
        self._set_image(img)
        self.status.showMessage(f"Loaded: {path}")
        self.ocr_btn.setEnabled(True)
        self.save_btn.setEnabled(False)

    def toggle_camera(self):
        if not self.capturing:
            self.capture = cv2.VideoCapture(1, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(1)
            if not self.capture.isOpened():
                QtWidgets.QMessageBox.critical(self, "Error", "Cannot open camera.")
                return
            self.capturing = True
            self.cam_btn.setText("Close Camera")
            self.cap_btn.setEnabled(True)
            self.timer.start(30)
            self.status.showMessage("Camera opened.")
        else:
            self.timer.stop()
            if self.capture:
                self.capture.release()
            self.capturing = False
            self.cam_btn.setText("Open Camera")
            self.cap_btn.setEnabled(False)
            self.status.showMessage("Camera closed.")

    def _grab_frame(self):
        if not self.capture:
            return
        ret, frame = self.capture.read()
        if not ret:
            return
        frame = cv2.flip(frame, -1)
        self._set_image(frame, live=True)

    def capture_frame(self):
        if not self.capture:
            return
        ret, frame = self.capture.read()
        if not ret:
            self.status.showMessage("Capture failed.")
            return
        frame = cv2.flip(frame, -1)
        self._set_image(frame)
        self.status.showMessage("Frame captured.")

    def _set_image(self, img, live=False):
        self.cv_image = img.copy()
        self.display_image = img.copy()
        self._update_label()
        self.ocr_btn.setEnabled(True)

    def _update_label(self):
        if self.display_image is None:
            return
        qimg = cv2_to_qimage(self.display_image)
        pix = QtGui.QPixmap.fromImage(qimg)
        pix = pix.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.image_label.setPixmap(pix)

    # ROI via mouse
    def eventFilter(self, src, event):
        if src is self.image_label:
            if event.type() == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton and self.cv_image is not None:
                self._mouse_down = True
                self._start_pos = event.pos()
                return True
            elif event.type() == QtCore.QEvent.MouseMove and self._mouse_down and self.cv_image is not None:
                self._draw_temp_roi(self._start_pos, event.pos())
                return True
            elif event.type() == QtCore.QEvent.MouseButtonRelease and self._mouse_down and self.cv_image is not None:
                self._mouse_down = False
                self._finalize_roi(self._start_pos, event.pos())
                return True
        return super().eventFilter(src, event)

    def _label_to_image(self, qpoint):
        lbl_w, lbl_h = self.image_label.width(), self.image_label.height()
        img_h, img_w = self.cv_image.shape[:2]
        scale = min(lbl_w / img_w, lbl_h / img_h)
        new_w, new_h = int(img_w*scale), int(img_h*scale)
        x_off = (lbl_w - new_w)//2
        y_off = (lbl_h - new_h)//2
        x = (qpoint.x() - x_off) / scale
        y = (qpoint.y() - y_off) / scale
        return int(x), int(y)

    def _draw_temp_roi(self, start, cur):
        temp = self.cv_image.copy()
        x1,y1 = self._label_to_image(start)
        x2,y2 = self._label_to_image(cur)
        cv2.rectangle(temp, (x1,y1), (x2,y2), (255,0,0), 2)
        self.display_image = temp
        self._update_label()

    def _finalize_roi(self, start, end):
        x1,y1 = self._label_to_image(start)
        x2,y2 = self._label_to_image(end)
        x1,x2 = sorted((max(0,x1), max(0,x2)))
        y1,y2 = sorted((max(0,y1), max(0,y2)))
        h,w = self.cv_image.shape[:2]
        x1,x2 = np.clip([x1,x2], 0, w-1)
        y1,y2 = np.clip([y1,y2], 0, h-1)
        if x2 - x1 < 10 or y2 - y1 < 10:
            self.roi = None
            self.status.showMessage("ROI too small; ignored.")
        else:
            self.roi = (x1,y1,x2,y2)
            self.status.showMessage(f"ROI set: {self.roi}")
        self._draw_roi_box()

    def _draw_roi_box(self):
        if self.cv_image is None:
            return
        out = self.cv_image.copy()
        if self.roi is not None:
            x1,y1,x2,y2 = self.roi
            cv2.rectangle(out, (x1,y1), (x2,y2), (255,0,0), 2)
        self.display_image = out
        self._update_label()

    # OCR
    def run_ocr(self):
        if self.cv_image is None:
            QtWidgets.QMessageBox.warning(self, "No image", "Load or capture an image first.")
            return
        target = None
        if self.roi is not None:
            x1,y1,x2,y2 = self.roi
            target = self.cv_image[y1:y2, x1:x2]
        else:
            target = self.cv_image

        psm_map = {0:"3", 1:"6", 2:"11", 3:"12"}
        psm_val = psm_map.get(self.psm.currentIndex(), "6")
        config = f"--psm {psm_val}"

        if self.preproc_cb.isChecked():
            proc = preprocess_for_ocr(target)
            text, data = ocr_image(proc, config=config)
        else:
            text, data = ocr_image(target, config=config)

        self.text_edit.setPlainText(text.strip())
        self.save_btn.setEnabled(True)

        # Overlay
        overlay = self.cv_image.copy()
        if self.roi is not None:
            # draw on a copy of roi then paste (data coords relative to target)
            temp = overlay.copy()
            draw_ocr_boxes(temp, data, offset=(self.roi[0], self.roi[1]))
            overlay = temp
        else:
            draw_ocr_boxes(overlay, data, offset=(0,0))

        if not self.overlay_cb.isChecked():
            overlay = self.cv_image.copy()

        self.display_image = overlay
        self._update_label()
        self.status.showMessage("OCR finished.")

    def save_text(self):
        text = self.text_edit.toPlainText()
        if not text.strip():
            QtWidgets.QMessageBox.information(self, "No text", "No extracted text to save.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save text", "extracted.txt", "Text files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self.status.showMessage(f"Saved text: {path}")

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = OCRScanner()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
