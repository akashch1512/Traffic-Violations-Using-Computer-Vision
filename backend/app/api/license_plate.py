import io
import re
import cv2
import numpy as np
import tempfile
import os
from fastapi import APIRouter, File, HTTPException, UploadFile
from pathlib import Path
from ultralytics import YOLO

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "backend" / "license_plate_detector.pt"

_model = None
_ocr_reader = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model not found at {MODEL_PATH}")
        _model = YOLO(str(MODEL_PATH))
    return _model


def get_ocr_reader():
    """Lazy-load EasyOCR reader (heavy init, do it once)."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _ocr_reader


def read_plate_ocr(plate_img: np.ndarray) -> str:
    """
    Run multiple preprocessing pipelines on the cropped plate image,
    pick the best OCR result. Optimized for Indian license plates.
    """
    if plate_img is None or plate_img.size == 0:
        return ""

    h, w = plate_img.shape[:2]
    if h < 5 or w < 5:
        return ""

    reader = get_ocr_reader()
    candidates = []

    # Scale plate to a large consistent height for better OCR
    target_h = 150
    scale = target_h / max(h, 1)
    plate_resized = cv2.resize(
        plate_img, (max(1, int(w * scale)), target_h),
        interpolation=cv2.INTER_CUBIC
    )
    gray = cv2.cvtColor(plate_resized, cv2.COLOR_BGR2GRAY)

    # --- Pipeline 1: Bilateral filter + Adaptive threshold ---
    p1 = cv2.bilateralFilter(gray, 11, 17, 17)
    p1 = cv2.adaptiveThreshold(
        p1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 8
    )
    candidates.append(p1)

    # --- Pipeline 2: CLAHE + Otsu ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    p2 = clahe.apply(gray)
    _, p2 = cv2.threshold(p2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(p2)

    # --- Pipeline 3: Morphological close + Otsu ---
    p3 = cv2.GaussianBlur(gray, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    p3 = cv2.morphologyEx(p3, cv2.MORPH_CLOSE, kernel)
    _, p3 = cv2.threshold(p3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(p3)

    # --- Pipeline 4: Raw grayscale ---
    candidates.append(gray)

    # --- Pipeline 5: Sharpen + threshold ---
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    p5 = cv2.filter2D(gray, -1, sharpen_kernel)
    _, p5 = cv2.threshold(p5, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(p5)

    # --- Pipeline 6: Resized color image (EasyOCR handles color well) ---
    candidates.append(plate_resized)

    best_text = ""
    best_conf = 0.0
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

    for img in candidates:
        try:
            results = reader.readtext(
                img,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                detail=1,
                paragraph=False,
            )
            if not results:
                continue

            # Sort fragments LEFT-TO-RIGHT by x-coordinate
            results.sort(key=lambda r: r[0][0][0])

            texts, confs = [], []
            for (bbox, text, conf) in results:
                clean = "".join(c for c in text.upper() if c in allowed)
                if clean and conf > 0.1:
                    texts.append(clean)
                    confs.append(conf)

            if texts:
                combined = "".join(texts)
                avg_conf = float(np.mean(confs))
                # Prefer longer reads at similar confidence
                score = avg_conf + (len(combined) * 0.02)
                if score > best_conf:
                    best_conf = score
                    best_text = combined
        except Exception:
            continue

    # Post-process for Indian plates
    raw = best_text.replace(" ", "")
    if len(raw) >= 4:
        corrected = list(raw)

        # Indian state codes (first 2 chars must be letters)
        INDIAN_STATES = {
            "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "GA",
            "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH",
            "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK",
            "TN", "TR", "TS", "UK", "UP", "WB",
        }

        # Fix common OCR confusions for first 2 chars (must be letters)
        DIGIT_TO_LETTER = {"0": "O", "1": "I", "8": "B", "5": "S", "6": "G", "4": "A"}
        if len(corrected) >= 2:
            for i in range(2):
                if corrected[i] in DIGIT_TO_LETTER:
                    corrected[i] = DIGIT_TO_LETTER[corrected[i]]

        # Check if state code matches; try common substitutions if not
        state = "".join(corrected[:2])
        if state not in INDIAN_STATES:
            # Try swapping H↔N (very common confusion)
            swaps = {"H": "N", "N": "H", "D": "O", "Q": "O"}
            for i in range(2):
                if corrected[i] in swaps:
                    trial = list(corrected[:2])
                    trial[i] = swaps[corrected[i]]
                    if "".join(trial) in INDIAN_STATES:
                        corrected[i] = swaps[corrected[i]]
                        break

        # Format: insert spaces like "TN 87 A 8935"
        plate = "".join(corrected)
        # Try to match Indian plate regex: 2 letters + 2 digits + 1-2 letters + 4 digits
        m = re.match(r'^([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{1,4})$', plate)
        if m:
            best_text = f"{m.group(1)} {m.group(2)} {m.group(3)} {m.group(4)}"
        else:
            best_text = plate

    return best_text


@router.post("/detect")
async def detect_license_plate(file: UploadFile = File(...)):
    is_video = file.content_type.startswith("video/")
    is_image = file.content_type.startswith("image/")

    if not (is_image or is_video):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Must be image or video."
        )

    model = get_model()
    predictions = []

    def process_frame(img_frame):
        frame_preds = []
        results = model(img_frame, verbose=False)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])
                cls_name = model.names[cls_idx]

                # Add padding around the plate crop for better OCR
                pad_x = int((x2 - x1) * 0.1) + 5
                pad_y = int((y2 - y1) * 0.15) + 5
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(img_frame.shape[1], x2 + pad_x)
                cy2 = min(img_frame.shape[0], y2 + pad_y)
                plate_img = img_frame[cy1:cy2, cx1:cx2]

                text = ""
                try:
                    text = read_plate_ocr(plate_img)
                except Exception:
                    text = ""

                frame_preds.append({
                    "class": cls_name,
                    "confidence": conf,
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "ocr_text": text,
                })
        return frame_preds

    content = await file.read()
    width, height = 0, 0

    if is_image:
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not read image")
        width, height = img.shape[1], img.shape[0]
        predictions.extend(process_frame(img))
    else:
        fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        with os.fdopen(fd, 'wb') as f:
            f.write(content)

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail="Could not open video")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_count = 0
        best_preds = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 10 == 0:
                preds = process_frame(frame)
                for p in preds:
                    txt = p["ocr_text"]
                    key = txt if txt else f"_unknown_{frame_count}"
                    if key not in best_preds or p["confidence"] > best_preds[key]["confidence"]:
                        best_preds[key] = p

            frame_count += 1
            if frame_count > 300:
                break

        cap.release()
        os.remove(temp_path)
        predictions = list(best_preds.values())

    return {
        "predictions": predictions,
        "image": {
            "width": width,
            "height": height,
        },
        "is_video": is_video,
    }
