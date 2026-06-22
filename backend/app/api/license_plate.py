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
_paddle_reader = None
_tesseract_available = None

# ── OCR Engine Switch ──────────────────────────────────────────────
# Set USE_PADDLE = True  →  PaddleOCR (new, potentially higher accuracy)
# Set USE_PADDLE = False →  EasyOCR  (original, proven pipeline)
USE_PADDLE = True


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


def get_paddle_reader():
    """Lazy-load PaddleOCR reader (heavy init, do it once)."""
    global _paddle_reader
    if _paddle_reader is None:
        from paddleocr import PaddleOCR
        # use_angle_cls=True handles rotated / tilted plate text.
        # lang='en' for alphanumeric plates; show_log=False keeps logs clean.
        # (PaddleOCR 2.x API – no use_gpu kwarg needed for CPU inference)
        _paddle_reader = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            show_log=False,
        )
    return _paddle_reader






def is_tesseract_available():
    """Check if tesseract binary is installed on the system."""
    global _tesseract_available
    if _tesseract_available is None:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            _tesseract_available = True
        except Exception:
            _tesseract_available = False
    return _tesseract_available


# ── Indian state codes ─────────────────────────────────────────────
INDIAN_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "GA",
    "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH",
    "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK",
    "TN", "TR", "TS", "UK", "UP", "WB",
}

ALLOWED_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
ALLOWLIST_STR = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


# ── Perspective correction ─────────────────────────────────────────
def perspective_correct(plate_img: np.ndarray) -> np.ndarray:
    """
    Detect the plate rectangle via contours and warp it to a
    straight, front-facing rectangle. Falls back to the original
    image if no good quadrilateral is found.
    """
    try:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 200)

        # Dilate to close gaps in the plate border
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return plate_img

        # Find the largest contour that approximates a quadrilateral
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours[:5]:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)

                # Order: top-left, top-right, bottom-right, bottom-left
                s = pts.sum(axis=1)
                d = np.diff(pts, axis=1).ravel()
                ordered = np.array([
                    pts[np.argmin(s)],
                    pts[np.argmin(d)],
                    pts[np.argmax(s)],
                    pts[np.argmax(d)],
                ], dtype=np.float32)

                w = max(
                    np.linalg.norm(ordered[0] - ordered[1]),
                    np.linalg.norm(ordered[3] - ordered[2]),
                )
                h = max(
                    np.linalg.norm(ordered[0] - ordered[3]),
                    np.linalg.norm(ordered[1] - ordered[2]),
                )
                if w < 20 or h < 10:
                    continue

                dst = np.array([
                    [0, 0], [w, 0], [w, h], [0, h]
                ], dtype=np.float32)

                M = cv2.getPerspectiveTransform(ordered, dst)
                warped = cv2.warpPerspective(
                    plate_img, M, (int(w), int(h))
                )
                return warped
    except Exception:
        pass
    return plate_img


# ── Multi-scale preprocessing ─────────────────────────────────────
def build_candidates(plate_img: np.ndarray) -> list[np.ndarray]:
    """
    Generate many preprocessed versions of the plate at multiple
    scales for OCR to try.
    """
    candidates = []

    for target_h in (100, 150, 200):
        h, w = plate_img.shape[:2]
        if h < 5 or w < 5:
            continue
        scale = target_h / max(h, 1)
        resized = cv2.resize(
            plate_img,
            (max(1, int(w * scale)), target_h),
            interpolation=cv2.INTER_CUBIC,
        )
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # 1. Bilateral + Adaptive threshold
        p1 = cv2.bilateralFilter(gray, 11, 17, 17)
        p1 = cv2.adaptiveThreshold(
            p1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 8,
        )
        candidates.append(p1)

        # 2. CLAHE + Otsu
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        p2 = clahe.apply(gray)
        _, p2 = cv2.threshold(p2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(p2)

        # 3. Morphological + Otsu
        p3 = cv2.GaussianBlur(gray, (3, 3), 0)
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        p3 = cv2.morphologyEx(p3, cv2.MORPH_CLOSE, kern)
        _, p3 = cv2.threshold(p3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(p3)

        # 4. Raw grayscale
        candidates.append(gray)

        # 5. Sharpen + Otsu
        sharp_k = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        p5 = cv2.filter2D(gray, -1, sharp_k)
        _, p5 = cv2.threshold(p5, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(p5)

        # 6. Inverted (white-on-black plates)
        candidates.append(cv2.bitwise_not(p1))

        # 7. Color resized (EasyOCR handles color)
        candidates.append(resized)

    return candidates


# ── OCR runner ─────────────────────────────────────────────────────



def _run_easyocr(reader, img) -> list[tuple[str, float]]:
    """Run EasyOCR on one image, return list of (text, conf)."""
    results = reader.readtext(
        img,
        allowlist=ALLOWLIST_STR,
        detail=1,
        paragraph=False,
    )
    if not results:
        return []
    # Sort left-to-right
    results.sort(key=lambda r: r[0][0][0])
    out = []
    for (_bbox, text, conf) in results:
        clean = "".join(c for c in text.upper() if c in ALLOWED_CHARS)
        if clean and conf > 0.08:
            out.append((clean, conf))
    return out


def _run_paddleocr(reader, img) -> list[tuple[str, float]]:
    """
    Run PaddleOCR on one image, return list of (text, conf) sorted
    left-to-right by bbox x-position.

    PaddleOCR 2.x result shape:
      [ [ [bbox, (text, score)], ... ] ]   (outer list = one element per image)
      bbox = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    """
    if not isinstance(img, np.ndarray):
        return []
    try:
        result = reader.ocr(img, cls=True)
    except Exception:
        return []
    if not result or result[0] is None:
        return []

    # Collect with bbox x so we can sort left-to-right before dropping bbox
    with_x: list[tuple[float, str, float]] = []
    for line in result[0]:
        if line is None:
            continue
        bbox, (text, score) = line
        clean = "".join(c for c in str(text).upper() if c in ALLOWED_CHARS)
        if clean and score > 0.08:
            left_x = float(bbox[0][0])   # top-left corner x
            with_x.append((left_x, clean, float(score)))

    # Sort by leftmost x for correct reading order
    with_x.sort(key=lambda t: t[0])
    return [(text, score) for _, text, score in with_x]


def _run_tesseract(img) -> list[tuple[str, float]]:
    """Run pytesseract on one image if available."""
    try:
        import pytesseract
        configs = [
            '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        ]
        best = ""
        for cfg in configs:
            text = pytesseract.image_to_string(img, config=cfg)
            clean = "".join(c for c in text.upper() if c in ALLOWED_CHARS)
            if len(clean) > len(best):
                best = clean
        if best:
            return [(best, 0.5)]
    except Exception:
        pass
    return []


# ── Main OCR function ─────────────────────────────────────────────
# ── Main OCR function ─────────────────────────────────────────────
def read_plate_ocr(plate_img: np.ndarray) -> str:
    """
    Multi-engine, multi-scale, perspective-corrected OCR pipeline
    optimized for Indian license plates.

    Engine selection is controlled by the module-level USE_PADDLE flag:
      USE_PADDLE = True  → PaddleOCR (primary) + Tesseract (fallback)
      USE_PADDLE = False → EasyOCR   (primary) + Tesseract (fallback)  [original]
    """
    if plate_img is None or plate_img.size == 0:
        return ""
    h, w = plate_img.shape[:2]
    if h < 5 or w < 5:
        return ""

    use_tess = is_tesseract_available()

    # Step 1: Try perspective correction
    corrected_img = perspective_correct(plate_img)

    # Step 2: Build candidates from both original and corrected
    candidates = build_candidates(corrected_img)
    if not np.array_equal(corrected_img, plate_img):
        candidates += build_candidates(plate_img)

    # Step 3: Run OCR on every candidate, collect all readings
    all_readings: list[tuple[str, float]] = []

    if USE_PADDLE:
        # ── PaddleOCR branch ───────────────────────────────────────
        paddle_reader = get_paddle_reader()
        for img in candidates:
            try:
                # PaddleOCR works best with 3-channel images
                if img.ndim == 2:
                    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    img_color = img
                paddle_results = _run_paddleocr(paddle_reader, img_color)
                if paddle_results:
                    combined = "".join(t for t, _ in paddle_results)
                    avg_conf = float(np.mean([c for _, c in paddle_results]))
                    all_readings.append((combined, avg_conf))
            except Exception:
                continue
    else:
        # ── EasyOCR branch (original, unchanged) ──────────────────
        reader = get_ocr_reader()
        for img in candidates:
            try:
                easy_results = _run_easyocr(reader, img)
                if easy_results:
                    combined = "".join(t for t, _ in easy_results)
                    avg_conf = float(np.mean([c for _, c in easy_results]))
                    all_readings.append((combined, avg_conf))
            except Exception:
                continue

    # Also run tesseract on a subset (top 3 candidates) if available
    if use_tess:
        for img in candidates[:3]:
            try:
                tess_results = _run_tesseract(img)
                all_readings.extend(tess_results)
            except Exception:
                continue

    if not all_readings:
        return ""

    # Step 4: Score each reading – prefer longer text + higher conf
    scored = []
    for text, conf in all_readings:
        length_bonus = min(len(text), 10) * 0.03
        score = conf + length_bonus
        scored.append((text, conf, score))
    scored.sort(key=lambda x: x[2], reverse=True)

    # Step 5: Consensus voting – if the top-3 reads share a substring
    #          of length >= 6, prefer that (it's almost certainly correct)
    if len(scored) >= 3:
        top3 = [s[0] for s in scored[:5]]
        for i, a in enumerate(top3):
            for b in top3[i + 1:]:
                common = _longest_common_substring(a, b)
                if len(common) >= 6:
                    # Use the longer reading that contains this common part
                    winner = a if len(a) >= len(b) else b
                    return _post_process_indian(winner)

    best_text = scored[0][0]
    return _post_process_indian(best_text)


def _longest_common_substring(a: str, b: str) -> str:
    """Find the longest common substring between two strings."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return ""
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    length, end = 0, 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > length:
                    length = dp[i][j]
                    end = i
    return a[end - length: end]


# ── Indian plate post-processing ──────────────────────────────────
def _post_process_indian(raw_text: str) -> str:
    """
    Clean up OCR output using Indian plate format knowledge.
    Format: XX 00 X(XX) 0000
    """
    raw = raw_text.replace(" ", "")
    if len(raw) < 4:
        return raw

    corrected = list(raw)

    # Fix first 2 chars – must be letters (state code)
    DIGIT_TO_LETTER = {
        "0": "O", "1": "I", "8": "B", "5": "S", "6": "G", "4": "A",
    }
    for i in range(min(2, len(corrected))):
        if corrected[i] in DIGIT_TO_LETTER:
            corrected[i] = DIGIT_TO_LETTER[corrected[i]]

    # State code validation with common OCR swaps
    state = "".join(corrected[:2])
    if state not in INDIAN_STATES:
        # Try common visual confusions
        swaps = {
            "H": "N", "N": "H", "D": "O", "Q": "O", "U": "V",
            "V": "U", "C": "G", "E": "B", "L": "I", "Z": "2",
            "X": "N", "F": "P", "Y": "V", "W": "M", "K": "X",
        }
        # Try single char swaps first
        found = False
        for i in range(2):
            if corrected[i] in swaps:
                trial = list(corrected[:2])
                trial[i] = swaps[corrected[i]]
                if "".join(trial) in INDIAN_STATES:
                    corrected[i] = swaps[corrected[i]]
                    found = True
                    break
        # If single swap didn't work, try all single-char alternatives
        if not found:
            for i in range(2):
                for alt_from, alt_to in swaps.items():
                    trial = list(corrected[:2])
                    if trial[i] == alt_from:
                        trial[i] = alt_to
                        if "".join(trial) in INDIAN_STATES:
                            corrected[i] = alt_to
                            found = True
                            break
                if found:
                    break

    # Chars at index 2-3 must be digits (district code)
    LETTER_TO_DIGIT = {
        "O": "0", "I": "1", "l": "1", "Z": "2", "S": "5",
        "G": "6", "T": "7", "B": "8", "g": "9",
    }
    for i in range(2, min(4, len(corrected))):
        if corrected[i] in LETTER_TO_DIGIT:
            corrected[i] = LETTER_TO_DIGIT[corrected[i]]

    # If plate is long enough, the chars after district code and before
    # the final digits should be letters (series code)
    if len(corrected) >= 7:
        # Find where the trailing digits start
        last_digit_start = len(corrected)
        for k in range(len(corrected) - 1, 3, -1):
            if corrected[k].isdigit():
                last_digit_start = k
            else:
                break

        # Characters between district code (idx 4) and trailing digits
        # should be letters
        for i in range(4, min(last_digit_start, len(corrected))):
            if corrected[i] in DIGIT_TO_LETTER:
                corrected[i] = DIGIT_TO_LETTER[corrected[i]]

        # Trailing chars should be digits
        for i in range(last_digit_start, len(corrected)):
            if corrected[i] in LETTER_TO_DIGIT:
                corrected[i] = LETTER_TO_DIGIT[corrected[i]]

    # Format with spaces: XX 00 XX 0000
    plate = "".join(corrected)
    m = re.match(r'^([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{1,4})$', plate)
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)} {m.group(4)}"

    return plate


# ── API Endpoint ───────────────────────────────────────────────────
@router.post("/detect")
async def detect_license_plate(file: UploadFile = File(...)):
    is_video = file.content_type.startswith("video/")
    is_image = file.content_type.startswith("image/")

    if not (is_image or is_video):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Must be image or video.",
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

                # Generous padding for better OCR
                pad_x = int((x2 - x1) * 0.12) + 8
                pad_y = int((y2 - y1) * 0.20) + 8
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
