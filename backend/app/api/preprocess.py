import base64
import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter()

def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

@router.post("")
@router.post("/")
async def preprocess_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a valid image file")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # Metrics
    brightness = int(np.mean(img))
    # Convert to grayscale for some metrics
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = int(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = int(gray.std())
    
    # Scale to 0-100 percentage for UI (rough mapping)
    b_score = min(100, max(0, int((brightness / 255) * 100)))
    s_score = min(100, max(0, int((sharpness / 500) * 100)))
    c_score = min(100, max(0, int((contrast / 100) * 100)))
    
    # 1. Original
    original_b64 = encode_image(img)
    
    # 2. Normalized
    # Resize to 640x640 (standard YOLO input size)
    normalized = cv2.resize(img, (640, 640))
    normalized_b64 = encode_image(normalized)
    
    # 3. Enhanced (Low light & Shadows)
    # Apply Gamma Correction followed by CLAHE
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(normalized, table)
    
    lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    enhanced_b64 = encode_image(enhanced)
    
    # 4. Noise & Weather Reduction (Motion Blur, Rain, Noise)
    # Apply a gentle Bilateral Filter to smooth noise/rain without degrading image quality (avoiding "plastic" look)
    denoised_bilateral = cv2.bilateralFilter(enhanced, d=5, sigmaColor=25, sigmaSpace=25)
    
    # Subtle Unsharp Masking to crispen edges for OCR without introducing heavy artifacts
    gaussian = cv2.GaussianBlur(denoised_bilateral, (0, 0), 1.5)
    final_output = cv2.addWeighted(denoised_bilateral, 1.2, gaussian, -0.2, 0)
    
    denoised_b64 = encode_image(final_output)
    
    return {
        "original": original_b64,
        "normalized": normalized_b64,
        "enhanced": enhanced_b64,
        "denoised": denoised_b64,
        "metrics": {
            "brightness": b_score,
            "sharpness": s_score,
            "contrast": c_score
        }
    }
