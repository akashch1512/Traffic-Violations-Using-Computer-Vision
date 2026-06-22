<div align="center">
  <h1>🚦 Traffic Violation AI Pipeline</h1>
  <p>An intelligent, end-to-end computer vision web application to detect and analyze traffic violations.</p>

  <!-- Badges -->
  <p>
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python" />
    <img src="https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi" alt="FastAPI" />
    <img src="https://img.shields.io/badge/React-19-61DAFB.svg?logo=react" alt="React" />
    <img src="https://img.shields.io/badge/Vite-8.0-646CFF.svg?logo=vite" alt="Vite" />
    <img src="https://img.shields.io/badge/YOLO-Ultralytics-FF9900.svg" alt="YOLO" />
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License" />
  </p>
</div>

---

## 📖 What it does

**Traffic Violation AI** is a comprehensive proof-of-concept application built to automate the detection of traffic violations from video feeds or images. Utilizing state-of-the-art object detection and optical character recognition (OCR) models, it identifies common infractions and logs actionable evidence.

### 🎯 Key Features & Benefits

- 🏍️ **Helmet Detection:** Automatically flags riders not wearing helmets.
- 👨‍👩‍👦 **Triple Rider Detection:** Identifies motorcycles carrying more than the legal limit of passengers.
- 🚗 **Seatbelt & Mobile Phone Detection:** Spots drivers using their phones or failing to wear seatbelts.
- 📸 **License Plate OCR:** Extracts text from license plates using PaddleOCR or EasyOCR with built-in preprocessing for high accuracy.
- 📊 **End-to-End Pipeline Demo:** Upload traffic videos for frame-by-frame analysis, complete with analytics, charts, and interactive evidence review.
- ⚡ **Modern Stack:** Blazing fast React+Vite frontend powered by a robust Python FastAPI backend.

This tool is incredibly useful for civic tech projects, smart city hackathons, or automated traffic enforcement systems looking to reduce human monitoring effort.

---

## 🚀 How to get started

### Prerequisites

Ensure you have the following installed on your system:
- **Python 3.10+**
- **Node.js 18+** & **npm**

### 1. Clone the repository

```bash
git clone https://github.com/akashch1512/Traffic-Violations-Using-Computer-Vision.git
cd Traffic-Violations-Using-Computer-Vision
```

### 2. Backend Setup

The backend handles all AI inferences, frame extraction, and OCR.

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
> **Note:** The API health check is available at `http://localhost:8000/` and interactive Swagger docs at `http://localhost:8000/docs`. By default, PaddleOCR is enabled. You can switch to EasyOCR in `backend/app/api/license_plate.py`.

### 3. Frontend Setup

The frontend provides the interactive dashboard and video analysis UI.

```bash
# Open a new terminal instance
cd frontend

# Install Node dependencies
npm install

# Start the Vite development server
npm run dev -- --port 5173
```
> Access the web application at **[http://localhost:5173/](http://localhost:5173/)**.

### 💡 Usage Example

1. Open the frontend URL in your browser.
2. Go to the **Pipeline Demo** or **Traffic Analysis** tab.
3. Upload a sample `.mp4` traffic clip.
4. Click **Analyze video**. The backend will extract frames, run the YOLO models and OCR, and return a comprehensive summary of vehicles detected, violations found, and license plates matched!

---

## 🤝 Who maintains and contributes

This project is actively maintained. We welcome contributions from the open-source community—whether it's improving OCR accuracy, adding new YOLO classes, or refining the React dashboard!

- **Maintainer:** [akashch1512](https://github.com/akashch1512)
- **Contributing:** We love pull requests! Please refer to `docs/CONTRIBUTING.md` (if available) or simply open an issue to discuss proposed changes before submitting a PR.
  1. Fork the Project
  2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
  3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
  4. Push to the Branch (`git push origin feature/AmazingFeature`)
  5. Open a Pull Request

---

## ❓ Where to get help

Having trouble setting up the environment or modifying a model?
- **Issues:** Check the [GitHub Issues](https://github.com/akashch1512/Traffic-Violations-Using-Computer-Vision/issues) tab to see if your problem has been discussed.
- **Discussions:** Use the GitHub Discussions board for general questions and architecture discussions.
- **Documentation:** For backend endpoint details, visit the auto-generated Swagger UI at `http://localhost:8000/docs` while the server is running.

<div align="center">
  <p><i>Built with ❤️ for safer roads.</i></p>
</div>
