import { useRef, useState } from "react";
import { Upload, X, Camera, Car, Loader2 } from "lucide-react";

type Prediction = {
  class: string;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
  ocr_text: string;
};

type InferenceResult = {
  predictions: Prediction[];
  image: { width: number; height: number };
  is_video?: boolean;
};

const CLASS_COLORS: Record<string, string> = {
  license_plate: "#facc15",
  default: "#3b82f6",
};

function getColor(cls: string) {
  const key = cls.toLowerCase();
  return CLASS_COLORS[key] ?? CLASS_COLORS["default"];
}

export default function LicensePlateDetection() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  function handleFile(file: File | undefined) {
    if (!file) return;
    if (!file.type.startsWith("image/") && !file.type.startsWith("video/")) {
      setError("Please upload an image or video file.");
      return;
    }
    setError(null);
    setResult(null);
    setFileName(file.name);
    setSelectedFile(file);
    const reader = new FileReader();
    reader.onload = (e) => setImageSrc(e.target?.result as string);
    reader.readAsDataURL(file);
  }

  async function runInference() {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 120_000);

    try {
      const response = await fetch("/api/license-plate/detect", {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeout);

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Inference failed.");
      }

      const data: InferenceResult = await response.json();
      setResult(data);
      if (!data.is_video) {
        drawBoxes(data);
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setError("Request timed out. The backend may still be loading the AI model — please try again.");
      } else {
        setError(err instanceof Error ? err.message : "Inference failed.");
      }
    } finally {
      clearTimeout(timeout);
      setLoading(false);
    }
  }

  function drawBoxes(data: InferenceResult) {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;

    const { width: iw, height: ih } = data.image;
    const scaleX = img.clientWidth / iw;
    const scaleY = img.clientHeight / ih;

    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    data.predictions.forEach((p) => {
      const color = getColor(p.class);
      const x = (p.x - p.width / 2) * scaleX;
      const y = (p.y - p.height / 2) * scaleY;
      const w = p.width * scaleX;
      const h = p.height * scaleY;

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);

      const label = `${p.ocr_text || p.class} ${Math.round(p.confidence * 100)}%`;
      ctx.font = "bold 12px Inter, sans-serif";
      const tw = ctx.measureText(label).width;

      ctx.fillStyle = color;
      ctx.fillRect(x, y - 20, tw + 8, 20);
      ctx.fillStyle = "#1e293b";
      ctx.fillText(label, x + 4, y - 5);
    });
  }

  function clearAll() {
    setImageSrc(null);
    setFileName("");
    setSelectedFile(null);
    setResult(null);
    setError(null);
    if (fileRef.current) fileRef.current.value = "";
  }

  return (
    <div className="min-h-screen bg-white text-slate-900 flex flex-col">
      <header className="border-b border-slate-100 px-6 py-4 flex items-center gap-3">
        <Camera className="h-5 w-5 text-slate-700" />
        <span className="font-semibold text-slate-900 tracking-tight">
          License Plate & OCR
        </span>
        <span className="ml-auto text-xs text-slate-400">
          YOLOv8 + Tesseract OCR
        </span>
      </header>

      <main className="flex-1 flex flex-col items-center justify-center px-4 py-12 gap-8">
        <div className="w-full max-w-lg">
          {!imageSrc ? (
            <label
              htmlFor="plate-image-input"
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                handleFile(e.dataTransfer.files[0]);
              }}
              className="flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed border-slate-200 bg-slate-50 py-16 px-6 text-center cursor-pointer hover:border-slate-400 hover:bg-slate-100 transition-colors"
            >
              <Upload className="h-8 w-8 text-slate-400" />
              <div>
                <p className="font-medium text-slate-700">Drop a vehicle image or video here</p>
                <p className="text-sm text-slate-400 mt-0.5">
                  or click to browse
                </p>
              </div>
              <input
                id="plate-image-input"
                ref={fileRef}
                type="file"
                accept="image/*,video/*"
                className="sr-only"
                onChange={(e) => handleFile(e.target.files?.[0])}
              />
            </label>
          ) : (
            <div className="space-y-3">
              <div className="relative rounded-xl overflow-hidden border border-slate-200 bg-black flex justify-center">
                {selectedFile?.type.startsWith("video/") ? (
                  <video
                    src={imageSrc!}
                    controls
                    className="w-full max-h-[400px] object-contain"
                  />
                ) : (
                  <>
                    <img
                      ref={imgRef}
                      src={imageSrc!}
                      alt="Uploaded"
                      className="w-full object-contain max-h-[400px]"
                      onLoad={() => result && !result.is_video && drawBoxes(result)}
                    />
                    <canvas
                      ref={canvasRef}
                      className="absolute inset-0 pointer-events-none"
                      style={{ width: "100%", height: "100%" }}
                    />
                  </>
                )}
              </div>

              <div className="flex items-center justify-between text-sm text-slate-500">
                <span className="truncate">{fileName}</span>
                <button
                  onClick={clearAll}
                  className="ml-3 shrink-0 flex items-center gap-1 text-slate-400 hover:text-slate-700 transition-colors"
                >
                  <X className="h-4 w-4" /> Clear
                </button>
              </div>
            </div>
          )}
        </div>

        {imageSrc && (
          <button
            onClick={runInference}
            disabled={loading}
            className="flex items-center gap-2 rounded-lg bg-slate-900 text-white px-6 py-2.5 text-sm font-medium hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Reading Plates…
              </>
            ) : (
              "Detect License Plates"
            )}
          </button>
        )}

        {error && (
          <p className="text-sm text-red-600 bg-red-50 rounded-lg px-4 py-2 max-w-lg w-full">
            {error}
          </p>
        )}

        {result && (
          <div className="w-full max-w-lg space-y-4">
            {result.predictions.length > 0 ? (
              <div className="rounded-xl border border-slate-100 divide-y divide-slate-100">
                {result.predictions.map((p, i) => {
                  const color = getColor(p.class);
                  return (
                    <div
                      key={i}
                      className="flex items-center justify-between px-4 py-3"
                    >
                      <div className="flex items-center gap-2.5">
                        <span
                          className="h-2.5 w-2.5 rounded-full shrink-0"
                          style={{ backgroundColor: color }}
                        />
                        <span className="text-sm font-medium text-slate-800 capitalize">
                          {p.ocr_text || "Unreadable Plate"}
                        </span>
                      </div>
                      <span className="text-sm text-slate-500 tabular-nums">
                        {Math.round(p.confidence * 100)}% Conf
                      </span>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="text-sm text-slate-500 text-center py-4">
                No license plates detected.
              </p>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
