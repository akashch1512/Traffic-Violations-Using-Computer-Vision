import { useRef, useState } from "react";
import { Upload, X, ShieldAlert, Smartphone, Users, Loader2 } from "lucide-react";

const ROBOFLOW_API_KEY = "8uvtxZId3oOxVg80LO8f";
const MODEL_ID = "3riders/2";

type Prediction = {
  class: string;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
};

type InferenceResult = {
  predictions: Prediction[];
  image: { width: number; height: number };
};

const CLASS_COLORS: Record<string, string> = {
  phone: "#ef4444",
  "using phone": "#ef4444",
  mobile: "#ef4444",
  "using_mobile": "#ef4444",
  "triple riding": "#f59e0b",
  triple: "#f59e0b",
  "more_than_two_persons": "#f59e0b",
  rider: "#3b82f6",
  default: "#8b5cf6",
};

function getColor(cls: string) {
  const key = cls.toLowerCase();
  for (const c in CLASS_COLORS) {
    if (key.includes(c)) return CLASS_COLORS[c];
  }
  return CLASS_COLORS["default"];
}

export default function TripleRiderDetection() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  function handleFile(file: File | undefined) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file.");
      return;
    }
    setError(null);
    setResult(null);
    setFileName(file.name);
    const reader = new FileReader();
    reader.onload = (e) => setImageSrc(e.target?.result as string);
    reader.readAsDataURL(file);
  }

  async function runInference() {
    if (!imageSrc) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Strip the data:image/...;base64, prefix
      const base64 = imageSrc.split(",")[1];

      const response = await fetch(
        `https://serverless.roboflow.com/${MODEL_ID}?api_key=${ROBOFLOW_API_KEY}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body: base64,
        }
      );

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Inference failed.");
      }

      const data: InferenceResult = await response.json();
      setResult(data);
      drawBoxes(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inference failed.");
    } finally {
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

      const label = `${p.class} ${Math.round(p.confidence * 100)}%`;
      ctx.font = "bold 12px Inter, sans-serif";
      const tw = ctx.measureText(label).width;

      ctx.fillStyle = color;
      ctx.fillRect(x, y - 20, tw + 8, 20);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x + 4, y - 5);
    });
  }

  function clearAll() {
    setImageSrc(null);
    setFileName("");
    setResult(null);
    setError(null);
    if (fileRef.current) fileRef.current.value = "";
  }

  const phoneCount =
    result?.predictions.filter((p) =>
      p.class.toLowerCase().includes("phone") || p.class.toLowerCase().includes("mobile")
    ).length ?? 0;
  const tripleCount =
    result?.predictions.filter((p) =>
      p.class.toLowerCase().includes("triple") || 
      p.class.toLowerCase().includes("3") ||
      p.class.toLowerCase().includes("more_than_two")
    ).length ?? 0;

  return (
    <div className="min-h-screen bg-white text-slate-900 flex flex-col">
      {/* Header */}
      <header className="border-b border-slate-100 px-6 py-4 flex items-center gap-3">
        <Users className="h-5 w-5 text-slate-700" />
        <span className="font-semibold text-slate-900 tracking-tight">
          Triple Rider & Phone Detection
        </span>
        <span className="ml-auto text-xs text-slate-400">
          YOLOv8 · Roboflow
        </span>
      </header>

      {/* Main */}
      <main className="flex-1 flex flex-col items-center justify-center px-4 py-12 gap-8">
        {/* Upload area */}
        <div className="w-full max-w-lg">
          {!imageSrc ? (
            <label
              htmlFor="triple-image-input"
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                handleFile(e.dataTransfer.files[0]);
              }}
              className="flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed border-slate-200 bg-slate-50 py-16 px-6 text-center cursor-pointer hover:border-slate-400 hover:bg-slate-100 transition-colors"
            >
              <Upload className="h-8 w-8 text-slate-400" />
              <div>
                <p className="font-medium text-slate-700">Drop an image here</p>
                <p className="text-sm text-slate-400 mt-0.5">
                  or click to browse
                </p>
              </div>
              <input
                id="triple-image-input"
                ref={fileRef}
                type="file"
                accept="image/*"
                className="sr-only"
                onChange={(e) => handleFile(e.target.files?.[0])}
              />
            </label>
          ) : (
            <div className="space-y-3">
              {/* Image + canvas overlay */}
              <div className="relative rounded-xl overflow-hidden border border-slate-200 bg-black">
                <img
                  ref={imgRef}
                  src={imageSrc}
                  alt="Uploaded"
                  className="w-full object-contain max-h-[400px]"
                  onLoad={() => result && drawBoxes(result)}
                />
                <canvas
                  ref={canvasRef}
                  className="absolute inset-0 pointer-events-none"
                  style={{ width: "100%", height: "100%" }}
                />
              </div>

              {/* File name + clear */}
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

        {/* Run button */}
        {imageSrc && (
          <button
            onClick={runInference}
            disabled={loading}
            className="flex items-center gap-2 rounded-lg bg-slate-900 text-white px-6 py-2.5 text-sm font-medium hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Running inference…
              </>
            ) : (
              "Run Detection"
            )}
          </button>
        )}

        {/* Error */}
        {error && (
          <p className="text-sm text-red-600 bg-red-50 rounded-lg px-4 py-2 max-w-lg w-full">
            {error}
          </p>
        )}

        {/* Results summary */}
        {result && (
          <div className="w-full max-w-lg space-y-4">
            {/* Counts */}
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg border border-amber-100 bg-amber-50 px-4 py-3 flex items-center gap-3">
                <Users className="h-5 w-5 text-amber-600 shrink-0" />
                <div>
                  <p className="text-xs text-amber-700 font-medium">Triple Riding</p>
                  <p className="text-2xl font-semibold text-amber-800">
                    {tripleCount}
                  </p>
                </div>
              </div>
              <div className="rounded-lg border border-red-100 bg-red-50 px-4 py-3 flex items-center gap-3">
                <Smartphone className="h-5 w-5 text-red-600 shrink-0" />
                <div>
                  <p className="text-xs text-red-700 font-medium">Phone Usage</p>
                  <p className="text-2xl font-semibold text-red-800">
                    {phoneCount}
                  </p>
                </div>
              </div>
            </div>

            {/* Prediction list */}
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
                          {p.class}
                        </span>
                      </div>
                      <span className="text-sm text-slate-500 tabular-nums">
                        {Math.round(p.confidence * 100)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="text-sm text-slate-500 text-center py-4">
                No objects detected.
              </p>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
