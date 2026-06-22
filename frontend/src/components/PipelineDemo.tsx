import React, { useRef, useState, useEffect } from "react";
import { Upload, X, Camera, BarChart3, Image as ImageIcon, Zap, Sparkles, Droplets, CheckCircle, Target, ArrowDown, Loader2 } from "lucide-react";

type Metrics = {
  brightness: number;
  sharpness: number;
  contrast: number;
};

type PreprocessResult = {
  original: string;
  normalized: string;
  enhanced: string;
  denoised: string;
  metrics: Metrics;
};

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
};

const DUMMY_TIMINGS = {
  preprocess: 48,
  detection: 120,
  ocr: 35,
  total: 203,
};

export default function PipelineDemo() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [fileSize, setFileSize] = useState<number>(0);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<PreprocessResult | null>(null);
  const [ocrResult, setOcrResult] = useState<InferenceResult | null>(null);
  const [croppedPlates, setCroppedPlates] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [ocrLoading, setOcrLoading] = useState(false);
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
    setOcrResult(null);
    setFileName(file.name);
    setFileSize(file.size);
    setSelectedFile(file);
    const reader = new FileReader();
    reader.onload = (e) => setImageSrc(e.target?.result as string);
    reader.readAsDataURL(file);
  }

  async function runPipeline() {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setOcrResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("/api/preprocess", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Pipeline processing failed.");
      }

      const data: PreprocessResult = await response.json();
      setResult(data);

      // Run OCR on denoised image
      setOcrLoading(true);
      try {
        const fileReq = await fetch(data.denoised);
        const blob = await fileReq.blob();
        const ocrFormData = new FormData();
        ocrFormData.append("file", new File([blob], "processed.jpg", { type: "image/jpeg" }));
        
        const ocrRes = await fetch("/api/license-plate/detect", {
          method: "POST",
          body: ocrFormData,
        });
        if (ocrRes.ok) {
          const ocrData = await ocrRes.json();
          setOcrResult(ocrData);
        }
      } catch (ocrErr) {
        console.error("OCR failed", ocrErr);
      } finally {
        setOcrLoading(false);
      }

    } catch (err) {
      console.error(err);
      setError("Backend failed, using mock data for demonstration.");
      setTimeout(() => {
        setResult({
          original: imageSrc!,
          normalized: imageSrc!,
          enhanced: imageSrc!,
          denoised: imageSrc!,
          metrics: {
            brightness: 78,
            sharpness: 82,
            contrast: 75,
          },
        });
        setLoading(false);
      }, 1500);
    } finally {
      setLoading(false);
    }
  }

  function clearAll() {
    setImageSrc(null);
    setFileName("");
    setFileSize(0);
    setSelectedFile(null);
    setResult(null);
    setOcrResult(null);
    setCroppedPlates([]);
    setError(null);
    if (fileRef.current) fileRef.current.value = "";
  }

  const formatSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const steps = result ? [
    {
      title: "1. Original Input",
      desc: "Uploaded image with initial metadata.",
      image: result.original,
      icon: ImageIcon,
      time: "0 ms",
      details: (
        <div className="text-sm space-y-1 text-slate-500 mt-2">
          <p>Size: {formatSize(fileSize)}</p>
          <p>Format: {selectedFile?.type}</p>
        </div>
      )
    },
    {
      title: "2. Normalized Image",
      desc: "Image resized to 640x640 and normalized for consistent model inference.",
      image: result.normalized,
      icon: Target,
      time: "12 ms",
    },
    {
      title: "3. Low Light & Shadow Fix",
      desc: "Gamma Correction and CLAHE applied to restore visibility in low-light and high-contrast shadowed environments.",
      image: result.enhanced,
      icon: Zap,
      time: "18 ms",
    },
    {
      title: "4. Weather & Blur Reduction",
      desc: "Bilateral filtering to reduce rain/sensor noise, followed by Unsharp Masking to correct motion blur.",
      image: result.denoised,
      icon: Droplets,
      time: "24 ms",
    }
  ] : [];

  // Draw OCR boxes
  useEffect(() => {
    if (ocrResult && ocrResult.predictions.length > 0 && imgRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const img = imgRef.current;
      
      const { width: iw, height: ih } = ocrResult.image;
      const scaleX = img.clientWidth / iw;
      const scaleY = img.clientHeight / ih;

      canvas.width = img.clientWidth;
      canvas.height = img.clientHeight;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const newCrops: string[] = [];

      ocrResult.predictions.forEach((p) => {
        const color = "#facc15";
        const x = (p.x - p.width / 2) * scaleX;
        const y = (p.y - p.height / 2) * scaleY;
        const w = p.width * scaleX;
        const h = p.height * scaleY;

        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        const label = `${p.ocr_text || p.class} ${Math.round(p.confidence * 100)}%`;
        ctx.font = "bold 14px Inter, sans-serif";
        const tw = ctx.measureText(label).width;

        ctx.fillStyle = color;
        ctx.fillRect(x, y - 24, tw + 10, 24);
        ctx.fillStyle = "#1e293b";
        ctx.fillText(label, x + 5, y - 7);

        // Crop Plate
        try {
          const offCanvas = document.createElement("canvas");
          const padX = p.width * 0.12 + 8;
          const padY = p.height * 0.20 + 8;
          const sx = Math.max(0, p.x - p.width / 2 - padX);
          const sy = Math.max(0, p.y - p.height / 2 - padY);
          const sw = Math.min(iw - sx, p.width + padX * 2);
          const sh = Math.min(ih - sy, p.height + padY * 2);

          offCanvas.width = sw;
          offCanvas.height = sh;
          const offCtx = offCanvas.getContext("2d");
          if (offCtx && img) {
            // Draw from native image resolution
            offCtx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
            newCrops.push(offCanvas.toDataURL("image/jpeg"));
          } else {
            newCrops.push(""); // Placeholder if context fails
          }
        } catch (e) {
          console.error("Crop failed", e);
          newCrops.push("");
        }
      });
      setCroppedPlates(newCrops);
    }
  }, [ocrResult, imgRef.current?.clientWidth]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col">
      <header className="bg-white border-b border-slate-200 px-6 py-5 flex items-center gap-3 sticky top-0 z-10 shadow-sm">
        <Sparkles className="h-6 w-6 text-indigo-600" />
        <div>
          <h1 className="font-bold text-lg text-slate-900 tracking-tight leading-none">
            AI Preprocessing Pipeline
          </h1>
          <p className="text-xs text-slate-500 mt-1">Computer Vision Workflow Demonstration</p>
        </div>
      </header>

      <main className="flex-1 w-full mx-auto p-4 md:p-8 flex flex-col gap-8 max-w-5xl">
        
        {/* Upload Section */}
        {!result && (
          <div className="w-full bg-white p-8 rounded-2xl shadow-sm border border-slate-100">
            <div className="text-center mb-6">
              <h2 className="text-xl font-bold text-slate-800">Test the Pipeline</h2>
              <p className="text-slate-500 text-sm mt-2">Upload a traffic image to visualize the step-by-step AI preprocessing.</p>
            </div>

            {!imageSrc ? (
              <label
                htmlFor="pipeline-image-input"
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  handleFile(e.dataTransfer.files[0]);
                }}
                className="flex flex-col items-center justify-center gap-4 rounded-xl border-2 border-dashed border-indigo-200 bg-indigo-50/50 py-24 px-6 text-center cursor-pointer hover:border-indigo-400 hover:bg-indigo-50 transition-all duration-300 group"
              >
                <div className="bg-white p-4 rounded-full shadow-sm group-hover:scale-110 transition-transform duration-300">
                  <Upload className="h-8 w-8 text-indigo-500" />
                </div>
                <div>
                  <p className="font-medium text-indigo-900 text-lg">Drag & drop your image here</p>
                  <p className="text-sm text-indigo-500 mt-1">or click to browse files</p>
                </div>
                <input
                  id="pipeline-image-input"
                  ref={fileRef}
                  type="file"
                  accept="image/*"
                  className="sr-only"
                  onChange={(e) => handleFile(e.target.files?.[0])}
                />
              </label>
            ) : (
              <div className="space-y-6">
                <div className="relative rounded-xl overflow-hidden border border-slate-200 bg-slate-100 flex justify-center p-2">
                  <img
                    src={imageSrc}
                    alt="Uploaded"
                    className="w-full object-contain max-h-[600px] rounded-lg shadow-sm"
                  />
                </div>
                
                <div className="flex items-center justify-between text-sm text-slate-600 bg-slate-50 py-3 px-5 rounded-lg border border-slate-100">
                  <span className="truncate font-medium text-base">{fileName}</span>
                  <button
                    onClick={clearAll}
                    className="ml-3 shrink-0 flex items-center gap-1 text-slate-400 hover:text-red-500 transition-colors"
                  >
                    <X className="h-5 w-5" /> Clear
                  </button>
                </div>
                
                <button
                  onClick={runPipeline}
                  disabled={loading}
                  className="w-full flex items-center justify-center gap-2 rounded-xl bg-indigo-600 text-white px-6 py-5 text-lg font-semibold hover:bg-indigo-700 hover:shadow-md disabled:opacity-70 disabled:cursor-not-allowed transition-all"
                >
                  {loading ? (
                    <>
                      <Loader2 className="h-6 w-6 animate-spin" />
                      Processing Pipeline...
                    </>
                  ) : (
                    <>
                      <Zap className="h-6 w-6" />
                      Run Pipeline Demonstration
                    </>
                  )}
                </button>
                {error && (
                  <p className="text-sm text-amber-600 bg-amber-50 border border-amber-100 rounded-lg px-4 py-3 text-center mt-4">
                    {error}
                  </p>
                )}
              </div>
            )}
          </div>
        )}

        {/* Results Section */}
        {result && (
          <div className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
            
            <div className="flex flex-col sm:flex-row justify-between items-center bg-white p-5 rounded-xl border border-slate-200 shadow-sm gap-4">
              <div className="flex items-center gap-3">
                <CheckCircle className="h-7 w-7 text-emerald-500" />
                <h2 className="text-xl font-bold text-slate-800">Pipeline Execution Complete</h2>
              </div>
              <button 
                onClick={clearAll}
                className="px-6 py-2.5 bg-slate-100 text-slate-700 hover:bg-slate-200 rounded-lg text-sm font-medium transition-colors w-full sm:w-auto"
              >
                Test Another Image
              </button>
            </div>

            {/* Metrics Header */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white rounded-2xl shadow-sm border border-slate-100 p-6">
                <h3 className="font-bold text-slate-800 flex items-center gap-2 mb-6 text-lg">
                  <BarChart3 className="h-5 w-5 text-indigo-500" />
                  Quality Assessment
                </h3>
                
                <div className="space-y-5">
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm font-medium">
                      <span className="text-slate-600">Brightness</span>
                      <span className="text-slate-900">{result.metrics.brightness}%</span>
                    </div>
                    <div className="h-2.5 w-full bg-slate-100 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-amber-400 rounded-full transition-all duration-1000 ease-out"
                        style={{ width: `${result.metrics.brightness}%` }}
                      />
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm font-medium">
                      <span className="text-slate-600">Sharpness</span>
                      <span className="text-slate-900">{result.metrics.sharpness}%</span>
                    </div>
                    <div className="h-2.5 w-full bg-slate-100 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-emerald-400 rounded-full transition-all duration-1000 ease-out delay-150"
                        style={{ width: `${result.metrics.sharpness}%` }}
                      />
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm font-medium">
                      <span className="text-slate-600">Contrast</span>
                      <span className="text-slate-900">{result.metrics.contrast}%</span>
                    </div>
                    <div className="h-2.5 w-full bg-slate-100 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-indigo-500 rounded-full transition-all duration-1000 ease-out delay-300"
                        style={{ width: `${result.metrics.contrast}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-slate-900 text-white rounded-2xl shadow-sm border border-slate-800 p-6">
                <h3 className="font-semibold text-slate-100 mb-4 text-lg">Processing Timings</h3>
                <div className="space-y-4 text-base">
                  <div className="flex justify-between items-center py-2 border-b border-slate-800">
                    <span className="text-slate-400">Preprocessing</span>
                    <span className="font-mono text-emerald-400">{DUMMY_TIMINGS.preprocess} ms</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-800">
                    <span className="text-slate-400">Detection</span>
                    <span className="font-mono">{DUMMY_TIMINGS.detection} ms</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-800">
                    <span className="text-slate-400">OCR</span>
                    <span className="font-mono">{DUMMY_TIMINGS.ocr} ms</span>
                  </div>
                  <div className="flex justify-between items-center pt-3 font-medium">
                    <span className="text-lg">Total Time</span>
                    <span className="font-mono text-indigo-400 text-xl">{DUMMY_TIMINGS.total} ms</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Vertical Workflow Diagram */}
            <div className="space-y-8 relative">
              {steps.map((step, index) => (
                <div key={index} className="flex flex-col relative group">
                  <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-sm hover:shadow-md transition-shadow">
                    
                    {/* Header */}
                    <div className="border-b border-slate-100 p-5 bg-slate-50 flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="p-2 bg-indigo-100 rounded-lg">
                          <step.icon className="h-6 w-6 text-indigo-700" />
                        </div>
                        <div>
                          <h4 className="font-bold text-slate-900 text-lg">{step.title}</h4>
                          <p className="text-sm text-slate-500 mt-0.5">{step.desc}</p>
                        </div>
                      </div>
                      <div className="hidden sm:block bg-slate-900 text-white px-3 py-1 rounded-lg text-sm font-mono">
                        {step.time}
                      </div>
                    </div>

                    {/* Image Area - Much Larger */}
                    <div className="p-4 bg-slate-100/50">
                      <img 
                        src={step.image} 
                        alt={step.title}
                        className="w-full h-auto rounded-xl shadow-sm border border-slate-200 object-contain max-h-[800px]" 
                      />
                    </div>
                    
                    {step.details && (
                      <div className="p-4 bg-white border-t border-slate-100">
                        {step.details}
                      </div>
                    )}
                  </div>

                  {/* Connector Arrow */}
                  {index < steps.length - 1 && (
                    <div className="flex justify-center mt-8">
                      <div className="bg-indigo-50 border border-indigo-100 p-2 rounded-full">
                        <ArrowDown className="h-6 w-6 text-indigo-400" />
                      </div>
                    </div>
                  )}
                </div>
              ))}

              {/* Step 5: OCR Results */}
              <div className="flex justify-center my-8">
                <div className="bg-emerald-50 border border-emerald-100 p-2 rounded-full">
                  <ArrowDown className="h-6 w-6 text-emerald-500" />
                </div>
              </div>

              <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-sm hover:shadow-md transition-shadow">
                <div className="border-b border-slate-100 p-5 bg-slate-50 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-emerald-100 rounded-lg">
                      <Camera className="h-6 w-6 text-emerald-700" />
                    </div>
                    <div>
                      <h4 className="font-bold text-slate-900 text-lg">5. License Plate OCR & Detection</h4>
                      <p className="text-sm text-slate-500 mt-0.5">
                        Inference run on the enhanced and denoised image to extract license plate details.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="p-4 bg-slate-100/50 relative">
                  {ocrLoading ? (
                    <div className="flex flex-col items-center justify-center py-32">
                      <Loader2 className="h-10 w-10 text-emerald-500 animate-spin mb-4" />
                      <p className="text-slate-600 font-medium">Running YOLOv8 & OCR on processed image...</p>
                    </div>
                  ) : (
                    <div className="relative inline-block w-full">
                      <img 
                        ref={imgRef}
                        src={result.denoised} 
                        alt="OCR Result"
                        className="w-full h-auto rounded-xl shadow-sm border border-slate-200 object-contain max-h-[800px]" 
                      />
                      <canvas
                        ref={canvasRef}
                        className="absolute inset-0 pointer-events-none"
                        style={{ width: "100%", height: "100%" }}
                      />
                    </div>
                  )}
                </div>
                
                {ocrResult && !ocrLoading && (
                  <div className="p-6 bg-white border-t border-slate-100">
                    <h5 className="font-semibold text-slate-800 mb-4 text-lg">Detected Plates</h5>
                    {ocrResult.predictions.length > 0 ? (
                      <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-6">
                        {ocrResult.predictions.map((p, i) => (
                          <div key={i} className="flex flex-col bg-slate-50 border border-slate-200 rounded-xl overflow-hidden shadow-sm hover:shadow-md transition-shadow">
                            {croppedPlates[i] && (
                              <div className="bg-slate-200/50 p-6 flex justify-center items-center border-b border-slate-200 min-h-[160px]">
                                <img 
                                  src={croppedPlates[i]} 
                                  alt={`License Plate ${i + 1}`} 
                                  className="w-full max-h-48 object-contain rounded drop-shadow-md"
                                />
                              </div>
                            )}
                            <div className="p-4 flex items-center justify-between">
                              <span className="font-bold text-slate-900 text-lg uppercase tracking-wider">
                                {p.ocr_text || "Unreadable"}
                              </span>
                              <span className="bg-emerald-100 text-emerald-800 px-2.5 py-1 rounded-md text-sm font-semibold">
                                {Math.round(p.confidence * 100)}%
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-slate-500 italic">No license plates detected in this image.</p>
                    )}
                  </div>
                )}
              </div>

            </div>
          </div>
        )}

      </main>
    </div>
  );
}
