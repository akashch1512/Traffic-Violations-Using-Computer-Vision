import { useMemo, useState } from "react";
import HelmetDetection from "./components/HelmetDetection";
import TripleRiderDetection from "./components/TripleRiderDetection";
import LicensePlateDetection from "./components/LicensePlateDetection";
import {
  AlertTriangle,
  BarChart3,
  Bike,
  Camera,
  Car,
  CheckCircle2,
  FileVideo,
  Gauge,
  Play,
  ShieldCheck,
  Upload,
  UserRound,
  Video,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Button } from "./components/ui/button";
import { Card, CardContent, CardHeader } from "./components/ui/card";
import { cn } from "./lib/utils";

const violationTypes = [
  "Helmet violation",
  "Seatbelt violation",
  "Triple riding",
  "Wrong-side driving",
  "Stop-line violation",
  "Red-light violation",
  "Illegal parking",
];

const detections = [
  { label: "Motorbike", count: 3, icon: Bike },
  { label: "Cars", count: 2, icon: Car },
  { label: "People", count: 7, icon: UserRound },
];

const fallbackViolations = [
  {
    type: "Helmet violation",
    confidence: 94,
    timestamp: "00:04",
    plate: "KA 05 MX 4412",
  },
  {
    type: "Triple riding",
    confidence: 89,
    timestamp: "00:06",
    plate: "KA 05 MX 4412",
  },
  {
    type: "Stop-line violation",
    confidence: 82,
    timestamp: "00:11",
    plate: "TN 09 BX 2381",
  },
];

type Detection = {
  label: string;
  count: number;
};

type Violation = {
  type: string;
  confidence: number;
  timestamp: string;
  plate: string;
};

type AnalysisResult = {
  frames: {
    count: number;
    folder_path: string;
  };
  detections: Detection[];
  violations: Violation[];
  summary: {
    vehicles_detected: number;
    violations_found: number;
    plates_detected: number;
    average_confidence: number;
  };
};

function App() {
  const [tab, setTab] = useState<"traffic" | "helmet" | "triple" | "license">("traffic");
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzed, setIsAnalyzed] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [fileName, setFileName] = useState("No video selected");

  const activeViolations = analysis?.violations ?? fallbackViolations;
  const activeDetections =
    analysis?.detections.map((item) => ({
      ...item,
      icon:
        item.label.toLowerCase().includes("bike") ||
        item.label.toLowerCase().includes("motorbike")
          ? Bike
          : item.label.toLowerCase().includes("people")
            ? UserRound
            : Car,
    })) ?? detections;

  const activeChartData = useMemo(() => {
    const counts = activeViolations.reduce<Record<string, number>>(
      (total, item) => {
        const label = item.type.split(" ")[0];
        total[label] = (total[label] ?? 0) + 1;
        return total;
      },
      {}
    );

    return Object.entries(counts).map(([type, count]) => ({ type, count }));
  }, [activeViolations]);

  const activeStatusData = useMemo(
    () => [
      {
        name: "Violations",
        value: analysis?.summary.violations_found ?? activeViolations.length,
        color: "#dc2626",
      },
      {
        name: "Frames",
        value: analysis?.frames.count ?? 1,
        color: "#16a34a",
      },
    ],
    [activeViolations.length, analysis]
  );

  const averageConfidence = useMemo(
    () =>
      Math.round(
        activeViolations.reduce((total, item) => total + item.confidence, 0) /
          activeViolations.length
      ),
    [activeViolations]
  );

  function handleFile(file?: File) {
    if (!file) return;
    if (!file.type.startsWith("video/")) {
      setErrorMessage("Select a video file.");
      return;
    }
    setFileName(file.name);
    if (selectedVideo) URL.revokeObjectURL(selectedVideo);
    setSelectedFile(file);
    setSelectedVideo(URL.createObjectURL(file));
    setIsAnalyzed(false);
    setAnalysis(null);
    setErrorMessage(null);
  }

  async function analyzeVideo() {
    if (!selectedFile) {
      setErrorMessage("Upload a traffic video before analysis.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    setIsAnalyzing(true);
    setErrorMessage(null);

    try {
      const response = await fetch("/api/violations/analyze-video", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail ?? "Video analysis failed.");
      }

      const result = (await response.json()) as AnalysisResult;
      setAnalysis(result);
      setIsAnalyzed(true);
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Video analysis failed."
      );
      setIsAnalyzed(false);
    } finally {
      setIsAnalyzing(false);
    }
  }

  if (tab === "helmet") {
    return (
      <div>
        <nav className="border-b border-slate-100 bg-white px-6 flex gap-1 py-2">
          <button
            onClick={() => setTab("traffic")}
            className="px-4 py-1.5 rounded-md text-sm text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition-colors"
          >
            Traffic Analysis
          </button>
          <button
            onClick={() => setTab("helmet")}
            className="px-4 py-1.5 rounded-md text-sm font-medium bg-slate-900 text-white"
          >
            Helmet Detection
          </button>
          <button
            onClick={() => setTab("triple")}
            className="px-4 py-1.5 rounded-md text-sm text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition-colors"
          >
            Triple Rider & Phone
          </button>
          <button
            onClick={() => setTab("license")}
            className="px-4 py-1.5 rounded-md text-sm text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition-colors"
          >
            License Plate OCR
          </button>
        </nav>
        <HelmetDetection />
      </div>
    );
  }

  if (tab === "triple") {
    return (
      <div>
        <nav className="border-b border-slate-100 bg-white px-6 flex gap-1 py-2">
          <button
            onClick={() => setTab("traffic")}
            className="px-4 py-1.5 rounded-md text-sm text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition-colors"
          >
            Traffic Analysis
          </button>
          <button
            onClick={() => setTab("helmet")}
            className="px-4 py-1.5 rounded-md text-sm text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition-colors"
          >
            Helmet Detection
          </button>
          <button
            onClick={() => setTab("triple")}
            className="px-4 py-1.5 rounded-md text-sm font-medium bg-slate-900 text-white"
          >
            Triple Rider & Phone
          </button>
          <button
            onClick={() => setTab("license")}
            className="px-4 py-1.5 rounded-md text-sm text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition-colors"
          >
            License Plate OCR
          </button>
        </nav>
        <TripleRiderDetection />
      </div>
    );
  }

  if (tab === "license") {
    return (
      <div>
        <nav className="border-b border-slate-100 bg-white px-6 flex gap-1 py-2">
          <button
            onClick={() => setTab("traffic")}
            className="px-4 py-1.5 rounded-md text-sm text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition-colors"
          >
            Traffic Analysis
          </button>
          <button
            onClick={() => setTab("helmet")}
            className="px-4 py-1.5 rounded-md text-sm text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition-colors"
          >
            Helmet Detection
          </button>
          <button
            onClick={() => setTab("triple")}
            className="px-4 py-1.5 rounded-md text-sm text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition-colors"
          >
            Triple Rider & Phone
          </button>
          <button
            onClick={() => setTab("license")}
            className="px-4 py-1.5 rounded-md text-sm font-medium bg-slate-900 text-white"
          >
            License Plate OCR
          </button>
        </nav>
        <LicensePlateDetection />
      </div>
    );
  }

  return (
    <main className="min-h-screen bg-background text-foreground">
      <nav className="border-b border-slate-100 bg-white px-6 flex gap-1 py-2">
        <button
          onClick={() => setTab("traffic")}
          className="px-4 py-1.5 rounded-md text-sm font-medium bg-slate-900 text-white"
        >
          Traffic Analysis
        </button>
        <button
          onClick={() => setTab("helmet")}
          className="px-4 py-1.5 rounded-md text-sm text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition-colors"
        >
          Helmet Detection
        </button>
        <button
          onClick={() => setTab("triple")}
          className="px-4 py-1.5 rounded-md text-sm text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition-colors"
        >
          Triple Rider & Phone
        </button>
        <button
          onClick={() => setTab("license")}
          className="px-4 py-1.5 rounded-md text-sm text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition-colors"
        >
          License Plate OCR
        </button>
      </nav>
      <section className="border-b border-border bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-10 px-5 py-8 md:px-8 lg:flex-row lg:items-center lg:py-12">
          <div className="flex-1 space-y-5">
            <div className="inline-flex items-center gap-2 rounded-full border border-blue-100 bg-blue-50 px-3 py-1 text-sm font-medium text-blue-700">
              <ShieldCheck className="h-4 w-4" />
              Computer vision proof of concept
            </div>
            <div className="space-y-3">
              <h1 className="max-w-3xl text-4xl font-semibold leading-tight text-slate-950 md:text-5xl">
                Traffic Violation AI
              </h1>
              <p className="max-w-2xl text-base leading-7 text-slate-600 md:text-lg">
                Upload a traffic video, send it to the backend API, extract
                frames, and review detected vehicles, violations, plates, and
                evidence in one compact workflow.
              </p>
            </div>
            <div className="flex flex-col gap-3 sm:flex-row">
              <Button
                size="lg"
                onClick={() =>
                  document.getElementById("traffic-video-input")?.click()
                }
              >
                <Upload className="h-5 w-5" />
                Upload video
              </Button>
              <Button
                size="lg"
                variant="outline"
                onClick={analyzeVideo}
                disabled={isAnalyzing || !selectedFile}
              >
                <Play className="h-5 w-5" />
                {isAnalyzing ? "Analyzing..." : "Analyze video"}
              </Button>
            </div>
          </div>
          <div className="w-full max-w-xl rounded-lg border border-border bg-slate-50 p-3 shadow-soft">
            <EvidencePreview video={selectedVideo} analyzed={isAnalyzed} />
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-6 px-5 py-8 md:px-8 lg:grid-cols-[0.9fr_1.1fr]">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between gap-4">
              <div>
                <h2 className="text-xl font-semibold text-slate-950">
                  Upload & analysis
                </h2>
                <p className="mt-1 text-sm text-slate-500">
                  Drag a video here or choose one to start.
                </p>
              </div>
              <Video className="h-5 w-5 text-blue-600" />
            </div>
          </CardHeader>
          <CardContent>
            <label
              htmlFor="traffic-video-input"
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                handleFile(event.dataTransfer.files[0]);
              }}
              className="flex min-h-56 cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-blue-200 bg-blue-50/60 px-5 py-8 text-center transition hover:bg-blue-50"
            >
              <Upload className="mb-4 h-10 w-10 text-blue-600" />
              <span className="text-base font-semibold text-slate-900">
                Drop traffic video here
              </span>
              <span className="mt-2 text-sm text-slate-500">
                MP4, WEBM, MOV, or AVI. Backend extracts frames before analysis.
              </span>
              <input
                id="traffic-video-input"
                type="file"
                accept="video/*"
                className="sr-only"
                onChange={(event) => handleFile(event.target.files?.[0])}
              />
            </label>
            <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex min-w-0 items-center gap-2 text-sm text-slate-600">
                <FileVideo className="h-4 w-4 shrink-0 text-slate-400" />
                <span className="truncate">{fileName}</span>
              </div>
              <Button
                onClick={analyzeVideo}
                disabled={isAnalyzing || !selectedFile}
              >
                <Play className="h-4 w-4" />
                {isAnalyzing ? "Analyzing..." : "Analyze video"}
              </Button>
            </div>
            {errorMessage && (
              <p className="mt-3 text-sm font-medium text-red-600">
                {errorMessage}
              </p>
            )}
            {analysis && (
              <p className="mt-3 text-sm text-slate-500">
                Extracted {analysis.frames.count} frames to{" "}
                {analysis.frames.folder_path}.
              </p>
            )}
          </CardContent>
        </Card>

        <div className="grid gap-4 sm:grid-cols-2">
          <SummaryCard
            title="Vehicles detected"
            value={
              isAnalyzed ? String(analysis?.summary.vehicles_detected ?? 0) : "0"
            }
            icon={Car}
            helper="Cars and motorbikes"
            tone="blue"
          />
          <SummaryCard
            title="Violations found"
            value={
              isAnalyzed ? String(analysis?.summary.violations_found ?? 0) : "0"
            }
            icon={AlertTriangle}
            helper={isAnalyzed ? "Needs review" : "Run analysis"}
            tone="red"
          />
          <SummaryCard
            title="Plate detected"
            value={
              isAnalyzed ? String(analysis?.summary.plates_detected ?? 0) : "0"
            }
            icon={Camera}
            helper={isAnalyzed ? "OCR matched" : "Awaiting frame"}
            tone="green"
          />
          <SummaryCard
            title="Frames"
            value={isAnalyzed ? String(analysis?.frames.count ?? 0) : "0"}
            icon={Gauge}
            helper={
              isAnalyzed ? `${averageConfidence}% avg confidence` : "Extracted"
            }
            tone="blue"
          />
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-6 px-5 pb-10 md:px-8 lg:grid-cols-[1.15fr_0.85fr]">
        <Card>
          <CardHeader>
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-xl font-semibold text-slate-950">
                  Results
                </h2>
                <p className="mt-1 text-sm text-slate-500">
                  Video evidence, detections, and supported violation checks.
                </p>
              </div>
              <span
                className={cn(
                  "inline-flex w-fit items-center gap-2 rounded-full px-3 py-1 text-sm font-medium",
                  isAnalyzed
                    ? "bg-red-50 text-red-700"
                    : "bg-emerald-50 text-emerald-700"
                )}
              >
                {isAnalyzed ? (
                  <AlertTriangle className="h-4 w-4" />
                ) : (
                  <CheckCircle2 className="h-4 w-4" />
                )}
                {isAnalyzed ? "Violations detected" : "Ready to analyze"}
              </span>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid gap-5 lg:grid-cols-[1.2fr_0.8fr]">
              <EvidencePreview video={selectedVideo} analyzed={isAnalyzed} />
              <div className="space-y-5">
                <div>
                  <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                    Detected objects
                  </h3>
                  <div className="mt-3 grid gap-3">
                    {activeDetections.map((item) => (
                      <div
                        key={item.label}
                        className="flex items-center justify-between rounded-md border border-border bg-slate-50 px-3 py-3"
                      >
                        <div className="flex items-center gap-3">
                          <item.icon className="h-5 w-5 text-blue-600" />
                          <span className="font-medium text-slate-800">
                            {item.label}
                          </span>
                        </div>
                        <span className="text-lg font-semibold text-slate-950">
                          {isAnalyzed ? item.count : 0}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                    Supported checks
                  </h3>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {violationTypes.map((type) => (
                      <span
                        key={type}
                        className="rounded-full border border-slate-200 bg-white px-3 py-1 text-sm text-slate-700"
                      >
                        {type}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <h2 className="text-xl font-semibold text-slate-950">
              Evidence panel
            </h2>
            <p className="mt-1 text-sm text-slate-500">
              Violation records returned by the backend analysis API.
            </p>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {(isAnalyzed ? activeViolations : []).map((item) => (
                <div
                  key={`${item.type}-${item.timestamp}`}
                  className="rounded-md border border-border p-3"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="font-semibold text-slate-950">
                        {item.type}
                      </p>
                      <p className="mt-1 text-sm text-slate-500">
                        {item.timestamp} - Plate {item.plate}
                      </p>
                    </div>
                    <span className="rounded-full bg-red-50 px-2.5 py-1 text-sm font-medium text-red-700">
                      {item.confidence}%
                    </span>
                  </div>
                </div>
              ))}
              {!isAnalyzed && (
                <div className="rounded-md border border-dashed border-slate-200 p-6 text-center text-sm text-slate-500">
                  Run analysis to populate violation evidence.
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="mx-auto grid max-w-7xl gap-6 px-5 pb-12 md:px-8 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-blue-600" />
              <h2 className="text-xl font-semibold text-slate-950">
                Violation mix
              </h2>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={isAnalyzed ? activeChartData : []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="type" tickLine={false} axisLine={false} />
                  <YAxis
                    allowDecimals={false}
                    tickLine={false}
                    axisLine={false}
                  />
                  <Tooltip />
                  <Bar dataKey="count" fill="#2563eb" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <h2 className="text-xl font-semibold text-slate-950">
              Frame status
            </h2>
          </CardHeader>
          <CardContent>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={
                      isAnalyzed
                        ? activeStatusData
                        : [{ name: "Pending", value: 1, color: "#94a3b8" }]
                    }
                    innerRadius={58}
                    outerRadius={86}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {(isAnalyzed
                      ? activeStatusData
                      : [{ name: "Pending", value: 1, color: "#94a3b8" }]
                    ).map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </section>
    </main>
  );
}

function SummaryCard({
  title,
  value,
  helper,
  icon: Icon,
  tone,
}: {
  title: string;
  value: string;
  helper: string;
  icon: typeof Car;
  tone: "blue" | "red" | "green";
}) {
  const colors = {
    blue: "bg-blue-50 text-blue-700",
    red: "bg-red-50 text-red-700",
    green: "bg-emerald-50 text-emerald-700",
  };

  return (
    <Card>
      <CardContent className="p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-sm font-medium text-slate-500">{title}</p>
            <p className="mt-2 text-3xl font-semibold text-slate-950">
              {value}
            </p>
            <p className="mt-1 text-sm text-slate-500">{helper}</p>
          </div>
          <div className={cn("rounded-md p-2.5", colors[tone])}>
            <Icon className="h-5 w-5" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function EvidencePreview({
  video,
  analyzed,
}: {
  video: string | null;
  analyzed: boolean;
}) {
  return (
    <div className="relative aspect-[16/10] overflow-hidden rounded-md bg-slate-200">
      {video ? (
        <video
          src={video}
          controls
          muted
          playsInline
          className="h-full w-full object-cover"
        />
      ) : (
        <div className="flex h-full w-full items-center justify-center bg-[linear-gradient(135deg,#dbeafe,#f8fafc_45%,#e2e8f0)]">
          <div className="grid w-4/5 gap-3">
            <div className="h-20 rounded-md bg-slate-300/80" />
            <div className="grid grid-cols-3 gap-3">
              <div className="h-16 rounded-md bg-blue-200/80" />
              <div className="h-16 rounded-md bg-slate-300/80" />
              <div className="h-16 rounded-md bg-red-200/80" />
            </div>
          </div>
        </div>
      )}
      {analyzed && (
        <svg className="absolute inset-0 h-full w-full" viewBox="0 0 100 62.5">
          <rect
            x="12"
            y="27"
            width="28"
            height="18"
            fill="none"
            stroke="#dc2626"
            strokeWidth="1.8"
            rx="1"
          />
          <rect
            x="54"
            y="24"
            width="24"
            height="17"
            fill="none"
            stroke="#2563eb"
            strokeWidth="1.8"
            rx="1"
          />
          <rect
            x="20"
            y="16"
            width="8"
            height="12"
            fill="none"
            stroke="#dc2626"
            strokeWidth="1.6"
            rx="1"
          />
          <rect
            x="63"
            y="42"
            width="14"
            height="4"
            fill="rgba(255,255,255,0.85)"
            stroke="#16a34a"
            strokeWidth="1"
            rx="1"
          />
          <line
            x1="6"
            y1="49"
            x2="94"
            y2="49"
            stroke="#ef4444"
            strokeWidth="1.5"
            strokeDasharray="3 2"
          />
          <text x="13" y="25" fill="#dc2626" fontSize="4" fontWeight="700">
            Helmet violation
          </text>
          <text x="64" y="45" fill="#166534" fontSize="3" fontWeight="700">
            KA05MX4412
          </text>
        </svg>
      )}
    </div>
  );
}

export default App;
