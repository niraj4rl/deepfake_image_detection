import { useEffect, useMemo, useState } from "react";
import { Activity, FileImage, History, Info, Moon, ScanLine, Sun, Upload, X } from "lucide-react";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
type View = "analyze" | "batch" | "history" | "about";
type Result = { id?: string; filename?: string; created_at?: string; verdict: "real" | "fake" | "inconclusive"; fake_probability: number | null; confidence: number | null; processing_ms?: number; warnings?: string[]; thumbnail_url?: string; face_detected: boolean; model_name?: string };

async function jsonRequest(path: string, init?: RequestInit) { const response = await fetch(`${API}${path}`, init); if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail ?? `Request failed (${response.status})`); return response.json(); }

function Verdict({ result }: { result: Result }) { return <span className={`verdict ${result.verdict}`}>{result.verdict.toUpperCase()}</span>; }

function App() {
  const [view, setView] = useState<View>("analyze");
  const [dark, setDark] = useState(localStorage.getItem("forensiscan-theme") !== "light");
  const [health, setHealth] = useState<{ model_loaded: boolean } | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState<Result[]>([]);

  useEffect(() => { document.documentElement.dataset.theme = dark ? "dark" : "light"; localStorage.setItem("forensiscan-theme", dark ? "dark" : "light"); }, [dark]);
  useEffect(() => { jsonRequest("/api/health").then(setHealth).catch(() => setHealth(null)); }, []);
  useEffect(() => { if (view === "history") jsonRequest("/api/history").then((data) => setHistory(data.items)).catch(() => setHistory([])); }, [view]);

  async function analyze(file: File) {
    setBusy(true); setError(""); setResult(null);
    try { const form = new FormData(); form.append("file", file); setResult(await jsonRequest("/api/analyze", { method: "POST", body: form })); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Analysis failed"); }
    finally { setBusy(false); }
  }

  async function analyzeBatch(files: FileList | null) { if (!files) return; for (const file of Array.from(files).slice(0, 20)) { try { const form = new FormData(); form.append("file", file); const item = await jsonRequest("/api/analyze", { method: "POST", body: form }); setHistory((items) => [item, ...items]); } catch { /* individual batch failures stay local */ } } }
  const modelLabel = health?.model_loaded ? "MODEL READY" : "MODEL NOT TRAINED";
  return <div className="app-shell">
    <header className="topbar"><button className="brand" onClick={() => setView("analyze")}><span className="brand-mark"><ScanLine size={20} /></span><span>Forensi<span>Scan</span></span></button><nav>{([['analyze', 'Analyze'], ['batch', 'Batch'], ['history', 'History'], ['about', 'About']] as [View, string][]).map(([key, label]) => <button className={view === key ? "active" : ""} onClick={() => setView(key)} key={key}>{label}</button>)}</nav><div className="top-actions"><span className={`status ${health?.model_loaded ? "ready" : "offline"}`}><Activity size={14} /> {modelLabel}</span><button className="icon-button" title="Toggle theme" onClick={() => setDark((value) => !value)}>{dark ? <Sun size={17} /> : <Moon size={17} />}</button></div></header>
    <main>{view === "analyze" && <AnalyzeView busy={busy} error={error} result={result} onAnalyze={analyze} />}{view === "batch" && <BatchView onFiles={analyzeBatch} />}{view === "history" && <HistoryView history={history} />}{view === "about" && <AboutView />}</main>
    <footer><span>ForensiScan / classical DIP research tool</span><span>Local processing only · probabilistic indicator, not proof</span></footer>
  </div>;
}

function AnalyzeView({ busy, error, result, onAnalyze }: { busy: boolean; error: string; result: Result | null; onAnalyze: (file: File) => void }) {
  const [fileName, setFileName] = useState("");
  function choose(file: File | undefined) { if (!file) return; if (!file.type.startsWith("image/")) return; setFileName(file.name); onAnalyze(file); }
  return <section className="content"><div className="eyebrow">IMAGE FORENSICS / 01</div><div className="hero-row"><div><h1>Read the artifacts.<br /><em>Question the image.</em></h1><p className="lede">A classical signal lab for face-swap analysis. Frequency, texture, compression, and edge evidence stay visible in the process.</p></div><div className="method-note"><span>METHOD</span><strong>FFT · ELA · LBP · GLCM</strong><small>CPU pipeline / FF++ trained</small></div></div><label className={`drop-zone ${busy ? "scanning" : ""}`}><input type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => choose(event.target.files?.[0])} /><Upload size={26} /><strong>{busy ? "Extracting signal..." : fileName || "Drop a face image here"}</strong><span>{busy ? "The local model is working" : "JPG, PNG or WEBP · up to 10 MB"}</span></label>{error && <div className="error"><X size={17} />{error}</div>}{result && <ResultCard result={result} />}</section>;
}

function ResultCard({ result }: { result: Result }) { const percent = result.fake_probability === null ? 0 : Math.round(result.fake_probability * 100); return <article className="result-card"><div className="result-head"><div><div className="eyebrow">ANALYSIS RESULT</div><h2><Verdict result={result} /></h2></div><span className="mono">{result.processing_ms ?? "--"} ms</span></div><div className="gauge"><div className="gauge-fill" style={{ width: `${percent}%` }} /><span style={{ left: `${percent}%` }} /></div><div className="probability"><span>FAKE PROBABILITY</span><strong>{result.fake_probability === null ? "—" : `${percent}%`}</strong></div><div className="result-meta"><span>Face detected <b>{result.face_detected ? "Yes" : "No"}</b></span><span>Confidence <b>{result.confidence === null ? "—" : `${Math.round(result.confidence * 100)}%`}</b></span><span>Model <b>{result.model_name ?? "Unavailable"}</b></span></div>{result.warnings?.map((warning) => <p className="warning" key={warning}>{warning}</p>)}</article>; }

function BatchView({ onFiles }: { onFiles: (files: FileList | null) => void }) { return <section className="content"><div className="eyebrow">BATCH ANALYSIS / 02</div><h1>Process a small set.</h1><p className="lede">Queue up to 20 local images for individual analysis. Each file is handled independently.</p><label className="batch-picker"><FileImage size={24} /><strong>Select up to 20 images</strong><span>Results appear in History as they complete</span><input type="file" multiple accept="image/jpeg,image/png,image/webp" onChange={(event) => onFiles(event.target.files)} /></label></section>; }

function HistoryView({ history }: { history: Result[] }) { return <section className="content"><div className="eyebrow">LOCAL HISTORY / 03</div><div className="section-head"><div><h1>Past examinations.</h1><p className="lede">Stored locally so you can compare the trail of your work.</p></div><a className="button" href={`${API}/api/history/export.csv`}>Export CSV</a></div>{history.length === 0 ? <div className="empty"><History size={30} /><strong>No analyses yet</strong><span>Run an image through the Analyze view to start a local record.</span></div> : <div className="history-list">{history.map((item) => <div className="history-row" key={item.id}><div className="history-thumb">{item.thumbnail_url && <img src={`${API}${item.thumbnail_url}`} />}</div><div><strong>{item.filename}</strong><small>{item.created_at ? new Date(item.created_at).toLocaleString() : ""}</small></div><Verdict result={item} /><b className="mono">{item.fake_probability === null ? "—" : `${Math.round(item.fake_probability * 100)}%`}</b></div>)}</div>}</section>; }

function AboutView() { return <section className="content about"><div className="eyebrow">LAB NOTES / 04</div><h1>Classical evidence,<br /><em>careful conclusions.</em></h1><div className="tech-grid">{[["FFT / DCT", "Resampling and blending can disturb high-frequency energy."], ["ELA / residuals", "JPEG error maps and fixed high-pass filters expose inconsistent texture."], ["LBP / GLCM / Gabor", "Local texture descriptors compare skin-like structure across regions."], ["Color / edges", "Chrominance shifts and boundary density add context to the signal."]].map(([title, text]) => <article key={title}><span>0{Math.random() > 0 ? 1 : 1}</span><h3>{title}</h3><p>{text}</p></article>)}</div><div className="disclaimer"><Info size={20} /><div><strong>Use this as an indicator, never a verdict.</strong><p>ForensiScan is trained on FaceForensics++ face-swap data and will not reliably generalize to every generator or compression history. Uploaded images stay local. Follow the dataset's research and education terms.</p></div></div></section>; }

export default App;