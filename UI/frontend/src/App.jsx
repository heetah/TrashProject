import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, download, videoUrl } from "./api";

const STATUS_LABELS = {
  queued: "等待中",
  running: "分析中",
  completed: "已完成",
  failed: "失敗",
};

const ATTRIBUTION_LABELS = {
  resolved: "已找到 AI 歸因候選",
  dustbin: "NULL／證據不足",
  pending: "歸因尚未完成",
  legacy: "舊版歸因候選",
};

const PLATE_LABELS = {
  recognized: "已辨識",
  pending: "辨識中",
  attempted_no_result: "辨識失敗",
  not_requested: "未執行",
  not_applicable: "不適用",
};

function formatPercent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${Math.round(number * 100)}%` : "—";
}

function formatTime(value) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds)) return "—";
  const minutes = Math.floor(seconds / 60);
  return `${String(minutes).padStart(2, "0")}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat("zh-TW", {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      }).format(date);
}

function JobStatus({ job }) {
  if (job.status === "completed") {
    const summary = job.analysis_summary;
    return (
      <span className={`status status-${job.review_status}`}>
        {job.review_status === "reviewed" ? "已審核" : `${job.reviewed_units}/${job.total_units} 已審`}
      </span>
    );
  }
  return <span className={`status status-${job.status}`}>{STATUS_LABELS[job.status] || job.status}</span>;
}

function IntakePanel({ config, onChanged }) {
  const [folderPath, setFolderPath] = useState("");
  const [files, setFiles] = useState([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  async function submitFiles(event) {
    event.preventDefault();
    if (!files.length) return;
    const form = event.currentTarget;
    const formData = new FormData();
    files.forEach((file) => formData.append("videos", file));
    setBusy(true);
    setMessage("");
    try {
      const result = await api("/api/jobs/upload", { method: "POST", body: formData });
      setFiles([]);
      form.reset();
      setMessage(`已加入 ${result.jobs.length} 支影片`);
      onChanged();
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function submitFolder(event) {
    event.preventDefault();
    if (!folderPath.trim()) return;
    setBusy(true);
    setMessage("");
    try {
      const result = await api("/api/jobs/folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder_path: folderPath.trim() }),
      });
      setMessage(`資料夾內 ${result.jobs.length} 支影片已加入佇列`);
      setFolderPath("");
      onChanged();
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function discover() {
    setBusy(true);
    setMessage("");
    try {
      const result = await api("/api/jobs/discover", { method: "POST" });
      setMessage(result.imported ? `找到 ${result.imported} 筆既有結果` : "既有結果已同步");
      onChanged();
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <details className="intake-panel">
      <summary>新增影片分析</summary>
      <div className="intake-content">
        <form onSubmit={submitFiles} className="intake-form">
          <label htmlFor="video-upload">上傳影片</label>
          <input
            id="video-upload"
            type="file"
            accept="video/*,.mkv"
            multiple
            onChange={(event) => setFiles(Array.from(event.target.files || []))}
          />
          <button type="submit" disabled={busy || !files.length}>上傳並排程</button>
        </form>

        <div className="divider"><span>或</span></div>

        <form onSubmit={submitFolder} className="intake-form">
          <label htmlFor="folder-path">伺服器資料夾路徑</label>
          <div className="inline-form">
            <input
              id="folder-path"
              type="text"
              value={folderPath}
              onChange={(event) => setFolderPath(event.target.value)}
              placeholder={config?.allowed_input_roots?.[0] || "/path/to/videos"}
            />
            <button type="submit" disabled={busy || !folderPath.trim()}>加入資料夾</button>
          </div>
          <small>只接受 .env 中 UI_ALLOWED_INPUT_ROOTS 允許的路徑。</small>
        </form>

        <button type="button" className="text-button" disabled={busy} onClick={discover}>
          重新掃描既有 analysis.json
        </button>
        {message && <p className="form-message" role="status">{message}</p>}
      </div>
    </details>
  );
}

function CaseList({ jobs, selectedId, onSelect }) {
  if (!jobs.length) {
    return (
      <div className="empty-list">
        <p>目前沒有符合條件的影片。</p>
        <span>可從上方新增影片或同步既有輸出。</span>
      </div>
    );
  }
  return (
    <div className="case-list" aria-label="影片案件清單">
      {jobs.map((job) => {
        const summary = job.analysis_summary || {};
        return (
          <button
            type="button"
            key={job.id}
            className={`case-row ${selectedId === job.id ? "selected" : ""}`}
            onClick={() => onSelect(job.id)}
          >
            <span className="case-row-top">
              <strong title={job.original_name}>{job.original_name}</strong>
              <JobStatus job={job} />
            </span>
            <span className="case-meta">
              {job.status === "completed"
                ? `${summary.litter_event_count || 0} 件垃圾事件 · ${formatPercent(summary.average_litter_confidence)}`
                : job.status_message}
            </span>
            <span className="case-date">{formatDate(job.created_at)}</span>
          </button>
        );
      })}
    </div>
  );
}

function Metric({ label, value, note }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
      {note && <small>{note}</small>}
    </div>
  );
}

function Confidence({ label, value }) {
  const number = Number(value);
  const percent = Number.isFinite(number) ? Math.max(0, Math.min(100, number * 100)) : 0;
  return (
    <div className="confidence-block">
      <div><span>{label}</span><strong>{formatPercent(value)}</strong></div>
      <div className="confidence-track" aria-hidden="true">
        <span style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}

function PencilIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 20h4.2L19 9.2 14.8 5 4 15.8V20Zm2-3.4 8.8-8.8 1.4 1.4L7.4 18H6v-1.4ZM17.6 2.2a1.5 1.5 0 0 1 2.1 0l2.1 2.1a1.5 1.5 0 0 1 0 2.1l-1.4 1.4-4.2-4.2 1.4-1.4Z" />
    </svg>
  );
}

function PlateEditor({ jobId, unit, onSaved }) {
  const aiPlate = unit.event?.plate || "";
  const correctedPlate = unit.plate_correction?.corrected_plate || "";
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(correctedPlate || aiPlate);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setEditing(false);
    setValue(correctedPlate || aiPlate);
    setError("");
  }, [unit.event_key, correctedPlate, aiPlate]);

  async function save(event) {
    event.preventDefault();
    if (!value.trim()) {
      setError("請輸入車牌");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const result = await api(
        `/api/jobs/${jobId}/events/${encodeURIComponent(unit.event_key)}/plate`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ corrected_plate: value }),
        },
      );
      onSaved(result.case);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  async function restoreAiPlate() {
    setBusy(true);
    setError("");
    try {
      const result = await api(
        `/api/jobs/${jobId}/events/${encodeURIComponent(unit.event_key)}/plate`,
        { method: "DELETE" },
      );
      onSaved(result.case);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  function cancelEditing() {
    setValue(correctedPlate || aiPlate);
    setError("");
    setEditing(false);
  }

  if (editing) {
    return (
      <form className="plate-editor" onSubmit={save}>
        <input
          aria-label="人工修正車牌"
          autoFocus
          maxLength={32}
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder="輸入人工確認車牌"
        />
        <span className="plate-editor-actions">
          <button type="submit" disabled={busy}>{busy ? "儲存中…" : "儲存"}</button>
          <button type="button" className="secondary-button" disabled={busy} onClick={cancelEditing}>取消</button>
          {correctedPlate && (
            <button type="button" className="secondary-button" disabled={busy} onClick={restoreAiPlate}>恢復 AI 值</button>
          )}
        </span>
        {error && <span className="plate-error" role="alert">{error}</span>}
      </form>
    );
  }

  return (
    <span className="plate-display">
      <span>{correctedPlate || aiPlate || "未取得"}</span>
      {correctedPlate && <small>人工修正</small>}
      <button
        type="button"
        className="icon-button"
        aria-label="編輯車牌"
        title="編輯車牌"
        onClick={() => setEditing(true)}
      >
        <PencilIcon />
      </button>
      {correctedPlate && <span className="ai-plate-original">AI 原值：{aiPlate || "未取得"}</span>}
    </span>
  );
}

function ReviewEditor({ jobId, unit, onSaved }) {
  const savedVerdict = ["accepted", "rejected"].includes(unit.review?.verdict)
    ? unit.review.verdict
    : "";
  const [verdict, setVerdict] = useState(savedVerdict);
  const [note, setNote] = useState(unit.review?.note || "");
  const [reviewer, setReviewer] = useState(unit.review?.reviewer || "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setVerdict(["accepted", "rejected"].includes(unit.review?.verdict) ? unit.review.verdict : "");
    setNote(unit.review?.note || "");
    setReviewer(unit.review?.reviewer || "");
  }, [unit.event_key, unit.review]);

  async function submit(event) {
    event.preventDefault();
    if (!verdict) return;
    setBusy(true);
    setError("");
    try {
      const result = await api(
        `/api/jobs/${jobId}/reviews/${encodeURIComponent(unit.event_key)}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ verdict, note, reviewer }),
        },
      );
      onSaved(result.case);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  const noEvent = unit.kind === "no_ai_event";
  return (
    <form className="review-form" onSubmit={submit}>
      <fieldset>
        <legend>人工判定</legend>
        <label className={verdict === "accepted" ? "choice active" : "choice"}>
          <input type="radio" name={`verdict-${unit.event_key}`} value="accepted" checked={verdict === "accepted"} onChange={(event) => setVerdict(event.target.value)} />
          {noEvent ? "無事件，判定正確" : "AI 辨識正確"}
        </label>
        <label className={verdict === "rejected" ? "choice active danger" : "choice"}>
          <input type="radio" name={`verdict-${unit.event_key}`} value="rejected" checked={verdict === "rejected"} onChange={(event) => setVerdict(event.target.value)} />
          {noEvent ? "可能漏判" : "AI 誤判"}
        </label>
      </fieldset>
      <div className="review-fields">
        <label>
          審核者
          <input value={reviewer} maxLength={80} onChange={(event) => setReviewer(event.target.value)} placeholder="選填" />
        </label>
        <label>
          備註
          <textarea value={note} maxLength={2000} onChange={(event) => setNote(event.target.value)} placeholder="記錄誤判原因、遮擋或需再次確認的證據" />
        </label>
      </div>
      <div className="review-submit">
        <span>
          {unit.review?.verdict === "uncertain"
            ? "舊版「證據不足」判定，請重新選擇"
            : unit.review ? `上次審核 ${formatDate(unit.review.reviewed_at)}` : "尚未審核"}
        </span>
        <button type="submit" disabled={busy || !verdict}>{busy ? "儲存中…" : "儲存判定"}</button>
      </div>
      {error && <p className="form-message error" role="alert">{error}</p>}
    </form>
  );
}

function EvidenceCard({ jobId, unit, onSeek, onSaved }) {
  const event = unit.event;
  if (!event) {
    return (
      <article className="evidence-card no-event-card">
        <div className="evidence-heading">
          <div><span className="eyebrow">整支影片</span><h3>AI 未確認垃圾事件</h3></div>
        </div>
        <p>這不等於影片中一定沒有違規；請人工播放影片確認是否有漏判。</p>
      </article>
    );
  }

  const isLitter = event.type === "litter";
  return (
    <article className="evidence-card">
      <div className="evidence-heading">
        <div>
          <span className="eyebrow">{isLitter ? `垃圾事件 #${event.id}` : `隨地便溺 · Track ${event.track_id ?? "—"}`}</span>
          <h3>{isLitter ? `${formatTime(event.start_sec)}–${formatTime(event.end_sec)}` : formatTime(event.time_sec)}</h3>
        </div>
        <button type="button" className="seek-button" onClick={() => onSeek(event.start_sec ?? event.time_sec)}>跳到事件</button>
      </div>
      <Confidence label={isLitter ? "RT-DETR 模型信心" : "STGCN 模型信心"} value={event.confidence} />
      <dl className="evidence-grid">
        <div><dt>歸因狀態</dt><dd>{ATTRIBUTION_LABELS[event.attribution_status] || event.attribution_status || "NULL／未提供"}</dd></div>
        <div><dt>可能關聯車輛</dt><dd>{event.vehicle || "NULL"}</dd></div>
        <div><dt>車牌</dt><dd><PlateEditor jobId={jobId} unit={unit} onSaved={onSaved} /></dd></div>
        <div><dt>OCR 狀態</dt><dd>{PLATE_LABELS[event.plate_status] || event.plate_status || "—"}</dd></div>
        <div><dt>OCR 信心</dt><dd>{formatPercent(event.plate_confidence)}</dd></div>
        <div><dt>事件層級</dt><dd>{isLitter ? "confirmed litter" : "confirmed urinate"}</dd></div>
      </dl>
      <p className="evidence-note">模型信心不是準確率；resolved、車輛或車牌仍須人工核對，無可靠關聯時保留 NULL。</p>
    </article>
  );
}

function CaseDetail({ detail, onUpdated, onRetry }) {
  const videoRef = useRef(null);
  if (!detail) return <div className="detail-placeholder">從左側選擇一支影片查看結果。</div>;
  const { job, analysis, review_units: units } = detail;

  function seek(seconds) {
    const value = Number(seconds);
    if (!videoRef.current || !Number.isFinite(value)) return;
    videoRef.current.currentTime = Math.max(0, value);
    videoRef.current.play().catch(() => {});
    videoRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  if (job.status !== "completed") {
    return (
      <div className="processing-card">
        <span className={`large-status status-${job.status}`}>{STATUS_LABELS[job.status] || job.status}</span>
        <h2>{job.original_name}</h2>
        <p>{job.status_message}</p>
        {job.error_message && <pre>{job.error_message}</pre>}
        {job.status === "failed" && <button onClick={() => onRetry(job.id)}>重新排入佇列</button>}
      </div>
    );
  }

  const summary = analysis?.summary || {};
  return (
    <div className="detail-content">
      <div className="case-workspace">
        <div className="video-column">
          <section className="video-section">
            {job.video_available ? (
              <video ref={videoRef} src={videoUrl(job.video_url)} controls preload="metadata" />
            ) : (
              <div className="video-missing">找不到 annotated MP4，仍可檢視 JSON 與審核資料。</div>
            )}
          </section>

          <section className="summary-grid" aria-label="影片摘要">
            <Metric label="確認垃圾事件" value={summary.litter_event_count ?? 0} note="不含 pending candidate" />
            <Metric label="平均模型信心" value={formatPercent(summary.average_litter_confidence)} note="不是 accuracy" />
            <Metric label="經過車輛 Track" value={summary.passed_vehicle_count ?? 0} note="可能受 ID fragmentation 影響" />
            <Metric label="已讀車牌" value={(summary.littering_plates || []).join("、") || "無"} note="OCR 失敗不補造" />
          </section>

          <section className="review-progress">
            <div><span className="eyebrow">人工複核進度</span><strong>{job.reviewed_units} / {job.total_units}</strong></div>
            <div className="progress-track"><span style={{ width: `${job.total_units ? (job.reviewed_units / job.total_units) * 100 : 0}%` }} /></div>
          </section>
        </div>

        <section className="events-section">
          <div className="section-heading"><h2>事件與可能證據</h2><span>{units.length} 個審核項目</span></div>
          <div className="event-list">
            {units.map((unit) => (
              <div className="review-unit" key={unit.event_key}>
                <EvidenceCard jobId={job.id} unit={unit} onSeek={seek} onSaved={onUpdated} />
                <ReviewEditor jobId={job.id} unit={unit} onSaved={onUpdated} />
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}

function App() {
  const initialTab = window.location.hash === "#/reviewed" ? "reviewed" : "unreviewed";
  const [tab, setTab] = useState(initialTab);
  const [jobs, setJobs] = useState([]);
  const [counts, setCounts] = useState({ unreviewed: 0, reviewed: 0, running: 0 });
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [config, setConfig] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [exportBusy, setExportBusy] = useState(false);
  const [exportMessage, setExportMessage] = useState("");

  const loadJobs = useCallback(async (quiet = false) => {
    if (!quiet) setLoading(true);
    try {
      const result = await api(`/api/jobs?review_status=${tab}`);
      setJobs(result.items);
      setCounts(result.counts);
      setSelectedId((current) => result.items.some((item) => item.id === current) ? current : result.items[0]?.id || null);
      setError("");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      if (!quiet) setLoading(false);
    }
  }, [tab]);

  useEffect(() => {
    api("/api/config").then(setConfig).catch((requestError) => setError(requestError.message));
  }, []);

  useEffect(() => {
    loadJobs();
    const interval = window.setInterval(() => loadJobs(true), (config?.poll_seconds || 3) * 1000);
    return () => window.clearInterval(interval);
  }, [loadJobs, config?.poll_seconds]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    api(`/api/jobs/${selectedId}`).then(setDetail).catch((requestError) => setError(requestError.message));
  }, [selectedId]);

  const selectedIndex = useMemo(() => jobs.findIndex((item) => item.id === selectedId), [jobs, selectedId]);

  function changeTab(nextTab) {
    window.location.hash = `/${nextTab}`;
    setTab(nextTab);
    setSelectedId(null);
    setDetail(null);
    setExportMessage("");
  }

  async function exportReviewed() {
    setExportBusy(true);
    setExportMessage("");
    try {
      const result = await download("/api/exports/reviewed", { method: "POST" });
      const url = URL.createObjectURL(result.blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = result.filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      setExportMessage("已下載違規片段與 Excel 壓縮檔");
    } catch (requestError) {
      setExportMessage(requestError.message);
    } finally {
      setExportBusy(false);
    }
  }

  async function updatedCase(nextCase) {
    setDetail(nextCase);
    await loadJobs(true);
  }

  async function retry(jobId) {
    try {
      await api(`/api/jobs/${jobId}/retry`, { method: "POST" });
      await loadJobs(true);
      setDetail(await api(`/api/jobs/${jobId}`));
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <IntakePanel config={config} onChanged={() => loadJobs()} />
        <nav className="review-tabs" aria-label="審核分類">
          <button className={tab === "unreviewed" ? "active" : ""} onClick={() => changeTab("unreviewed")}>
            未審核 <span>{counts.unreviewed}</span>
          </button>
          <button className={tab === "reviewed" ? "active" : ""} onClick={() => changeTab("reviewed")}>
            已審核 <span>{counts.reviewed}</span>
          </button>
        </nav>
        {tab === "reviewed" && (
          <div className="review-export">
            <button type="button" disabled={exportBusy || counts.reviewed === 0} onClick={exportReviewed}>
              {exportBusy ? "正在剪輯與整理…" : "匯出已審核違規"}
            </button>
            <small>只匯出完整審核案件中判定「AI 辨識正確」的事件。</small>
            {exportMessage && <span role="status">{exportMessage}</span>}
          </div>
        )}
        {error && <div className="global-error" role="alert">{error}</div>}
        {loading ? <div className="list-loading">載入案件中…</div> : <CaseList jobs={jobs} selectedId={selectedId} onSelect={setSelectedId} />}
        <div className="case-navigation">
          <button disabled={selectedIndex <= 0} onClick={() => setSelectedId(jobs[selectedIndex - 1]?.id)}>上一支</button>
          <span>{selectedIndex >= 0 ? `${selectedIndex + 1} / ${jobs.length}` : `0 / ${jobs.length}`}</span>
          <button disabled={selectedIndex < 0 || selectedIndex >= jobs.length - 1} onClick={() => setSelectedId(jobs[selectedIndex + 1]?.id)}>下一支</button>
        </div>
      </aside>

      <main className="main-panel">
        <CaseDetail detail={detail} onUpdated={updatedCase} onRetry={retry} />
      </main>
    </div>
  );
}

export default App;
