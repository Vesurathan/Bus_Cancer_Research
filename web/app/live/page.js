"use client";
import { useCallback, useRef, useState } from "react";
import { motion } from "framer-motion";
import { predict } from "@/lib/api";

const MALIGNANT = "#f87171";
const BENIGN = "#34d399";

function pct(x) {
  return `${Math.round((x ?? 0) * 100)}%`;
}

export default function LivePage() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [res, setRes] = useState(null);
  const inputRef = useRef(null);

  const choose = useCallback((f) => {
    if (!f) return;
    setFile(f);
    setRes(null);
    setErr(null);
    setPreview(URL.createObjectURL(f));
  }, []);

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      choose(e.dataTransfer.files?.[0]);
    },
    [choose]
  );

  const run = useCallback(async () => {
    if (!file) return;
    setBusy(true);
    setErr(null);
    setRes(null);
    try {
      const out = await predict(file);
      setRes(out);
    } catch (e) {
      setErr(e.message || "request failed");
    } finally {
      setBusy(false);
    }
  }, [file]);

  const malignant = res?.decision === "malignant";
  const tone = malignant ? MALIGNANT : BENIGN;

  return (
    <main style={{ maxWidth: 1120, margin: "0 auto", padding: "26px 22px 64px" }}>
      <header style={{ marginBottom: 20 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ width: 12, height: 12, borderRadius: 3, background: "var(--accent)", boxShadow: "0 0 14px var(--accent)" }} />
          <h1 style={{ margin: 0, fontSize: 22 }}>Live Breast-Ultrasound Analysis</h1>
          <a href="/" style={{ marginLeft: "auto", fontSize: 13, color: "var(--muted)" }}>← replay demo</a>
        </div>
        <p style={{ margin: "6px 0 0", color: "var(--muted)", fontSize: 13.5 }}>
          Upload a BUS image. The agent fuses a ViT, BiomedCLIP retrieval and lesion
          morphology into a calibrated P(malignant), localises the lesion with Grad-CAM,
          and drafts a structured report.
        </p>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "360px 1fr", gap: 18, alignItems: "start" }}>
        {/* Uploader */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div
            className="card"
            onDragOver={(e) => e.preventDefault()}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            style={{
              padding: 16, cursor: "pointer", textAlign: "center",
              borderStyle: "dashed", minHeight: 220, display: "flex",
              flexDirection: "column", justifyContent: "center", gap: 10,
            }}
          >
            {preview ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={preview} alt="upload" style={{ width: "100%", borderRadius: 10, display: "block" }} />
            ) : (
              <>
                <div style={{ fontSize: 34 }}>🩻</div>
                <div style={{ fontSize: 14 }}>Drop a BUS image here</div>
                <div style={{ fontSize: 12, color: "var(--muted)" }}>or click to browse — PNG / JPG</div>
              </>
            )}
            <input ref={inputRef} type="file" accept="image/*" hidden
              onChange={(e) => choose(e.target.files?.[0])} />
          </div>

          <button onClick={run} disabled={!file || busy} style={{
            padding: "11px 12px", borderRadius: 10, fontSize: 14, fontWeight: 700,
            cursor: !file || busy ? "not-allowed" : "pointer",
            background: !file || busy ? "var(--panel)" : "linear-gradient(180deg,#123049,#0e2136)",
            color: "var(--text)", border: "1px solid rgba(56,189,248,0.4)",
            opacity: !file || busy ? 0.6 : 1,
          }}>
            {busy ? "Analysing…" : "▶ Analyse image"}
          </button>

          {err && (
            <div className="card" style={{ padding: 12, borderColor: "var(--danger)", color: "var(--danger)", fontSize: 13 }}>
              {err}
              <div style={{ color: "var(--muted)", marginTop: 6, fontSize: 12 }}>
                Is the backend reachable? Check NEXT_PUBLIC_API_URL.
              </div>
            </div>
          )}
        </div>

        {/* Result */}
        <div style={{ minHeight: 220 }}>
            {busy && (
              <div className="card" style={{ padding: 24, color: "var(--muted)", fontSize: 14 }}>
                Running the agent — fusion → findings → report → Grad-CAM…
              </div>
            )}

            {!busy && !res && (
              <div className="card" style={{ padding: 24, color: "var(--muted)", fontSize: 14 }}>
                The analysis will appear here.
              </div>
            )}

            {!busy && res && (
              <motion.div key="res" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
                style={{ display: "flex", flexDirection: "column", gap: 14 }}>

                {/* Decision */}
                <div className="card" style={{ padding: 18, borderColor: tone }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
                    <span style={{ fontSize: 20, fontWeight: 800, color: tone, textTransform: "capitalize" }}>
                      {res.decision}
                    </span>
                    <span style={{ fontSize: 13, color: "var(--muted)" }}>
                      P(malignant) <b style={{ color: "var(--text)" }}>{pct(res.pMalignant)}</b>
                    </span>
                    <span style={{ fontSize: 13, color: "var(--muted)" }}>
                      confidence <b style={{ color: "var(--text)" }}>{pct(res.confidence)}</b>
                    </span>
                    <span style={{ marginLeft: "auto", fontSize: 12.5, fontWeight: 700, padding: "4px 10px", borderRadius: 8, border: `1px solid ${tone}`, color: tone }}>
                      BI-RADS {res.biRads}
                    </span>
                  </div>
                  {/* probability bar */}
                  <div style={{ marginTop: 12, height: 8, borderRadius: 6, background: "var(--panel-2)", overflow: "hidden" }}>
                    <div style={{ width: pct(res.pMalignant), height: "100%", background: tone }} />
                  </div>
                  {res.rationale && (
                    <div style={{ marginTop: 10, fontSize: 12.5, color: "var(--muted)" }}>{res.rationale}</div>
                  )}
                  {res.review?.flagged && (
                    <div style={{ marginTop: 10, fontSize: 12.5, color: "var(--medium)", background: "rgba(251,191,36,0.08)", border: "1px solid rgba(251,191,36,0.3)", borderRadius: 8, padding: "7px 10px" }}>
                      ⚑ Flagged for human review — {res.review.reason}
                    </div>
                  )}
                </div>

                {/* Images: input + Grad-CAM */}
                <div className="card" style={{ padding: 14 }}>
                  <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 10 }}>Lesion localisation (Grad-CAM)</div>
                  <div style={{ display: "grid", gridTemplateColumns: res.heatmap ? "1fr 1fr" : "1fr", gap: 12 }}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <figure style={{ margin: 0 }}>
                      <img src={res.image} alt="input" style={{ width: "100%", borderRadius: 10, display: "block" }} />
                      <figcaption style={{ fontSize: 11.5, color: "var(--muted)", textAlign: "center", marginTop: 6 }}>input</figcaption>
                    </figure>
                    {res.heatmap && (
                      // eslint-disable-next-line @next/next/no-img-element
                      <figure style={{ margin: 0 }}>
                        <img src={res.heatmap} alt="grad-cam" style={{ width: "100%", borderRadius: 10, display: "block" }} />
                        <figcaption style={{ fontSize: 11.5, color: "var(--muted)", textAlign: "center", marginTop: 6 }}>where the ViT looked</figcaption>
                      </figure>
                    )}
                  </div>
                </div>

                {/* Findings */}
                {(res.findings?.length > 0 || res.possibleFindings?.length > 0) && (
                  <div className="card" style={{ padding: 14 }}>
                    <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 10 }}>Findings</div>
                    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                      {res.findings.map((f) => (
                        <span key={f.term} className="mono" style={{ fontSize: 11.5, padding: "3px 9px", borderRadius: 7, background: "rgba(52,211,153,0.10)", border: "1px solid rgba(52,211,153,0.35)", color: "var(--head)" }}>
                          {f.term} · {f.prob}
                        </span>
                      ))}
                      {res.possibleFindings.map((f) => (
                        <span key={f.term} className="mono" title="surfaced from similar cases (context, not asserted)" style={{ fontSize: 11.5, padding: "3px 9px", borderRadius: 7, background: "rgba(56,189,248,0.06)", border: "1px dashed var(--border)", color: "var(--muted)" }}>
                          {f.term}?
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Report */}
                {res.report && (
                  <div className="card" style={{ padding: 14 }}>
                    <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 10 }}>Generated report</div>
                    <div style={{ fontSize: 13.5, lineHeight: 1.5 }}>{res.report}</div>
                  </div>
                )}

                <div style={{ fontSize: 11.5, color: "var(--muted)", textAlign: "center", lineHeight: 1.6 }}>
                  {res.disclaimer}
                </div>
              </motion.div>
            )}
        </div>
      </div>
    </main>
  );
}
