"use client";
import { useCallback, useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import PipelineFlow from "@/components/PipelineFlow";
import ExemplarStrip from "@/components/ExemplarStrip";
import FindingsPanel from "@/components/FindingsPanel";
import VerifyPanel from "@/components/VerifyPanel";
import Typewriter from "@/components/Typewriter";
import { biRadsInfo, parseReport } from "@/lib/theme";

const STEP_MS = [1700, 2100, 2300, 1600, 2400]; // retrieve, find, draft, verify, refine

function StepCard({ index, step, title, hint, children }) {
  const visible = step >= index;
  return (
    <AnimatePresence>
      {visible && (
        <motion.section
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45 }}
          className="card"
          style={{ padding: 18 }}
        >
          <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 12 }}>
            <span style={{ fontSize: 12, fontWeight: 700, color: "var(--accent)", letterSpacing: 0.4 }}>
              {String(index + 1).padStart(2, "0")}
            </span>
            <h3 style={{ margin: 0, fontSize: 15.5 }}>{title}</h3>
            <span style={{ marginLeft: "auto", fontSize: 11.5, color: "var(--muted)" }}>{hint}</span>
          </div>
          {children}
        </motion.section>
      )}
    </AnimatePresence>
  );
}

function ReportBlock({ text, run }) {
  const parts = parseReport(text);
  if (!parts.length) return <div className="mono" style={{ fontSize: 13 }}><Typewriter text={text} run={run} /></div>;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {parts.map((p) => (
        <div key={p.label} style={{ display: "flex", gap: 10, alignItems: "baseline" }}>
          <span style={{ fontSize: 11.5, color: "var(--muted)", width: 128, flexShrink: 0, textAlign: "right" }}>{p.label}</span>
          <span style={{ fontSize: 13.5, lineHeight: 1.4 }}>{p.value}</span>
        </div>
      ))}
    </div>
  );
}

export default function Page() {
  const [data, setData] = useState(null);
  const [sel, setSel] = useState(0);
  const [step, setStep] = useState(-1);

  useEffect(() => {
    fetch("/demo_data.json").then((r) => r.json()).then(setData).catch(() => setData({ cases: [] }));
  }, []);

  const cases = data?.cases || [];
  const c = cases[sel];
  const refineApplied = c?.steps?.refine?.applied;

  // Timeline: advance step-by-step. Refine step is skipped when no refine happened.
  useEffect(() => {
    if (step < 0 || step > 4 || !c) return;
    if (step === 4 && !refineApplied) { setStep(5); return; }
    const id = setTimeout(() => setStep((s) => s + 1), STEP_MS[step]);
    return () => clearTimeout(id);
  }, [step, c, refineApplied]);

  const play = useCallback(() => setStep(0), []);
  const pick = useCallback((i) => { setSel(i); setStep(-1); }, []);
  useEffect(() => { if (data && step === -1) { const t = setTimeout(play, 500); return () => clearTimeout(t); } }, [data, sel]); // eslint-disable-line

  const finalReport = refineApplied ? c?.steps?.refine?.report : c?.steps?.draft?.report;
  const br = biRadsInfo(c?.biRads);
  const skipped = useMemo(() => (refineApplied ? [] : [4]), [refineApplied]);

  if (!data) return <div style={{ padding: 40, color: "var(--muted)" }}>Loading demo…</div>;
  if (!c) return <div style={{ padding: 40, color: "var(--muted)" }}>No demo data found. Run <code>python export_demo.py</code>.</div>;

  return (
    <main style={{ maxWidth: 1240, margin: "0 auto", padding: "26px 22px 60px" }}>
      {/* Header */}
      <header style={{ display: "flex", alignItems: "flex-end", gap: 16, marginBottom: 20, flexWrap: "wrap" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ width: 12, height: 12, borderRadius: 3, background: "var(--accent)", boxShadow: "0 0 14px var(--accent)" }} />
            <h1 style={{ margin: 0, fontSize: 22, letterSpacing: 0.2 }}>Agentic Breast-Ultrasound Reporting</h1>
          </div>
          <p style={{ margin: "6px 0 0", color: "var(--muted)", fontSize: 13.5 }}>
            An agent turns a BUS image into a structured report: <b style={{ color: "var(--text)" }}>retrieve → find → draft → verify → refine</b>.
          </p>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 8, flexWrap: "wrap" }}>
          <Badge label={data.meta?.dataset} />
          <Badge label={`${data.meta?.nTerms} interpretation terms`} />
          <Badge label={`refiner: ${data.meta?.model}`} accent />
        </div>
      </header>

      <PipelineFlow active={step >= 0 && step <= 4 ? step : -1} done={step - 1} skipped={skipped} />

      <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: 18, marginTop: 18 }}>
        {/* Left: image + gallery */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div className="card" style={{ padding: 12 }}>
            <div style={{ position: "relative" }}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={`/${c.image}`} alt={c.id} style={{ width: "100%", borderRadius: 12, display: "block", filter: "contrast(1.06)" }} />
              <span style={{ position: "absolute", top: 10, left: 10, fontSize: 12, padding: "3px 9px", borderRadius: 8, background: "rgba(4,18,31,0.82)", border: "1px solid var(--border)" }} className="mono">
                {c.id}
              </span>
              <span style={{ position: "absolute", top: 10, right: 10, fontSize: 12, fontWeight: 700, padding: "3px 9px", borderRadius: 8, background: "rgba(4,18,31,0.82)", border: `1px solid ${br.color}`, color: br.color }}>
                {br.label}
              </span>
            </div>
            <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 8, textAlign: "center" }}>{br.risk}</div>
            <button onClick={play} style={btn}>▶ Replay agent</button>
          </div>

          <div className="card" style={{ padding: 12 }}>
            <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 8 }}>Cases</div>
            <div className="scroll-thin" style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 6, maxHeight: 210, overflowY: "auto" }}>
              {cases.map((cc, i) => (
                <button key={cc.id} onClick={() => pick(i)} title={cc.id}
                  style={{ padding: 0, border: i === sel ? "2px solid var(--accent)" : "1px solid var(--border)", borderRadius: 8, overflow: "hidden", cursor: "pointer", background: "none" }}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={`/${cc.image}`} alt={cc.id} style={{ width: "100%", height: 48, objectFit: "cover", display: "block", opacity: i === sel ? 1 : 0.7 }} />
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right: the agent story */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <StepCard index={0} step={step} title="Retrieve — similar cases" hint="BiomedCLIP + FAISS, top-3 by cosine similarity">
            <ExemplarStrip exemplars={c.steps.retrieve.exemplars} active={step >= 0} />
          </StepCard>

          <StepCard index={1} step={step} title="Find — long-tailed interpretation" hint="LDAM-DRW + curriculum-margin classifier">
            <FindingsPanel findings={c.steps.find.findings} goldTerms={c.goldTerms} active={step >= 1} />
          </StepCard>

          <StepCard index={2} step={step} title="Draft — generated report" hint="ViT encoder + Transformer decoder">
            <ReportBlock text={c.steps.draft.report} run={step === 2} />
          </StepCard>

          <StepCard index={3} step={step} title="Verify — consistency gate" hint="coverage · contradiction · structure">
            <VerifyPanel issues={c.steps.verify.issues} verified={c.steps.verify.verified} active={step >= 3} />
          </StepCard>

          {refineApplied && (
            <StepCard index={4} step={step} title="Refine — LLM rewrite" hint={`local ${data.meta?.model}, preserves fluency`}>
              <ReportBlock text={c.steps.refine.report} run={step === 4} />
            </StepCard>
          )}

          <AnimatePresence>
            {step >= 5 && (
              <motion.section initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="card" style={{ padding: 18, borderColor: "rgba(52,211,153,0.35)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
                  <span style={{ color: "var(--ok)", fontWeight: 800 }}>✓ Final report</span>
                  <span style={{ fontSize: 11.5, color: "var(--muted)" }}>
                    {c.final.attempts} pass{c.final.attempts > 1 ? "es" : ""} · verified consistent
                  </span>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                  <Compare title="Agent output" tone="accent"><ReportBlock text={finalReport} run={false} /></Compare>
                  <Compare title="Radiologist ground truth" tone="muted"><ReportBlock text={c.goldReport} run={false} /></Compare>
                </div>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 14 }}>
                  {c.goldTerms.map((t) => (
                    <span key={t} className="mono" style={{ fontSize: 11.5, padding: "3px 8px", borderRadius: 7, background: "rgba(56,189,248,0.08)", border: "1px solid var(--border)", color: "var(--muted)" }}>{t}</span>
                  ))}
                </div>
              </motion.section>
            )}
          </AnimatePresence>
        </div>
      </div>

      <footer style={{ marginTop: 28, fontSize: 12, color: "var(--muted)", textAlign: "center", lineHeight: 1.6 }}>
        Research demo replaying real model outputs on the BrEaST-Lesions-USG dataset — not a clinical device and not for diagnostic use.<br />
        Contribution: loss-level long-tail control (LDAM-DRW + curriculum margins) inside a retrieve→find→draft→verify→refine agent.
      </footer>
    </main>
  );
}

const btn = {
  width: "100%", marginTop: 10, padding: "9px 12px", borderRadius: 10, cursor: "pointer",
  background: "linear-gradient(180deg,#123049,#0e2136)", color: "var(--text)",
  border: "1px solid rgba(56,189,248,0.4)", fontSize: 13.5, fontWeight: 600,
};

function Badge({ label, accent }) {
  return (
    <span style={{ fontSize: 11.5, padding: "5px 10px", borderRadius: 999, background: accent ? "rgba(34,211,238,0.10)" : "var(--panel)", border: `1px solid ${accent ? "rgba(34,211,238,0.4)" : "var(--border)"}`, color: accent ? "var(--accent-2)" : "var(--muted)" }}>
      {label}
    </span>
  );
}

function Compare({ title, tone, children }) {
  return (
    <div className="card" style={{ padding: 14, background: "var(--bg-soft)" }}>
      <div style={{ fontSize: 11.5, color: tone === "accent" ? "var(--accent-2)" : "var(--muted)", marginBottom: 10, textTransform: "uppercase", letterSpacing: 0.5 }}>{title}</div>
      {children}
    </div>
  );
}
