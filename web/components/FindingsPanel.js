"use client";
import { motion } from "framer-motion";
import { TIER_COLOR, TIER_LABEL } from "@/lib/theme";

// Animated per-term confidence bars, grouped head/medium/tail. `active` gates
// the fill animation so bars grow when this step runs. HIGH_CONF = 0.60 gate.
const GATE = 0.6;

export default function FindingsPanel({ findings, active, goldTerms = [] }) {
  // Always show the top terms by probability, plus any ground-truth term not
  // already in that set — so the long-tail chart reads fully without 28 flat rows.
  const top = findings.slice(0, 8);
  const goldExtra = findings.filter(
    (f) => goldTerms.includes(f.term) && !top.some((t) => t.term === f.term)
  );
  const shown = [...top, ...goldExtra].slice(0, 10);

  return (
    <div>
      <div style={{ display: "flex", gap: 14, marginBottom: 12 }}>
        {["head", "medium", "tail"].map((t) => (
          <span key={t} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--muted)" }}>
            <span style={{ width: 10, height: 10, borderRadius: 3, background: TIER_COLOR[t] }} />
            {TIER_LABEL[t]} tier
          </span>
        ))}
        <span style={{ marginLeft: "auto", fontSize: 12, color: "var(--muted)" }}>
          asserted @ ≥ {GATE.toFixed(2)}
        </span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
        {shown.map((f, i) => {
          const isGold = goldTerms.includes(f.term);
          const over = f.prob >= GATE;
          return (
            <div key={f.term} style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div
                style={{
                  width: 190, fontSize: 12.5, textAlign: "right", flexShrink: 0,
                  color: over ? "var(--text)" : "var(--muted)",
                  fontWeight: over ? 600 : 400,
                  whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                }}
                title={f.term}
              >
                {isGold && <span style={{ color: "var(--accent-2)" }}>● </span>}
                {f.term}
              </div>
              <div
                style={{
                  position: "relative", flex: 1, height: 16, borderRadius: 8,
                  background: "#0c1424", border: "1px solid var(--border)", overflow: "hidden",
                }}
              >
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: active ? `${Math.max(2, f.prob * 100)}%` : 0 }}
                  transition={{ duration: 0.9, delay: 0.06 * i, ease: "easeOut" }}
                  style={{
                    height: "100%",
                    background: over
                      ? `linear-gradient(90deg, ${TIER_COLOR[f.tier]}, ${TIER_COLOR[f.tier]}cc)`
                      : "#2b3a58",
                    boxShadow: over ? `0 0 14px -3px ${TIER_COLOR[f.tier]}` : "none",
                  }}
                />
                {/* gate marker */}
                <span style={{ position: "absolute", left: `${GATE * 100}%`, top: 0, bottom: 0, width: 1, background: "rgba(255,255,255,0.25)" }} />
              </div>
              <div className="mono" style={{ width: 44, fontSize: 12, textAlign: "right", color: over ? "var(--text)" : "var(--muted)" }}>
                {f.prob.toFixed(2)}
              </div>
            </div>
          );
        })}
      </div>
      <div style={{ fontSize: 11.5, color: "var(--muted)", marginTop: 12 }}>
        <span style={{ color: "var(--accent-2)" }}>●</span> = present in the ground-truth report.
        The classifier is long-tail–aware (LDAM-DRW + curriculum margins); tail terms are hardest.
      </div>
    </div>
  );
}
