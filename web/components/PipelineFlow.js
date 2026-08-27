"use client";
import { motion } from "framer-motion";
import { STEPS } from "@/lib/theme";

// Horizontal 5-node agent flow. `active` = index currently running,
// `done` = highest index completed. Animated dashed connectors between nodes.
export default function PipelineFlow({ active, done, skipped = [] }) {
  return (
    <div className="card" style={{ padding: "18px 20px" }}>
      <div style={{ display: "flex", alignItems: "stretch", gap: 0 }}>
        {STEPS.map((s, i) => {
          const isActive = i === active;
          const isDone = i <= done && i !== active;
          const isSkip = skipped.includes(i);
          return (
            <div key={s.key} style={{ display: "flex", alignItems: "center", flex: 1 }}>
              <motion.div
                initial={false}
                animate={{
                  scale: isActive ? 1.04 : 1,
                  opacity: isSkip ? 0.4 : 1,
                }}
                className={`card ${isActive ? "glow" : ""}`}
                style={{
                  flex: 1,
                  padding: "12px 12px",
                  background: isActive
                    ? "linear-gradient(180deg,#12233b,#0f1a2c)"
                    : isDone
                    ? "linear-gradient(180deg,#0f1d16,#0d1622)"
                    : "var(--panel)",
                  borderColor: isActive
                    ? "rgba(56,189,248,0.55)"
                    : isDone
                    ? "rgba(52,211,153,0.35)"
                    : "var(--border)",
                  textAlign: "center",
                  position: "relative",
                }}
              >
                <div style={{ display: "flex", justifyContent: "center", marginBottom: 6 }}>
                  <span
                    style={{
                      width: 26, height: 26, borderRadius: 999,
                      display: "grid", placeItems: "center",
                      fontSize: 13, fontWeight: 700,
                      color: isActive ? "#04121f" : isDone ? "#04120c" : "var(--muted)",
                      background: isActive ? "var(--accent)" : isDone ? "var(--ok)" : "#1c2942",
                    }}
                  >
                    {isDone ? "✓" : i + 1}
                  </span>
                </div>
                <div style={{ fontSize: 14, fontWeight: 650 }}>{s.label}</div>
                <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>{s.sub}</div>
                {isActive && (
                  <span
                    className="pulse-dot"
                    style={{
                      position: "absolute", top: 8, right: 10, width: 8, height: 8,
                      borderRadius: 999, background: "var(--accent-2)",
                    }}
                  />
                )}
              </motion.div>

              {i < STEPS.length - 1 && (
                <svg width="34" height="20" viewBox="0 0 34 20" style={{ flexShrink: 0 }}>
                  <line
                    x1="2" y1="10" x2="32" y2="10"
                    stroke={i < done ? "var(--ok)" : "#2a3a58"}
                    strokeWidth="2"
                    className={i === active - 1 || i === active ? "flow-line" : ""}
                  />
                  <path
                    d="M27,5 L32,10 L27,15"
                    fill="none"
                    stroke={i < done ? "var(--ok)" : "#2a3a58"}
                    strokeWidth="2"
                  />
                </svg>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
