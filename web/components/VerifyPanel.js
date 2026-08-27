"use client";
import { motion } from "framer-motion";

// Shows the verifier's consistency issues (coverage / contradiction) as badges.
export default function VerifyPanel({ issues = [], verified, active }) {
  const clean = verified || issues.length === 0;
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
        <motion.span
          initial={{ scale: 0 }}
          animate={active ? { scale: 1 } : { scale: 0 }}
          transition={{ type: "spring", stiffness: 300, damping: 18 }}
          style={{
            width: 30, height: 30, borderRadius: 999, display: "grid", placeItems: "center",
            background: clean ? "rgba(52,211,153,0.15)" : "rgba(248,113,113,0.15)",
            border: `1px solid ${clean ? "var(--ok)" : "var(--danger)"}`,
            color: clean ? "var(--ok)" : "var(--danger)", fontWeight: 800,
          }}
        >
          {clean ? "✓" : issues.length}
        </motion.span>
        <div style={{ fontSize: 13.5 }}>
          {clean ? (
            <span>Draft is consistent with the findings.</span>
          ) : (
            <span>{issues.length} inconsistenc{issues.length === 1 ? "y" : "ies"} found — sending to refine.</span>
          )}
        </div>
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
        {issues.map((it, i) => {
          const contra = it.kind === "contradiction";
          return (
            <motion.span
              key={i}
              initial={{ opacity: 0, y: 8 }}
              animate={active ? { opacity: 1, y: 0 } : { opacity: 0 }}
              transition={{ delay: 0.1 * i }}
              style={{
                fontSize: 12, padding: "5px 10px", borderRadius: 8,
                background: contra ? "rgba(248,113,113,0.10)" : "rgba(251,191,36,0.10)",
                border: `1px solid ${contra ? "rgba(248,113,113,0.4)" : "rgba(251,191,36,0.4)"}`,
                color: contra ? "var(--danger)" : "var(--medium)",
              }}
            >
              {contra ? "contradiction" : "missing"} · {it.term}
            </motion.span>
          );
        })}
      </div>
    </div>
  );
}
