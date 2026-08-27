"use client";
import { motion } from "framer-motion";

// Retrieved nearest-neighbour cases from the train corpus (BiomedCLIP + FAISS),
// with cosine similarity. These ground the report (RAG).
export default function ExemplarStrip({ exemplars = [], active }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
      {exemplars.map((e, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, y: 14 }}
          animate={active ? { opacity: 1, y: 0 } : { opacity: 0.2, y: 8 }}
          transition={{ delay: 0.18 * i, duration: 0.5 }}
          className="card"
          style={{ padding: 8, overflow: "hidden" }}
        >
          <div style={{ position: "relative" }}>
            {e.image ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={`/${e.image}`} alt={`exemplar ${i}`} style={{ width: "100%", height: 88, objectFit: "cover", borderRadius: 8, filter: "contrast(1.05)" }} />
            ) : (
              <div style={{ height: 88, borderRadius: 8, background: "#0c1424" }} />
            )}
            <span
              className="mono"
              style={{
                position: "absolute", top: 6, right: 6, fontSize: 11,
                padding: "2px 6px", borderRadius: 6,
                background: "rgba(4,18,31,0.8)", color: "var(--accent-2)",
                border: "1px solid rgba(34,211,238,0.35)",
              }}
            >
              {e.sim?.toFixed(2)}
            </span>
          </div>
          <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 6, lineHeight: 1.35, maxHeight: 46, overflow: "hidden" }}>
            {e.report?.replace(/Tissue composition:\s*/, "") || "—"}
          </div>
        </motion.div>
      ))}
    </div>
  );
}
