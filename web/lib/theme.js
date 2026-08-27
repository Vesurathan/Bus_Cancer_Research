// Shared visual helpers for the agent demo.

export const TIER_COLOR = {
  head: "var(--head)",
  medium: "var(--medium)",
  tail: "var(--tail)",
};

export const TIER_LABEL = {
  head: "Head",
  medium: "Medium",
  tail: "Tail",
};

export const STEPS = [
  { key: "retrieve", label: "Retrieve", sub: "BiomedCLIP + FAISS" },
  { key: "find", label: "Find", sub: "LDAM-DRW classifier" },
  { key: "draft", label: "Draft", sub: "ViT + decoder" },
  { key: "verify", label: "Verify", sub: "consistency gate" },
  { key: "refine", label: "Refine", sub: "LLM rewrite" },
];

// BI-RADS category -> {label, color} for the risk badge.
export function biRadsInfo(cat) {
  if (!cat) return { label: "n/a", color: "var(--muted)", risk: "unspecified" };
  const n = parseInt(String(cat), 10);
  if (n <= 2) return { label: `BI-RADS ${cat}`, color: "var(--head)", risk: "benign" };
  if (n === 3) return { label: `BI-RADS ${cat}`, color: "var(--medium)", risk: "probably benign" };
  return { label: `BI-RADS ${cat}`, color: "var(--danger)", risk: "suspicious" };
}

// Split the three-field report into labelled parts for nicer rendering.
export function parseReport(text) {
  if (!text) return [];
  const fields = ["Tissue composition", "Shape and margin", "Findings"];
  const parts = [];
  for (let i = 0; i < fields.length; i++) {
    const start = text.indexOf(fields[i]);
    if (start < 0) continue;
    const nextStart = i + 1 < fields.length ? text.indexOf(fields[i + 1]) : -1;
    const chunk = (nextStart > 0 ? text.slice(start, nextStart) : text.slice(start))
      .replace(/^\s*[A-Za-z ]+:\s*/, "")
      .replace(/\.\s*$/, "")
      .trim();
    parts.push({ label: fields[i], value: chunk });
  }
  return parts;
}
