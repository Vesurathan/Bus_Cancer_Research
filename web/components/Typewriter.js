"use client";
import { useEffect, useState } from "react";

// Streams `text` character by character while `run` is true. Calls onDone once.
export default function Typewriter({ text = "", run, speed = 12, onDone }) {
  const [n, setN] = useState(0);

  useEffect(() => {
    if (!run) { setN(0); return; }
    if (n >= text.length) { onDone && onDone(); return; }
    const id = setTimeout(() => setN((k) => k + Math.max(1, Math.round(text.length / 220))), speed);
    return () => clearTimeout(id);
  }, [run, n, text, speed, onDone]);

  useEffect(() => { if (run) setN(0); }, [text, run]);

  return (
    <span>
      {text.slice(0, n)}
      {run && n < text.length && <span className="blink">▍</span>}
    </span>
  );
}
