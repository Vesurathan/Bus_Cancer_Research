import "./globals.css";

export const metadata = {
  title: "BUS Agent — Breast Ultrasound Report Generation",
  description:
    "Research demo: an agentic pipeline that generates radiology reports from breast-ultrasound images (retrieve → find → draft → verify → refine).",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
