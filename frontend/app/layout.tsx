import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SceneNapse Studio",
  description: "AI-powered cinematic scene composition",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-slate-950 font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
