"use client"

import React, { useState, useRef, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogTrigger } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Send, CheckCircle2, XCircle, LayoutGrid, Sparkles, Image as ImageIcon } from 'lucide-react';

// Types
type LogMessage = {
  type: string;
  message?: string;
  step?: string;
  highlight?: boolean;
}

type GeneratedImage = {
  seed: number;
  url: string;
  score: number;
  is_best: boolean;
}

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [mode, setMode] = useState("optimized");
  const [isGenerating, setIsGenerating] = useState(false);
  const [logs, setLogs] = useState<LogMessage[]>([]);
  const [images, setImages] = useState<GeneratedImage[]>([]);
  const [bestImage, setBestImage] = useState<GeneratedImage | null>(null);
  const [evalResult, setEvalResult] = useState<any>(null);
  const [history, setHistory] = useState<string[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const addLog = (msg: LogMessage) => {
    setLogs(prev => [...prev, msg]);
  };

  const handleSubmit = async () => {
    if (!prompt.trim()) return;

    setIsGenerating(true);
    setLogs([]);
    setImages([]);
    setBestImage(null);
    setEvalResult(null);

    // Add to history
    if (!history.includes(prompt)) {
      setHistory(prev => [prompt, ...prev]);
    }

    try {
      const response = await fetch("http://localhost:8000/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, mode })
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const jsonStr = line.slice(6);
            try {
              const event = JSON.parse(jsonStr);
              handleEvent(event);
            } catch (e) {
              console.error("Parse error", e);
            }
          }
        }
      }
    } catch (e) {
      addLog({ type: "error", message: `Connection Error: ${e}` });
    } finally {
      setIsGenerating(false);
    }
  };

  const handleEvent = (event: any) => {
    switch (event.type) {
      case "status":
        addLog({ type: "status", message: event.message, step: event.step });
        break;
      case "image_generated":
        setImages(prev => {
          const newImages = [...prev, event];
          // Sort by seed to keep grid stable
          return newImages.sort((a, b) => a.seed - b.seed);
        });
        if (event.is_best) {
          // Wait for explicit selection usually regarding final, but here update realtime
          // Actually best updates dynamically
        }
        break;
      case "best_selected":
        addLog({ type: "info", message: `🏆 Best Candidate Selected (Score: ${event.score.toFixed(4)})`, highlight: true });
        // Can find image in array and highlight it
        break;
      case "eval_result":
        setEvalResult(event);
        if (event.passed) {
          addLog({ type: "success", message: "✅ All Evaluations Passed!", highlight: true });
        } else {
          addLog({ type: "error", message: "❌ Evaluations Failed. Rewrite required." });
        }
        break;
      case "rewrite_done":
        addLog({ type: "info", message: `🔄 Rewritten Prompt: "${event.new_prompt}"` });
        break;
      case "success":
        addLog({ type: "success", message: event.message, highlight: true });
        break;
      case "failure":
        addLog({ type: "error", message: event.message });
        break;
      case "fibo_json":
        addLog({ type: "info", message: "Struct: " + JSON.stringify(event.json_prompt).slice(0, 50) + "..." });
        break;
    }
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-50 font-sans">
      {/* Sidebar history */}
      <div className="w-64 border-r border-slate-800 bg-slate-900/50 p-4 flex flex-col">
        <div className="flex items-center gap-2 mb-6 text-blue-400">
          <LayoutGrid className="w-6 h-6" />
          <span className="font-bold text-lg tracking-tight">Scenenapse</span>
        </div>
        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">History</h3>
        <ScrollArea className="flex-1">
          <div className="space-y-2">
            {history.map((h, i) => (
              <button key={i} onClick={() => setPrompt(h)} className="w-full text-left text-sm text-slate-300 hover:bg-slate-800 p-2 rounded truncate">
                {h}
              </button>
            ))}
          </div>
        </ScrollArea>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">

        {/* Header / Mode Select */}
        <div className="border-b border-slate-800 p-4 flex justify-between items-center bg-slate-900/30">
          <div className="flex items-center gap-4">
            <Badge variant="outline" className={mode === 'optimized' ? 'bg-blue-500/10 text-blue-400 border-blue-500/50' : 'text-slate-500'}>
              {mode === 'optimized' ? '✨ Optimized Mode' : 'Raw Mode'}
            </Badge>
          </div>

          <Select value={mode} onValueChange={setMode}>
            <SelectTrigger className="w-[200px] bg-slate-900 border-slate-700">
              <SelectValue placeholder="Select Mode" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="optimized">✨ Prompt Optimization</SelectItem>
              <SelectItem value="regular">🍌 Nano Banana Pro</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="flex-1 flex overflow-hidden">

          {/* Center Canvas (Grid) */}
          <div className="flex-1 p-6 overflow-y-auto">
            <div className="grid grid-cols-5 gap-4 auto-rows-min">
              {images.map((img) => (
                <Dialog key={img.seed}>
                  <DialogTrigger asChild>
                    <div className={`relative aspect-square group cursor-pointer rounded-lg overflow-hidden border-2 transition-all ${img.is_best ? 'border-green-500 shadow-[0_0_15px_rgba(34,197,94,0.3)]' : 'border-slate-800 hover:border-slate-600'}`}>
                      <img src={`http://localhost:8000${img.url}`} alt={`Seed ${img.seed}`} className="w-full h-full object-cover transition-transform group-hover:scale-110" />
                      <div className="absolute inset-x-0 bottom-0 bg-black/60 backdrop-blur-sm p-1 text-center">
                        <span className={`text-xs font-mono font-bold ${img.is_best ? 'text-green-400' : 'text-white'}`}>
                          {img.score.toFixed(4)}
                        </span>
                      </div>
                      {img.is_best && (
                        <div className="absolute top-2 right-2 bg-green-500 text-black text-[10px] font-bold px-1.5 py-0.5 rounded">
                          BEST
                        </div>
                      )}
                    </div>
                  </DialogTrigger>
                  <DialogContent className="max-w-3xl bg-slate-900 border-slate-800">
                    <img src={`http://localhost:8000${img.url}`} className="w-full h-auto rounded-lg" />
                    <div className="text-center font-mono text-slate-400 mt-2">
                      Seed: {img.seed} | Score: {img.score.toFixed(4)}
                    </div>
                  </DialogContent>
                </Dialog>
              ))}

              {/* Placeholders */}
              {images.length === 0 && !isGenerating && (
                <div className="col-span-5 row-span-2 h-64 flex flex-col items-center justify-center text-slate-700 border-2 border-dashed border-slate-800 rounded-xl">
                  <Sparkles className="w-8 h-8 mb-2 opacity-50" />
                  <p>Ready to generate</p>
                </div>
              )}
            </div>
          </div>

          {/* Right Status Panel */}
          <div className="w-96 border-l border-slate-800 flex flex-col bg-slate-900/20">
            <div className="p-4 border-b border-slate-800">
              <h2 className="font-bold text-sm text-slate-200">System Logs</h2>
            </div>

            <ScrollArea className="flex-1 p-4">
              <div className="space-y-2 font-mono text-xs">
                {logs.map((log, i) => (
                  <div key={i} className={`p-2 rounded ${log.highlight ? 'bg-blue-500/10 border border-blue-500/20 text-blue-200' : 'text-slate-400 bg-slate-900/50'}`}>
                    <span className="opacity-50 mr-2">[{new Date().toLocaleTimeString()}]</span>
                    {log.message}
                  </div>
                ))}
                <div ref={logsEndRef} />
              </div>
            </ScrollArea>

            {/* Eval Breakdown */}
            {evalResult && (
              <div className="h-64 border-t border-slate-800 p-4 bg-slate-900/50 overflow-y-auto">
                <h3 className="font-bold text-sm mb-3 flex items-center justify-between">
                  Evaluations
                  {evalResult.passed ? <Badge className="bg-green-600">PASS</Badge> : <Badge variant="destructive">FAIL</Badge>}
                </h3>

                <div className="space-y-2 text-xs">
                  <div className="grid grid-cols-2 gap-2">
                    <StatusRow label="Objects" passed={evalResult.breakdown.objects} />
                    <StatusRow label="Background" passed={evalResult.breakdown.background} />
                    <StatusRow label="Lighting" passed={evalResult.breakdown.lighting} />
                    <StatusRow label="Aesthetics" passed={evalResult.breakdown.aesthetics} />
                    <StatusRow label="Photo Specs" passed={evalResult.breakdown.photo} />
                    <StatusRow label="Style" passed={evalResult.breakdown.style} />
                  </div>

                  <div className="mt-4 pt-2 border-t border-slate-700">
                    <p className="font-semibold text-slate-300 mb-1">Alignment</p>
                    <StatusRow label="Subject" passed={evalResult.breakdown.alignment_subject} />
                    <StatusRow label="Elements" passed={evalResult.breakdown.alignment_elements} />
                  </div>

                  {!evalResult.passed && (
                    <div className="mt-2 text-red-400 italic">
                      Feedback: {evalResult.quality_feedback} {evalResult.alignment_feedback}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Bottom Input */}
        <div className="p-6 border-t border-slate-800 bg-slate-900/50">
          <div className="flex gap-4 max-w-5xl mx-auto">
            <div className="relative flex-1">
              <textarea
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit();
                  }
                }}
                placeholder="Describe your scene in detail..."
                className="w-full bg-slate-950 border border-slate-700 rounded-lg p-4 h-24 resize-none focus:ring-2 focus:ring-blue-600 focus:border-transparent text-sm"
                disabled={isGenerating}
              />
            </div>
            <Button onClick={handleSubmit} disabled={isGenerating || !prompt} className="h-24 w-32 bg-blue-600 hover:bg-blue-700 flex flex-col gap-2">
              {isGenerating ? (
                <>Running...</>
              ) : (
                <>Generate <Send className="w-4 h-4" /></>
              )}
            </Button>
          </div>
        </div>

      </div>
    </div>
  );
}

function StatusRow({ label, passed }: { label: string, passed: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-slate-400">{label}</span>
      {passed ? (
        <CheckCircle2 className="w-4 h-4 text-green-500" />
      ) : (
        <XCircle className="w-4 h-4 text-red-500" />
      )}
    </div>
  );
}
