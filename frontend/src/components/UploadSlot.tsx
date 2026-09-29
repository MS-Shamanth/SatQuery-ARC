import { AnimatePresence, motion } from "framer-motion";
import { useRef, useState } from "react";
import type { SlotSpec } from "../lib/modes";

interface Props {
  slot: SlotSpec;
  filled: boolean;
  filename?: string;
  progress: number | null;
  error: string | null;
  disabled?: boolean;
  onFile: (file: File) => void;
}

const ACCEPT = ".tif,.tiff,.png,.jpg,.jpeg,image/tiff,image/png,image/jpeg";

export function UploadSlot({
  slot,
  filled,
  filename,
  progress,
  error,
  disabled = false,
  onFile,
}: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);

  const busy = progress !== null && progress < 1;

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    setDragging(false);
    if (disabled) return;
    const file = event.dataTransfer.files?.[0];
    if (file) onFile(file);
  };

  const borderClass = error
    ? "border-disagree/60"
    : dragging
      ? "border-signal"
      : filled
        ? "border-agree/40"
        : "border-edge";

  return (
    <div
      onDragOver={(event) => {
        event.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={`relative overflow-hidden rounded-xl border border-dashed ${borderClass} bg-hull/50 transition-colors`}
    >
      {/* Radar sweep while the file is being read */}
      <AnimatePresence>
        {busy && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="pointer-events-none absolute inset-0"
            aria-hidden="true"
          >
            <div
              className="absolute inset-y-0 w-1/3 animate-[var(--animate-drift)]"
              style={{
                background:
                  "linear-gradient(90deg, transparent, rgba(34,211,238,0.12), transparent)",
                animationDuration: "1.6s",
              }}
            />
          </motion.div>
        )}
      </AnimatePresence>

      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        disabled={disabled || busy}
        aria-label={`${slot.title}. ${filled ? `Loaded ${filename}. ` : ""}Choose a file.`}
        className="relative flex w-full flex-col items-center gap-1.5 px-4 py-6 text-center transition disabled:cursor-not-allowed disabled:opacity-60"
      >
        <span
          className={`flex h-8 w-8 items-center justify-center rounded-lg border ${
            filled ? "border-agree/40 bg-agree/10" : "border-edge bg-panel"
          }`}
          aria-hidden="true"
        >
          {filled ? (
            <svg width="14" height="14" viewBox="0 0 14 14">
              <path
                d="M2 7.5 5.5 11 12 3.5"
                fill="none"
                stroke="var(--color-agree)"
                strokeWidth="1.8"
                strokeLinecap="round"
              />
            </svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 14 14">
              <path
                d="M7 11V3M3.5 6.5 7 3l3.5 3.5"
                fill="none"
                stroke="var(--color-ink-faint)"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
            </svg>
          )}
        </span>

        <span className="text-xs font-medium text-ink">{slot.title}</span>
        <span className="text-[10px] text-ink-faint">
          {filled && filename ? filename : slot.hint}
        </span>

        {busy && (
          <span className="tabular mt-1 text-[10px] text-signal">
            reading {Math.round((progress ?? 0) * 100)}%
          </span>
        )}
      </button>

      {progress !== null && (
        <div className="h-0.5 w-full bg-edge">
          <motion.div
            className={`h-full ${progress >= 1 ? "bg-agree" : "bg-signal"}`}
            initial={{ width: 0 }}
            animate={{ width: `${Math.round(progress * 100)}%` }}
            transition={{ duration: 0.25 }}
          />
        </div>
      )}

      {error && (
        <p
          role="alert"
          className="border-t border-disagree/30 bg-disagree/5 px-3 py-2 text-[11px] leading-snug text-disagree"
        >
          {error}
        </p>
      )}

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onFile(file);
          event.target.value = "";
        }}
      />
    </div>
  );
}
