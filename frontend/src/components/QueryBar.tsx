import { useState } from "react";
import type { SampleScene } from "../lib/types";

interface Props {
  suggestions: string[];
  disabled: boolean;
  busy: boolean;
  onSubmit: (query: string) => void;
  activeScene: SampleScene | null;
}

/**
 * The question input. Suggestions come from the loaded sample scene, which is
 * both a convenience and a way to keep a recorded demo on script.
 */
export function QueryBar({
  suggestions,
  disabled,
  busy,
  onSubmit,
  activeScene,
}: Props) {
  const [value, setValue] = useState("");

  const submit = () => {
    const query = value.trim();
    if (!query || disabled || busy) return;
    onSubmit(query);
  };

  return (
    <section className="panel p-4" aria-label="Ask a question">
      <div className="flex items-baseline justify-between">
        <h2 className="text-[10px] uppercase tracking-wider text-ink-faint">
          Ask about this imagery
        </h2>
        {activeScene && (
          <span className="text-[10px] text-ink-faint">{activeScene.demo}</span>
        )}
      </div>

      <div className="mt-2.5 flex gap-2">
        <label className="sr-only" htmlFor="satquery-question">
          Your question about the loaded imagery
        </label>
        <input
          id="satquery-question"
          type="text"
          value={value}
          disabled={disabled}
          placeholder={
            disabled
              ? "Load imagery first"
              : "Has the built-up area increased since 2020?"
          }
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") submit();
          }}
          className="min-w-0 flex-1 rounded-lg border border-edge bg-hull/60 px-3 py-2 text-xs text-ink placeholder:text-ink-faint focus:border-signal/60 focus:outline-none disabled:opacity-50"
        />
        <button
          type="button"
          onClick={submit}
          disabled={disabled || busy || !value.trim()}
          className="shrink-0 rounded-lg bg-signal px-4 py-2 text-xs font-semibold text-void transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy ? "Drafting\u2026" : "Draft contract"}
        </button>
      </div>

      {suggestions.length > 0 && (
        <div className="mt-2.5 flex flex-wrap gap-1.5">
          {suggestions.map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              disabled={disabled || busy}
              onClick={() => {
                setValue(suggestion);
                onSubmit(suggestion);
              }}
              className="rounded-md border border-edge bg-hull/40 px-2 py-1 text-left text-[10px] leading-snug text-ink-dim transition hover:border-signal/40 hover:text-signal disabled:opacity-40"
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}

      <p className="mt-2.5 text-[10px] leading-snug text-ink-faint">
        Nothing runs until you approve the contract. The language model reads the
        question and plans; it does not measure.
      </p>
    </section>
  );
}
