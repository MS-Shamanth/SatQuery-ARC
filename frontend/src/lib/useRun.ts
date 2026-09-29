import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api";
import type { RunEvent, RunTrace, TraceStage } from "./types";

export interface LogLine {
  seq: number;
  at: string;
  text: string;
}

interface UseRunResult {
  trace: RunTrace | null;
  log: LogLine[];
  starting: boolean;
  streaming: boolean;
  error: string | null;
  start: (contractHash: string) => Promise<void>;
  cancel: () => Promise<void>;
  clear: () => void;
}

const MAX_LOG_LINES = 200;

/**
 * Folds the run's event stream into a live trace.
 *
 * Events replace stages and steps by id rather than patching fields, so the
 * client cannot drift out of step with the server: a missed event is corrected
 * by the next one that carries the same stage, and the closing event carries the
 * authoritative trace.
 */
export function useRun(sessionId: string | undefined): UseRunResult {
  const [trace, setTrace] = useState<RunTrace | null>(null);
  const [log, setLog] = useState<LogLine[]>([]);
  const [starting, setStarting] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sourceRef = useRef<EventSource | null>(null);
  const runIdRef = useRef<string | null>(null);

  const closeStream = useCallback(() => {
    sourceRef.current?.close();
    sourceRef.current = null;
    setStreaming(false);
  }, []);

  useEffect(() => closeStream, [closeStream]);

  // A trace belongs to the imagery it ran against.
  useEffect(() => {
    closeStream();
    runIdRef.current = null;
    setTrace(null);
    setLog([]);
    setError(null);
  }, [sessionId, closeStream]);

  const apply = useCallback((event: RunEvent) => {
    if (event.trace) setTrace(event.trace);

    if (event.stage) {
      const incoming: TraceStage = event.stage;
      setTrace((current) =>
        current
          ? {
              ...current,
              stages: current.stages.map((stage) =>
                stage.id === incoming.id ? incoming : stage,
              ),
            }
          : current,
      );
    }

    if (event.step) {
      const incoming = event.step;
      setTrace((current) => {
        if (!current) return current;
        return {
          ...current,
          stages: current.stages.map((stage) => {
            if (stage.id !== incoming.stage) return stage;
            const known = stage.steps.some((step) => step.id === incoming.id);
            return {
              ...stage,
              status: stage.status === "pending" ? "running" : stage.status,
              steps: known
                ? stage.steps.map((step) =>
                    step.id === incoming.id ? incoming : step,
                  )
                : [...stage.steps, incoming],
            };
          }),
        };
      });
    }

    if (event.tool_run) {
      const run = event.tool_run;
      setTrace((current) => {
        if (!current) return current;
        const known = current.tool_runs.some(
          (item) => item.tool === run.tool && item.version === run.version,
        );
        if (known) return current;
        return {
          ...current,
          tool_runs: [...current.tool_runs, run],
          measurements: run.ok
            ? [...current.measurements, ...run.measurements]
            : current.measurements,
          masks: run.ok ? [...current.masks, ...run.masks] : current.masks,
        };
      });
    }

    if (event.message) {
      setLog((lines) =>
        [...lines, { seq: event.seq, at: event.at, text: event.message }].slice(
          -MAX_LOG_LINES,
        ),
      );
    }
  }, []);

  const attach = useCallback(
    (runId: string) => {
      if (!sessionId) return;
      closeStream();
      const source = new EventSource(api.runStreamUrl(sessionId, runId));
      sourceRef.current = source;
      setStreaming(true);

      source.onmessage = (message) => {
        let event: RunEvent;
        try {
          event = JSON.parse(message.data) as RunEvent;
        } catch {
          return;
        }
        apply(event);
        if (event.type === "run.finished") {
          source.close();
          sourceRef.current = null;
          setStreaming(false);
        }
      };

      // The server closes the stream when the run ends, which the browser sees
      // as an error. Falling back to a plain read of the trace means a finished
      // run still shows its result instead of an empty panel.
      source.onerror = () => {
        source.close();
        sourceRef.current = null;
        setStreaming(false);
        void api
          .runTrace(sessionId, runId)
          .then(setTrace)
          .catch(() => setError("Lost the connection to the run."));
      };
    },
    [apply, closeStream, sessionId],
  );

  const start = useCallback(
    async (contractHash: string) => {
      if (!sessionId) return;
      setStarting(true);
      setError(null);
      setLog([]);
      try {
        const pending = await api.startRun(sessionId, contractHash);
        runIdRef.current = pending.run_id;
        setTrace(pending);
        attach(pending.run_id);
      } catch (cause) {
        setError(
          cause instanceof ApiError ? cause.message : "Could not start the run",
        );
      } finally {
        setStarting(false);
      }
    },
    [attach, sessionId],
  );

  const cancel = useCallback(async () => {
    const runId = runIdRef.current;
    if (!sessionId || !runId) return;
    try {
      await api.cancelRun(sessionId, runId);
    } catch {
      setError("Could not cancel the run.");
    }
  }, [sessionId]);

  const clear = useCallback(() => {
    closeStream();
    runIdRef.current = null;
    setTrace(null);
    setLog([]);
    setError(null);
  }, [closeStream]);

  return { trace, log, starting, streaming, error, start, cancel, clear };
}
