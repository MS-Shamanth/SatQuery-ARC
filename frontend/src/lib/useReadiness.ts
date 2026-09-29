import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api";
import type { ReadinessReport, SessionRecord } from "./types";

interface UseReadinessResult {
  report: ReadinessReport | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

/**
 * Runs the readiness gate whenever the session's imagery changes.
 *
 * Keyed on the session id plus the content hashes of the loaded images, so
 * replacing a file re-runs the gate while an unrelated re-render does not.
 */
export function useReadiness(
  session: SessionRecord | null,
  enabled: boolean,
): UseReadinessResult {
  const [report, setReport] = useState<ReadinessReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mounted = useRef(true);

  const signature = session
    ? `${session.session_id}:${Object.values(session.images)
        .map((image) => image?.sha256.slice(0, 12))
        .sort()
        .join(",")}`
    : "";

  const sessionId = session?.session_id;

  const run = useCallback(async () => {
    if (!sessionId) return;
    setLoading(true);
    try {
      const next = await api.readiness(sessionId);
      if (!mounted.current) return;
      setReport(next);
      setError(null);
    } catch (cause) {
      if (!mounted.current) return;
      setError(
        cause instanceof ApiError
          ? cause.message
          : "Could not evaluate data readiness",
      );
    } finally {
      if (mounted.current) setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    mounted.current = true;
    if (!enabled || !sessionId) {
      setReport(null);
      setError(null);
      return () => {
        mounted.current = false;
      };
    }
    void run();
    return () => {
      mounted.current = false;
    };
    // signature captures "the imagery actually changed"; run captures the id.
  }, [enabled, sessionId, signature, run]);

  return { report, loading, error, refresh: run };
}
