import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api";
import type { HealthResponse } from "./types";

interface UseHealthResult {
  health: HealthResponse | null;
  error: string | null;
  loading: boolean;
  refresh: (deep?: boolean) => Promise<void>;
}

/**
 * Polls /api/health so the UI always reflects what the system can currently do.
 * A backend that is not running surfaces as an error string rather than an
 * unhandled rejection, because during a recording that state must be visible.
 */
export function useHealth(pollMs = 30_000): UseHealthResult {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const mounted = useRef(true);

  const load = useCallback(async (deep = false) => {
    try {
      const next = await api.health(deep);
      if (!mounted.current) return;
      setHealth(next);
      setError(null);
    } catch (cause) {
      if (!mounted.current) return;
      setError(
        cause instanceof ApiError ? cause.message : "Unknown error contacting backend",
      );
    } finally {
      if (mounted.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    void load();
    const id = window.setInterval(() => void load(), pollMs);
    return () => {
      mounted.current = false;
      window.clearInterval(id);
    };
  }, [load, pollMs]);

  return { health, error, loading, refresh: load };
}
