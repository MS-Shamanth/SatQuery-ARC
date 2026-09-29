import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api";
import type { SampleScene } from "./types";

interface UseSamplesResult {
  scenes: SampleScene[];
  loading: boolean;
  error: string | null;
  loadingKey: string | null;
  /** Loads a cached scene and returns the session it created. */
  load: (key: string) => Promise<Awaited<ReturnType<typeof api.loadSample>> | null>;
}

/** Reads the curated scene library and loads a scene into a fresh session. */
export function useSamples(): UseSamplesResult {
  const [scenes, setScenes] = useState<SampleScene[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [loadingKey, setLoadingKey] = useState<string | null>(null);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    (async () => {
      try {
        const manifest = await api.samples();
        if (!mounted.current) return;
        setScenes(Object.values(manifest.scenes));
        setError(null);
      } catch (cause) {
        if (!mounted.current) return;
        setError(
          cause instanceof ApiError
            ? cause.message
            : "Could not read the sample scene library",
        );
      } finally {
        if (mounted.current) setLoading(false);
      }
    })();
    return () => {
      mounted.current = false;
    };
  }, []);

  const load = useCallback(async (key: string) => {
    setLoadingKey(key);
    try {
      const session = await api.loadSample(key);
      if (!mounted.current) return null;
      setError(null);
      return session;
    } catch (cause) {
      if (!mounted.current) return null;
      setError(
        cause instanceof ApiError ? cause.message : `Could not load sample '${key}'`,
      );
      return null;
    } finally {
      if (mounted.current) setLoadingKey(null);
    }
  }, []);

  return { scenes, loading, error, loadingKey, load };
}
