import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { MapLayer } from "./types";

interface UseLayersResult {
  layers: MapLayer[];
  loading: boolean;
  error: string | null;
  reload: () => void;
}

/**
 * Fetches the map layers a run rendered.
 *
 * The server writes them before it emits the closing event, so fetching once the
 * run reaches a terminal status is enough: there is no polling and no window in
 * which the map is told about a layer that does not exist yet.
 */
export function useLayers(
  sessionId: string | undefined,
  runId: string | undefined,
  ready: boolean,
): UseLayersResult {
  const [layers, setLayers] = useState<MapLayer[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const load = useCallback(async () => {
    if (!sessionId || !runId) return;
    setLoading(true);
    try {
      const next = await api.runLayers(sessionId, runId);
      if (!mounted.current) return;
      setLayers(next);
      setError(null);
    } catch {
      if (mounted.current) setError("Could not load the map layers.");
    } finally {
      if (mounted.current) setLoading(false);
    }
  }, [runId, sessionId]);

  useEffect(() => {
    setLayers([]);
    setError(null);
    if (ready) void load();
  }, [load, ready]);

  return { layers, loading, error, reload: () => void load() };
}
