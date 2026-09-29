import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api";
import type {
  SessionRecord,
  SessionToolAvailability,
  ToolDescriptor,
} from "./types";

interface UseToolsResult {
  tools: ToolDescriptor[];
  availability: SessionToolAvailability | null;
  loading: boolean;
  error: string | null;
}

/**
 * Reads the tool registry once, then re-checks availability whenever the
 * session's imagery changes, since what a tool can do depends on the bands the
 * input actually carries.
 */
export function useTools(session: SessionRecord | null): UseToolsResult {
  const [tools, setTools] = useState<ToolDescriptor[]>([]);
  const [availability, setAvailability] = useState<SessionToolAvailability | null>(
    null,
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    (async () => {
      try {
        const descriptors = await api.tools();
        if (!mounted.current) return;
        setTools(descriptors);
        setError(null);
      } catch (cause) {
        if (!mounted.current) return;
        setError(
          cause instanceof ApiError ? cause.message : "Could not read the tool registry",
        );
      } finally {
        if (mounted.current) setLoading(false);
      }
    })();
    return () => {
      mounted.current = false;
    };
  }, []);

  const sessionId = session?.session_id;
  const signature = session
    ? Object.values(session.images)
        .map((image) => image?.sha256.slice(0, 12))
        .sort()
        .join(",")
    : "";

  useEffect(() => {
    mounted.current = true;
    if (!sessionId || !signature) {
      setAvailability(null);
      return () => {
        mounted.current = false;
      };
    }
    (async () => {
      try {
        const next = await api.sessionTools(sessionId);
        if (mounted.current) setAvailability(next);
      } catch {
        // Availability is advisory; the registry list still renders without it.
        if (mounted.current) setAvailability(null);
      }
    })();
    return () => {
      mounted.current = false;
    };
  }, [sessionId, signature]);

  return { tools, availability, loading, error };
}
