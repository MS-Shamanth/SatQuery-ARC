import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api";
import type { AnalysisContract, SessionRecord } from "./types";

interface UseContractResult {
  contract: AnalysisContract | null;
  loading: boolean;
  error: string | null;
  draft: (query: string) => Promise<void>;
  discard: () => void;
}

/** Drafts a contract for a query and holds it until approved or discarded. */
export function useContract(session: SessionRecord | null): UseContractResult {
  const [contract, setContract] = useState<AnalysisContract | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  // A contract belongs to the imagery it was planned against, so changing the
  // session invalidates it rather than leaving a stale plan on screen.
  const sessionId = session?.session_id;
  useEffect(() => {
    setContract(null);
    setError(null);
  }, [sessionId]);

  const draft = useCallback(
    async (query: string) => {
      if (!sessionId) return;
      setLoading(true);
      setContract(null);
      try {
        const next = await api.contract(sessionId, query);
        if (!mounted.current) return;
        setContract(next);
        setError(null);
      } catch (cause) {
        if (!mounted.current) return;
        setError(
          cause instanceof ApiError
            ? cause.message
            : "Could not draft an analysis contract",
        );
      } finally {
        if (mounted.current) setLoading(false);
      }
    },
    [sessionId],
  );

  const discard = useCallback(() => {
    setContract(null);
    setError(null);
  }, []);

  return { contract, loading, error, draft, discard };
}
