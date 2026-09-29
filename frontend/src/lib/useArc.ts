import { useCallback, useEffect, useState } from "react";
import {
  arc,
  type ArchiveOverview,
  type ChangeVerification,
  type ReviewState,
  type SearchQuery,
  type SearchResponse,
} from "./arc";

export function useArchive() {
  const [overview, setOverview] = useState<ArchiveOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    arc
      .archive()
      .then((o) => active && setOverview(o))
      .catch((e) => active && setError(e.message))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, []);

  return { overview, loading, error };
}

export function useSearch() {
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async (query: SearchQuery) => {
    setBusy(true);
    setError(null);
    try {
      const res = await arc.search(query);
      setResponse(res);
      return res;
    } catch (e) {
      setError((e as Error).message);
      return null;
    } finally {
      setBusy(false);
    }
  }, []);

  return { response, busy, error, run, setResponse };
}

export function useVerification() {
  const [verification, setVerification] = useState<ChangeVerification | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [aoiKey, setAoiKey] = useState<string | null>(null);

  const verify = useCallback(async (key: string, refresh = false) => {
    setBusy(true);
    setError(null);
    setAoiKey(key);
    try {
      const v = await arc.verify(key, refresh);
      setVerification(v);
      return v;
    } catch (e) {
      setError((e as Error).message);
      setVerification(null);
      return null;
    } finally {
      setBusy(false);
    }
  }, []);

  return { verification, busy, error, aoiKey, verify };
}

export function useReview() {
  const [state, setState] = useState<ReviewState | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setState(await arc.review());
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const decide = useCallback(
    async (itemId: string, decision: "confirm" | "reject" | "reopen") => {
      await arc.decide(itemId, decision);
      await refresh();
    },
    [refresh],
  );

  return { state, error, refresh, decide };
}
