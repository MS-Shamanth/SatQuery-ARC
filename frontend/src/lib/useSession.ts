import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api";
import { MODES, modeFor } from "./modes";
import type { ImageRole, InputConfiguration, SessionRecord } from "./types";

type PerRole<T> = Partial<Record<ImageRole, T>>;

interface UseSessionResult {
  session: SessionRecord | null;
  mode: Exclude<InputConfiguration, "incomplete">;
  setMode: (id: Exclude<InputConfiguration, "incomplete">) => void;
  progress: PerRole<number>;
  errors: PerRole<string>;
  starting: boolean;
  startError: string | null;
  uploadTo: (role: ImageRole, file: File) => Promise<void>;
  removeFrom: (role: ImageRole) => Promise<void>;
  reset: () => Promise<void>;
  /** Take over a session created elsewhere, such as by loading a sample. */
  adopt: (record: SessionRecord) => void;
}

/**
 * Owns the upload session: creates it lazily, tracks per-slot upload progress
 * and per-slot errors, and clears slots that do not belong to the selected
 * input configuration so the derived configuration stays coherent.
 */
export function useSession(): UseSessionResult {
  const [session, setSession] = useState<SessionRecord | null>(null);
  const [mode, setModeState] =
    useState<Exclude<InputConfiguration, "incomplete">>("single");
  const [progress, setProgress] = useState<PerRole<number>>({});
  const [errors, setErrors] = useState<PerRole<string>>({});
  const [starting, setStarting] = useState(true);
  const [startError, setStartError] = useState<string | null>(null);
  const mounted = useRef(true);

  const start = useCallback(async () => {
    setStarting(true);
    try {
      const record = await api.createSession();
      if (!mounted.current) return;
      setSession(record);
      setStartError(null);
    } catch (cause) {
      if (!mounted.current) return;
      setStartError(
        cause instanceof ApiError ? cause.message : "Could not start a session",
      );
    } finally {
      if (mounted.current) setStarting(false);
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    void start();
    return () => {
      mounted.current = false;
    };
  }, [start]);

  const uploadTo = useCallback(
    async (role: ImageRole, file: File) => {
      if (!session) return;
      setErrors((prev) => ({ ...prev, [role]: undefined }));
      setProgress((prev) => ({ ...prev, [role]: 0 }));
      try {
        await api.uploadImage(session.session_id, role, file, (fraction) => {
          if (mounted.current) {
            // Cap the bar at 95% until the server responds: the remaining time
            // is rasterio reading the file, not bytes on the wire.
            setProgress((prev) => ({ ...prev, [role]: Math.min(fraction, 0.95) }));
          }
        });
        const refreshed = await api.getSession(session.session_id);
        if (!mounted.current) return;
        setSession(refreshed);
        setProgress((prev) => ({ ...prev, [role]: 1 }));
      } catch (cause) {
        if (!mounted.current) return;
        setErrors((prev) => ({
          ...prev,
          [role]: cause instanceof ApiError ? cause.message : "Upload failed",
        }));
        setProgress((prev) => ({ ...prev, [role]: undefined }));
      }
    },
    [session],
  );

  const removeFrom = useCallback(
    async (role: ImageRole) => {
      if (!session) return;
      try {
        await api.removeImage(session.session_id, role);
      } catch {
        // Slot may already be empty; the refresh below is the source of truth.
      }
      const refreshed = await api.getSession(session.session_id);
      if (!mounted.current) return;
      setSession(refreshed);
      setProgress((prev) => ({ ...prev, [role]: undefined }));
      setErrors((prev) => ({ ...prev, [role]: undefined }));
    },
    [session],
  );

  const setMode = useCallback(
    (id: Exclude<InputConfiguration, "incomplete">) => {
      setModeState(id);
      const keep = new Set(modeFor(id).slots.map((slot) => slot.role));
      const toDrop = MODES.flatMap((m) => m.slots.map((s) => s.role)).filter(
        (role) => !keep.has(role),
      );
      if (session) {
        for (const role of new Set(toDrop)) {
          if (session.images[role]) void removeFrom(role);
        }
      }
      setProgress({});
      setErrors({});
    },
    [session, removeFrom],
  );

  const reset = useCallback(async () => {
    if (session) {
      try {
        await api.deleteSession(session.session_id);
      } catch {
        // Nothing to recover: a fresh session is created immediately below.
      }
    }
    setProgress({});
    setErrors({});
    setSession(null);
    await start();
  }, [session, start]);

  const adopt = useCallback((record: SessionRecord) => {
    setSession(record);
    setProgress({});
    setErrors({});
    setStartError(null);
    // The loaded scene dictates the configuration, so the selector follows the
    // data rather than the data being trimmed to match the selector.
    if (record.configuration !== "incomplete") {
      setModeState(record.configuration);
    }
  }, []);

  return {
    session,
    mode,
    setMode,
    progress,
    errors,
    starting,
    startError,
    uploadTo,
    removeFrom,
    reset,
    adopt,
  };
}
