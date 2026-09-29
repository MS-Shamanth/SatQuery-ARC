import type {
  AnalysisContract,
  EvidencePacket,
  HealthResponse,
  ImageRole,
  IngestedImage,
  MapLayer,
  ProvenanceResponse,
  ReadinessReport,
  RunTrace,
  SampleManifest,
  SessionRecord,
  SessionToolAvailability,
  ToolDescriptor,
} from "./types";

/** Vite proxies /api to the FastAPI backend, so relative paths work in dev. */
const BASE = "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly body?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: { Accept: "application/json", ...(init?.headers ?? {}) },
      ...init,
    });
  } catch (cause) {
    throw new ApiError(
      `Cannot reach the SatQuery backend at ${BASE}${path}. Is uvicorn running?`,
      0,
      cause,
    );
  }

  if (!response.ok) {
    let body: unknown;
    try {
      body = await response.json();
    } catch {
      body = await response.text().catch(() => undefined);
    }
    const detail =
      typeof body === "object" && body !== null && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : response.statusText;
    throw new ApiError(detail, response.status, body);
  }

  return (await response.json()) as T;
}

/**
 * Upload with progress. fetch() cannot report upload progress, so this uses
 * XMLHttpRequest: on a large GeoTIFF the progress bar is the difference between
 * "working" and "frozen" from the user's point of view.
 */
function upload<T>(
  path: string,
  form: FormData,
  onProgress?: (fraction: number) => void,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}${path}`);
    xhr.responseType = "json";

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress(event.loaded / event.total);
      }
    };

    xhr.onload = () => {
      const body = xhr.response as { detail?: string } | null;
      if (xhr.status >= 200 && xhr.status < 300) {
        onProgress?.(1);
        resolve(body as T);
      } else {
        reject(new ApiError(body?.detail ?? xhr.statusText, xhr.status, body));
      }
    };
    xhr.onerror = () =>
      reject(new ApiError("Network error during upload", 0));
    xhr.onabort = () => reject(new ApiError("Upload cancelled", 0));

    xhr.send(form);
  });
}

export const api = {
  health: (refresh = false) =>
    request<HealthResponse>(`/health${refresh ? "?refresh=true" : ""}`),
  provenance: () => request<ProvenanceResponse>("/provenance"),

  createSession: () => request<SessionRecord>("/sessions", { method: "POST" }),
  getSession: (id: string) => request<SessionRecord>(`/sessions/${id}`),
  deleteSession: (id: string) =>
    request<{ ok: boolean }>(`/sessions/${id}`, { method: "DELETE" }),

  uploadImage: (
    sessionId: string,
    role: ImageRole,
    file: File,
    onProgress?: (fraction: number) => void,
  ) => {
    const form = new FormData();
    form.append("role", role);
    form.append("file", file);
    return upload<IngestedImage>(`/sessions/${sessionId}/images`, form, onProgress);
  },

  removeImage: (sessionId: string, role: ImageRole) =>
    request<{ ok: boolean }>(`/sessions/${sessionId}/images/${role}`, {
      method: "DELETE",
    }),

  readiness: (sessionId: string) =>
    request<ReadinessReport>(`/sessions/${sessionId}/readiness`),

  samples: () => request<SampleManifest>("/samples"),
  loadSample: (key: string) =>
    request<SessionRecord>(`/samples/${key}/load`, { method: "POST" }),

  tools: () => request<ToolDescriptor[]>("/tools"),
  sessionTools: (sessionId: string) =>
    request<SessionToolAvailability>(`/sessions/${sessionId}/tools`),

  contract: (
    sessionId: string,
    query: string,
    options: { forceOffline?: boolean; refresh?: boolean } = {},
  ) =>
    request<AnalysisContract>(`/sessions/${sessionId}/contract`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        force_offline: options.forceOffline ?? false,
        refresh: options.refresh ?? false,
      }),
    }),

  /**
   * Approve a contract and execute it. Only the hash travels, so what runs is
   * necessarily the plan that was displayed.
   */
  startRun: (sessionId: string, contractHash: string) =>
    request<RunTrace>(`/sessions/${sessionId}/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ contract_hash: contractHash }),
    }),

  runTrace: (sessionId: string, runId: string) =>
    request<RunTrace>(`/sessions/${sessionId}/runs/${runId}`),

  cancelRun: (sessionId: string, runId: string) =>
    request<RunTrace>(`/sessions/${sessionId}/runs/${runId}/cancel`, {
      method: "POST",
    }),

  runLayers: (sessionId: string, runId: string) =>
    request<MapLayer[]>(`/sessions/${sessionId}/runs/${runId}/layers`),

  /** The packet's own manifest, read from disk so it describes the real files. */
  runPacket: (sessionId: string, runId: string) =>
    request<EvidencePacket>(`/sessions/${sessionId}/runs/${runId}/packet`),

  /** A direct download link, so the browser saves rather than navigates. */
  packetFileUrl: (sessionId: string, runId: string, filename: string) =>
    `${BASE}/sessions/${sessionId}/runs/${runId}/packet/${encodeURIComponent(
      filename,
    )}`,

  /** EventSource target. Replays from the first event, so joining late is safe. */
  runStreamUrl: (sessionId: string, runId: string) =>
    `${BASE}/sessions/${sessionId}/runs/${runId}/stream`,

  /** Cache-busted so a replaced image does not show a stale preview. */
  thumbnailUrl: (sessionId: string, role: ImageRole, version?: string) =>
    `${BASE}/sessions/${sessionId}/thumb/${role}${
      version ? `?v=${encodeURIComponent(version.slice(0, 12))}` : ""
    }`,
};
