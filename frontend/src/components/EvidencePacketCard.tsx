import { motion } from "framer-motion";
import { useMemo, useState } from "react";
import { api } from "../lib/api";
import { formatBytes } from "../lib/format";
import { usePrefersReducedMotion } from "../lib/useReducedMotion";
import type {
  EvidencePacket,
  PacketFigure,
  PacketFile,
  PacketFileRole,
} from "../lib/types";

const ROLE_ORDER: PacketFileRole[] = [
  "report",
  "figure",
  "trace",
  "measurements",
  "findings",
  "overlay",
  "manifest",
  "archive",
];

const ROLE_LABEL: Record<PacketFileRole, string> = {
  report: "Report",
  figure: "Map figure",
  trace: "Trace",
  measurements: "Measurements",
  findings: "Vectors",
  overlay: "Overlay",
  manifest: "Manifest",
  archive: "Archive",
};

/**
 * The run as files you can take away.
 *
 * The download buttons are the obvious part. The part that matters is the
 * attribution line: the report states how many of its own figures resolve to a
 * measurement or a declared threshold, and the figure list below shows the
 * source of each one. An export that cannot be checked is just a screenshot with
 * extra steps.
 */
export function EvidencePacketCard({
  packet,
  sessionId,
}: {
  packet: EvidencePacket | null;
  sessionId: string;
}) {
  const reduceMotion = usePrefersReducedMotion();
  const [showFiles, setShowFiles] = useState(false);
  const [showFigures, setShowFigures] = useState(false);

  const grouped = useMemo(() => groupByRole(packet?.files ?? []), [packet]);
  const measured = useMemo(
    () => (packet?.figures ?? []).filter((figure) => figure.measurement_key),
    [packet],
  );

  // Nullish, not strictly null: an absent key arrives as undefined and sailed
  // past `=== null` into `packet.files` below.
  if (packet == null) return null;

  const href = (file: PacketFile) =>
    api.packetFileUrl(sessionId, packet.run_id, file.filename);
  const report = packet.files.find((file) => file.role === "report") ?? null;
  const totalBytes = packet.files.reduce(
    (sum, file) => sum + file.size_bytes,
    0,
  );

  return (
    <motion.section
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: reduceMotion ? 0 : 0.45, ease: [0.16, 1, 0.3, 1] }}
      className="panel panel-raised overflow-hidden"
      aria-label="Evidence packet"
    >
      <div className="border-b border-edge px-4 py-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <p className="text-[10px] uppercase tracking-wider text-ink-faint">
            Take the evidence with you
          </p>
          <p className="tabular text-[10px] text-ink-faint">
            {packet.files.length} files &middot; {formatBytes(totalBytes)}
          </p>
        </div>
        <p className="mt-1 text-[11px] leading-snug text-ink-dim">
          The same run as files that outlive this tab: a report that reads on its
          own, the trace, the measurements as a table, and the footprints as
          vectors.
        </p>
      </div>

      <div className="flex flex-wrap gap-2 border-b border-edge px-4 py-3">
        {packet.archive && (
          <a
            href={href(packet.archive)}
            download
            className="rounded-lg bg-signal px-3 py-1.5 text-[11px] font-semibold text-void transition hover:brightness-110"
          >
            Download everything
            <span className="tabular ml-1.5 font-normal opacity-70">
              {formatBytes(packet.archive.size_bytes)}
            </span>
          </a>
        )}
        {report && (
          <a
            href={href(report)}
            download
            className="rounded-lg border border-edge-hi bg-hull/60 px-3 py-1.5 text-[11px] font-semibold text-ink transition hover:border-signal/50 hover:text-signal"
          >
            Report
            <span className="tabular ml-1.5 font-normal text-ink-faint">
              {formatBytes(report.size_bytes)}
            </span>
          </a>
        )}
        {ROLE_ORDER.filter(
          (role) => role !== "report" && role !== "archive" && grouped[role],
        ).map((role) => {
          const first = grouped[role]![0];
          const count = grouped[role]!.length;
          return (
            <a
              key={role}
              href={href(first)}
              download
              className="rounded-lg border border-edge bg-hull/40 px-2.5 py-1.5 text-[11px] text-ink-dim transition hover:border-signal/40 hover:text-signal"
              title={first.description}
            >
              {ROLE_LABEL[role]}
              {count > 1 && (
                <span className="tabular ml-1 text-[9px] text-ink-faint">
                  1 of {count}
                </span>
              )}
            </a>
          );
        })}
      </div>

      <div
        className={`border-b border-edge px-4 py-3 ${
          packet.audit.passed ? "bg-agree/5" : "bg-disagree/5"
        }`}
      >
        <p
          className={`text-[10px] uppercase tracking-wider ${
            packet.audit.passed ? "text-agree" : "text-disagree"
          }`}
        >
          {packet.audit.passed
            ? "Every figure in the report is attributed"
            : "Some figures could not be attributed"}
        </p>
        <p className="tabular mt-1 text-[11px] text-ink-dim">
          {packet.audit.traced} of {packet.audit.checked} figures trace to a
          measurement, a declared threshold, or the imagery&rsquo;s own metadata.
        </p>
        <p className="mt-1 text-[10px] leading-snug text-ink-faint">
          {packet.audit.note}
        </p>
        {!packet.audit.passed && (
          <ul className="mt-1.5 space-y-0.5">
            {packet.audit.untraceable.map((item) => (
              <li key={item} className="tabular text-[10px] text-disagree">
                {item}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="border-b border-edge">
        <button
          type="button"
          onClick={() => setShowFigures((value) => !value)}
          aria-expanded={showFigures}
          className="w-full px-4 py-2 text-left text-[10px] uppercase tracking-wider text-ink-faint transition hover:text-signal"
        >
          {showFigures ? "Hide" : "Show"} where each of the{" "}
          {measured.length} measured figures came from
        </button>
        {showFigures && (
          <ul className="space-y-1 px-4 pb-3">
            {measured.map((figure, index) => (
              <FigureRow key={`${figure.measurement_key}-${index}`} figure={figure} />
            ))}
          </ul>
        )}
      </div>

      <div className="border-b border-edge">
        <button
          type="button"
          onClick={() => setShowFiles((value) => !value)}
          aria-expanded={showFiles}
          className="w-full px-4 py-2 text-left text-[10px] uppercase tracking-wider text-ink-faint transition hover:text-signal"
        >
          {showFiles ? "Hide" : "Show"} all {packet.files.length} files with
          their checksums
        </button>
        {showFiles && (
          <ul className="space-y-1 px-4 pb-3">
            {packet.files.map((file) => (
              <li
                key={file.filename}
                className="border-l-2 border-edge-hi pl-2.5"
              >
                <div className="flex flex-wrap items-baseline justify-between gap-x-2">
                  <a
                    href={href(file)}
                    download
                    className="tabular text-[11px] text-signal hover:underline"
                  >
                    {file.filename}
                  </a>
                  <span className="tabular text-[10px] text-ink-faint">
                    {formatBytes(file.size_bytes)}
                  </span>
                </div>
                <p className="text-[10px] leading-snug text-ink-faint">
                  {file.description}
                </p>
                <p className="tabular text-[9px] text-ink-faint/70">
                  sha256 {file.sha256.slice(0, 32)}&hellip;
                </p>
              </li>
            ))}
          </ul>
        )}
      </div>

      {packet.omissions.length > 0 && (
        <div className="px-4 py-3">
          <p className="text-[10px] uppercase tracking-wider text-ink-faint">
            What this packet does not contain
          </p>
          <ul className="mt-1.5 space-y-1">
            {packet.omissions.map((omission) => (
              <li
                key={omission}
                className="flex gap-2 text-[10px] leading-snug text-ink-faint"
              >
                <span className="text-uncertain" aria-hidden="true">
                  &middot;
                </span>
                <span>{omission}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </motion.section>
  );
}

function FigureRow({ figure }: { figure: PacketFigure }) {
  return (
    <li className="border-l-2 border-agree/60 pl-2.5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-2">
        <p className="tabular text-[11px] text-signal">{figure.display}</p>
        <p className="text-[10px] text-ink-faint">{figure.where}</p>
      </div>
      <p className="tabular text-[9px] leading-snug text-ink-faint">
        {figure.measurement_key} &middot; {figure.source_tool} v
        {figure.source_version} &middot; {figure.formula}
      </p>
    </li>
  );
}

function groupByRole(files: PacketFile[]) {
  const grouped: Partial<Record<PacketFileRole, PacketFile[]>> = {};
  for (const file of files) {
    (grouped[file.role] ??= []).push(file);
  }
  return grouped;
}
