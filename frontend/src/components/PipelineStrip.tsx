import { motion } from "framer-motion";

/** The SatQuery ARC pipeline: ingest an archive, search it, verify the change. */
export const PIPELINE_STAGES = [
  { id: "ingest", label: "Ingest", hint: "GeoTIFF / COG, metadata & checksums" },
  { id: "quality", label: "Quality Gate", hint: "Cloud, haze, CRS, co-registration" },
  { id: "embed", label: "Tile & Embed", hint: "RemoteCLIP / DINOv2 - ONNX on CPU" },
  { id: "index", label: "Hybrid Index", hint: "pgvector HNSW + place / time / sensor" },
  { id: "search", label: "Search", hint: "ANN, re-rank, clusters" },
  { id: "verify", label: "Verify Change", hint: "Time-series + confounder firewall" },
  { id: "review", label: "Review", hint: "Before/after, confirm / reject" },
] as const;

export type StageId = (typeof PIPELINE_STAGES)[number]["id"];

/**
 * Horizontal stage rail. On the landing page it animates in as a preview of the
 * pipeline; the Studio reuses the same stage list for the live execution graph
 * so the two views never drift apart.
 */
export function PipelineStrip({ delay = 0 }: { delay?: number }) {
  return (
    <ol className="flex flex-wrap items-stretch gap-1.5" aria-label="Analysis pipeline">
      {PIPELINE_STAGES.map((stage, index) => (
        <motion.li
          key={stage.id}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{
            delay: delay + index * 0.07,
            duration: 0.5,
            ease: [0.16, 1, 0.3, 1],
          }}
          className="group relative flex-1 basis-[130px]"
        >
          <div className="panel h-full px-3 py-2.5 transition-colors group-hover:border-edge-hi">
            <div className="flex items-baseline gap-1.5">
              <span className="tabular text-[10px] text-signal/70">
                {String(index + 1).padStart(2, "0")}
              </span>
              <span className="text-xs font-medium text-ink">{stage.label}</span>
            </div>
            <p className="mt-1 text-[10px] leading-snug text-ink-faint">{stage.hint}</p>
          </div>
        </motion.li>
      ))}
    </ol>
  );
}
