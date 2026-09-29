import { motion } from "framer-motion";
import { formatDate } from "../lib/format";
import type { SampleScene } from "../lib/types";

const CONFIG_BADGE: Record<string, string> = {
  single: "1 image",
  cross_modal_pair: "optical + SAR",
  bi_temporal_pair: "2 dates",
  incomplete: "incomplete",
};

interface Props {
  scenes: SampleScene[];
  loading: boolean;
  error: string | null;
  loadingKey: string | null;
  activeKey: string | null;
  onLoad: (scene: SampleScene) => void;
}

function SceneCard({
  scene,
  busy,
  active,
  disabled,
  onLoad,
}: {
  scene: SampleScene;
  busy: boolean;
  active: boolean;
  disabled: boolean;
  onLoad: () => void;
}) {
  const assets = Object.values(scene.assets).filter(Boolean);
  const cached = assets.length > 0;
  const simulated = assets.some((asset) => asset?.provenance.is_simulated);

  // Real acquisitions are the credibility of the whole demo, so the dates and
  // the satellite are shown up front rather than buried in a tooltip.
  const dates = assets
    .map((asset) => asset?.provenance.acquisition_date)
    .filter((value): value is string => Boolean(value))
    .sort();
  const collections = Array.from(
    new Set(assets.map((asset) => asset?.provenance.collection).filter(Boolean)),
  );

  return (
    <div
      className={`rounded-lg border px-3 py-2.5 transition ${
        active
          ? "border-signal/60 bg-signal/10"
          : cached
            ? "border-edge bg-hull/40 hover:border-edge-hi"
            : "border-edge/60 bg-hull/20"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p
            className={`truncate text-xs font-medium ${
              cached ? "text-ink" : "text-ink-faint"
            }`}
          >
            {scene.title}
          </p>
          <p className="truncate text-[10px] text-ink-faint">{scene.place}</p>
        </div>
        <span className="shrink-0 rounded border border-edge-hi px-1.5 py-0.5 text-[9px] uppercase tracking-wide text-ink-faint">
          {CONFIG_BADGE[scene.configuration] ?? scene.configuration}
        </span>
      </div>

      {cached ? (
        <>
          <p className="tabular mt-1.5 text-[10px] text-ink-dim">
            {dates.map((date) => formatDate(date)).join("  \u2192  ")}
          </p>
          <p className="mt-0.5 text-[10px] text-ink-faint">
            {collections.join(" + ")}
            {simulated && (
              <span className="ml-1.5 rounded bg-uncertain/15 px-1 py-0.5 text-uncertain">
                includes simulated asset
              </span>
            )}
          </p>
          <button
            type="button"
            onClick={onLoad}
            disabled={disabled}
            className="mt-2 w-full rounded-md border border-edge px-2 py-1 text-[10px] text-ink-dim transition hover:border-signal/50 hover:text-signal disabled:cursor-not-allowed disabled:opacity-50"
          >
            {busy ? "Loading\u2026" : active ? "Loaded" : "Load this scene"}
          </button>
        </>
      ) : (
        <p className="mt-1.5 text-[10px] leading-snug text-ink-faint">
          Not cached. Run{" "}
          <code className="tabular text-ink-dim">scripts/fetch_samples.py</code> for
          real imagery, or{" "}
          <code className="tabular text-ink-dim">scripts/make_synthetic.py</code> to
          work offline.
        </p>
      )}
    </div>
  );
}

export function SampleGallery({
  scenes,
  loading,
  error,
  loadingKey,
  activeKey,
  onLoad,
}: Props) {
  return (
    <section className="panel p-4" aria-label="Sample scenes">
      <div className="flex items-baseline justify-between">
        <h2 className="text-[10px] uppercase tracking-wider text-ink-faint">
          Sample scenes
        </h2>
        <span className="text-[10px] text-ink-faint">Copernicus, over India</span>
      </div>

      {loading && (
        <p className="mt-2 text-[11px] text-ink-faint">Reading the scene library…</p>
      )}
      {error && (
        <p role="alert" className="mt-2 text-[11px] text-disagree">
          {error}
        </p>
      )}

      <div className="mt-2.5 space-y-2">
        {scenes.map((scene, index) => (
          <motion.div
            key={scene.key}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05, duration: 0.35 }}
          >
            <SceneCard
              scene={scene}
              busy={loadingKey === scene.key}
              active={activeKey === scene.key}
              disabled={loadingKey !== null}
              onLoad={() => onLoad(scene)}
            />
          </motion.div>
        ))}
      </div>
    </section>
  );
}
