import { AnimatePresence, motion } from "framer-motion";
import { Link } from "react-router-dom";
import { HealthPill } from "../components/HealthPill";
import { useEffect, useState } from "react";
import { AgentGraph } from "../components/AgentGraph";
import { CapabilityPanel } from "../components/CapabilityPanel";
import { ConfounderPanel } from "../components/ConfounderPanel";
import { DisagreementPanel } from "../components/DisagreementPanel";
import { MapWorkspace } from "../components/MapWorkspace";
import { RunTracePanel } from "../components/RunTracePanel";
import { VerdictCard } from "../components/VerdictCard";
import { RemedyOffers } from "../components/RemedyOffers";
import { EvidencePacketCard } from "../components/EvidencePacketCard";
import { EvidenceDisagreementPanel } from "../components/EvidenceDisagreementPanel";
import { PanelBoundary } from "../components/PanelBoundary";
import { TourButton } from "../components/GuidedTour";
import { ProfileMenu } from "../components/ProfileMenu";
import { useTourControls } from "../lib/tour";
import { useAuth } from "../lib/auth";
import {
  PresenterPanel,
  type PresenterPhase,
} from "../components/PresenterPanel";
import type { DemoScript } from "../lib/demos";
import { ContractCard } from "../components/ContractCard";
import { MetadataPanel } from "../components/MetadataPanel";
import { QueryBar } from "../components/QueryBar";
import { ReadinessPanel } from "../components/ReadinessPanel";
import { SampleGallery } from "../components/SampleGallery";
import { UploadSlot } from "../components/UploadSlot";
import { Wordmark } from "../components/Wordmark";
import { api } from "../lib/api";
import { MODES, modeFor } from "../lib/modes";
import { useContract } from "../lib/useContract";
import { useHealth } from "../lib/useHealth";
import { useLayers } from "../lib/useLayers";
import { useReadiness } from "../lib/useReadiness";
import { useRun } from "../lib/useRun";
import { useSamples } from "../lib/useSamples";
import { useSession } from "../lib/useSession";
import { useTools } from "../lib/useTools";
import type { RemedyOffer, SampleScene } from "../lib/types";

const CONFIG_LABEL: Record<string, string> = {
  single: "Single image",
  cross_modal_pair: "Cross-modal pair",
  bi_temporal_pair: "Bi-temporal pair",
  incomplete: "Awaiting input",
};

/**
 * What stands in for the findings while the run is still working.
 *
 * One placeholder rather than a row of empty panels. It names the stage in flight
 * so the wait is accounted for, and says plainly that nothing is being withheld:
 * the results are not ready, as opposed to being ready and unshown.
 */
function PendingFindings({ stage }: { stage: string | null }) {
  return (
    <section
      className="panel flex flex-col items-center justify-center gap-2 p-8 text-center"
      aria-label="Findings pending"
      role="status"
    >
      <div
        className="relative mb-1 h-10 w-10 rounded-full border border-edge"
        aria-hidden="true"
      >
        <div className="absolute inset-0 animate-[var(--animate-sweep)]">
          <div
            className="absolute left-1/2 top-1/2 h-1/2 w-px origin-top"
            style={{
              background:
                "linear-gradient(to bottom, var(--color-signal), transparent)",
            }}
          />
        </div>
      </div>
      <p className="text-[11px] text-ink-dim">
        {stage ? `${stage}\u2026` : "Working\u2026"}
      </p>
      <p className="max-w-sm text-[10px] leading-relaxed text-ink-faint">
        The map, the disagreement and the verdict appear together once the run
        finishes. A half-finished stage has no finding to report, and a number
        shown before its stage completes is not one you should be reading.
      </p>
      <p className="text-[10px] text-ink-faint">
        The pipeline above and the log below are live.
      </p>
    </section>
  );
}

/**
 * The investigation workspace.
 *
 * Task 2 delivers ingest: pick an input configuration, drop rasters in, and see
 * what was actually measured from the file. Readiness (Task 3), sample scenes
 * (4), the analysis contract (6), the live execution trace (7), and the map
 * workspace (8) attach to this same session.
 */
export function Studio() {
  const health = useHealth();
  const tour = useTourControls();
  const auth = useAuth();
  const {
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
  } = useSession();
  const samples = useSamples();
  const [activeSample, setActiveSample] = useState<string | null>(null);
  // A question waiting for a freshly loaded remedy scene to become askable.
  const [pendingQuery, setPendingQuery] = useState<string | null>(null);
  const [demo, setDemo] = useState<DemoScript | null>(null);

  const handleLoadSample = async (scene: SampleScene) => {
    const loaded = await samples.load(scene.key);
    if (loaded) {
      adopt(loaded);
      setActiveSample(scene.key);
    }
  };

  /**
   * Arm a scripted demonstration: load its scene and draft its question, then
   * stop. Approval is left to the presenter, because a demo that clicked through
   * the contract would be quietly disproving the guarantee it exists to show.
   */
  const handleStartDemo = async (script: DemoScript) => {
    setDemo(script);
    run.clear();
    contract.discard();
    const loaded = await samples.load(script.sampleKey);
    if (!loaded) return;
    adopt(loaded);
    setActiveSample(script.sampleKey);
    setPendingQuery(script.query);
  };

  /**
   * Accepting a remedy is deliberately one action: load the scene that removes
   * the objection and re-ask the question. Splitting it into load-then-type
   * would let the question drift, and a changed verdict would no longer be
   * attributable to the changed data.
   *
   * The question is parked rather than drafted here, because the contract has to
   * be planned against the new session and the readiness gate has to clear
   * first. An effect below fires it once both are true.
   */
  const handleTakeRemedy = async (offer: RemedyOffer) => {
    const question = offer.suggested_query ?? run.trace?.query ?? null;
    const loaded = await samples.load(offer.sample_key);
    if (!loaded) return;
    run.clear();
    contract.discard();
    adopt(loaded);
    setActiveSample(offer.sample_key);
    setPendingQuery(question);
  };

  const spec = modeFor(mode);
  const images = session?.images ?? {};
  const ingested = spec.slots
    .map((slot) => images[slot.role])
    .filter((image): image is NonNullable<typeof image> => Boolean(image));
  const allSlotsFilled = spec.slots.every((slot) => Boolean(images[slot.role]));

  // The gate runs as soon as any imagery exists, so an unusable file is called
  // out immediately rather than after the second upload.
  const readiness = useReadiness(session, ingested.length > 0);
  const toolInfo = useTools(session);
  const contract = useContract(session);
  const run = useRun(session?.session_id);
  // The server writes the layers before it closes the stream, so a terminal run
  // status is the signal that they are there to fetch.
  const layers = useLayers(
    session?.session_id,
    run.trace?.run_id,
    Boolean(run.trace && run.trace.status !== "running"),
  );

  const refused = readiness.report?.verdict === "refused";
  const canAsk = allSlotsFilled && !refused;

  // The run has stopped, whatever the outcome. Findings are shown from here and
  // not before. Refused, failed and cancelled count: each is a final answer, and
  // each has its own thing to say.
  const settled = Boolean(
    run.trace && run.trace.status !== "running" && !run.streaming,
  );
  const activeScene =
    samples.scenes.find((scene) => scene.key === activeSample) ?? null;

  // Fires the parked remedy question once the new imagery is in and has passed
  // the gate. Drafting earlier would plan against the scene being replaced.
  useEffect(() => {
    if (!pendingQuery || !canAsk) return;
    const question = pendingQuery;
    setPendingQuery(null);
    void contract.draft(question);
  }, [pendingQuery, canAsk, contract]);

  // Where the scripted run has got to. Derived from the same state the rest of
  // the page renders from, so the presenter's cue cannot disagree with what is
  // on screen.
  const phase: PresenterPhase = !demo
    ? "idle"
    : run.trace && run.trace.status !== "running"
      ? "answered"
      : run.streaming || run.starting || run.trace
        ? "running"
        : contract.contract
          ? "awaiting_approval"
          : pendingQuery || contract.loading
            ? "drafting"
            : samples.loadingKey || !allSlotsFilled
              ? "loading"
              : "drafting";

  return (
    <div className="min-h-screen bg-void">
      <header className="sticky top-0 z-40 flex items-center justify-between gap-3 border-b border-edge bg-void/85 px-3 py-3 backdrop-blur sm:px-6 sm:py-3.5">
        <div className="flex min-w-0 items-center gap-3 sm:gap-5">
          <Link to="/" aria-label="Back to the landing page" className="shrink-0">
            <Wordmark compact />
          </Link>
          <span className="hidden text-[11px] text-ink-faint sm:inline">Studio</span>
          {session && (
            <span
              className="tabular hidden text-[10px] text-ink-faint md:inline"
              title="Session identifier"
            >
              {session.session_id.slice(0, 8)}
            </span>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-1.5 sm:gap-2.5">
          {/* Hidden on a phone, where the input configuration panel already says it. */}
          <span
            className={`hidden whitespace-nowrap rounded-md border px-2.5 py-1 text-[10px] uppercase tracking-wide sm:inline-block ${
              session?.configuration === "incomplete"
                ? "border-edge-hi bg-panel text-ink-faint"
                : "border-agree/40 bg-agree/10 text-agree"
            }`}
          >
            {CONFIG_LABEL[session?.configuration ?? "incomplete"]}
          </span>
          <TourButton onStart={tour.start} />
          <HealthPill
            health={health.health}
            error={health.error}
            loading={health.loading}
            onRefresh={health.refresh}
          />
          {auth.signedIn ? (
            <PanelBoundary panel="The account menu">
              <ProfileMenu />
            </PanelBoundary>
          ) : (
            <button
              type="button"
              onClick={() => auth.signIn()}
              className="whitespace-nowrap rounded-md border border-edge px-2.5 py-1 text-[10px] uppercase tracking-wide text-ink-faint transition hover:border-signal/50 hover:text-signal"
            >
              Sign in
            </button>
          )}
        </div>
      </header>

      {/*
        Full width up to a large cap, so a wide monitor is not left as two empty
        margins. grid-cols-1 below lg matters: an implicit auto column grows to
        the widest unbreakable child (a truncated title, a nowrap readout), which
        pushed the whole page sideways on a phone.
      */}
      <main className="mx-auto grid w-full max-w-[2560px] grid-cols-1 gap-4 px-3 py-4 sm:px-6 sm:py-7 lg:grid-cols-[minmax(0,340px)_minmax(0,1fr)] lg:gap-6">
        {/*
          On a desktop the inputs column stays in view while the findings scroll,
          and scrolls on its own when it is taller than the window. Otherwise the
          left third of the page sits empty once a run pushes the results down.

          Below lg it is display: contents, so its panels join the page's single
          column directly and the tool list can be moved after the findings. On a
          phone that list is a screen and a half of reference material standing
          between loading a scene and asking about it.
        */}
        <div className="contents lg:sticky lg:top-20 lg:block lg:max-h-[calc(100dvh-6rem)] lg:space-y-4 lg:self-start lg:overflow-y-auto lg:[scrollbar-color:var(--color-edge-hi)_transparent] lg:[scrollbar-gutter:stable] lg:[scrollbar-width:thin]">
          <PresenterPanel
            active={demo}
            phase={phase}
            availableKeys={
              new Set(
                samples.scenes
                  .filter((scene) => Object.keys(scene.assets).length > 0)
                  .map((scene) => scene.key),
              )
            }
            busy={Boolean(samples.loadingKey) || contract.loading || run.streaming}
            onStart={(script) => void handleStartDemo(script)}
            onClear={() => setDemo(null)}
          />

          <SampleGallery
            scenes={samples.scenes}
            loading={samples.loading}
            error={samples.error}
            loadingKey={samples.loadingKey}
            activeKey={activeSample}
            onLoad={(scene) => void handleLoadSample(scene)}
          />

          <section className="panel p-4">
            <h2 className="text-[10px] uppercase tracking-wider text-ink-faint">
              Input configuration
            </h2>
            <div
              className="mt-2.5 grid gap-1.5"
              role="radiogroup"
              aria-label="Input configuration"
            >
              {MODES.map((option) => {
                const active = option.id === mode;
                return (
                  <button
                    key={option.id}
                    type="button"
                    role="radio"
                    aria-checked={active}
                    onClick={() => {
                      setActiveSample(null);
                      setMode(option.id);
                    }}
                    className={`rounded-lg border px-3 py-2 text-left transition ${
                      active
                        ? "border-signal/50 bg-signal/10"
                        : "border-edge bg-hull/40 hover:border-edge-hi"
                    }`}
                  >
                    <span
                      className={`block text-xs font-medium ${
                        active ? "text-signal" : "text-ink"
                      }`}
                    >
                      {option.label}
                    </span>
                    <span className="mt-0.5 block text-[10px] leading-snug text-ink-faint">
                      {option.hint}
                    </span>
                  </button>
                );
              })}
            </div>
          </section>

          <section className="panel p-4">
            <div className="flex items-center justify-between">
              <h2 className="text-[10px] uppercase tracking-wider text-ink-faint">
                Imagery
              </h2>
              {ingested.length > 0 && (
                <button
                  type="button"
                  onClick={() => {
                    setActiveSample(null);
                    void reset();
                  }}
                  className="text-[10px] text-ink-faint transition hover:text-disagree"
                >
                  Clear all
                </button>
              )}
            </div>

            <div className="mt-2.5 space-y-2.5">
              {spec.slots.map((slot) => (
                <UploadSlot
                  key={slot.role}
                  slot={slot}
                  filled={Boolean(images[slot.role])}
                  filename={images[slot.role]?.original_filename}
                  progress={progress[slot.role] ?? null}
                  error={errors[slot.role] ?? null}
                  disabled={starting || !session}
                  onFile={(file) => void uploadTo(slot.role, file)}
                />
              ))}
            </div>

            {startError && (
              <p role="alert" className="mt-3 text-[11px] leading-snug text-disagree">
                {startError}
              </p>
            )}

            <p className="mt-3 text-[10px] leading-snug text-ink-faint">
              GeoTIFF and TIFF carry the georeferencing the pipeline needs. PNG and
              JPEG are accepted for the prescribed benchmark datasets, where area
              and distance measurements are unavailable.
            </p>
          </section>

          {/* Last on a phone; in place in the desktop column, where order is inert. */}
          <div className="order-last">
            <CapabilityPanel
              tools={toolInfo.tools}
              availability={toolInfo.availability}
              loading={toolInfo.loading}
              error={toolInfo.error}
            />
          </div>
        </div>

        <div className="space-y-4">
          {ingested.length === 0 ? (
            <div className="panel flex min-h-[340px] flex-col items-center justify-center gap-2 p-10 text-center">
              <div
                className="relative mb-2 h-14 w-14 rounded-full border border-edge"
                aria-hidden="true"
              >
                <div className="absolute inset-0 animate-[var(--animate-sweep)]">
                  <div
                    className="absolute left-1/2 top-1/2 h-1/2 w-px origin-top"
                    style={{
                      background:
                        "linear-gradient(to bottom, var(--color-signal), transparent)",
                    }}
                  />
                </div>
              </div>
              <p className="text-sm text-ink-dim">No imagery loaded yet</p>
              <p className="max-w-sm text-[11px] leading-relaxed text-ink-faint">
                Drop a file into a slot on the left. Everything shown here is read
                from the file itself, never assumed from the filename.
              </p>
            </div>
          ) : (
            <>
              <ReadinessPanel
                report={readiness.report}
                loading={readiness.loading}
                error={readiness.error}
              />

              <QueryBar
                suggestions={activeScene?.suggested_queries ?? []}
                disabled={!canAsk}
                busy={contract.loading}
                onSubmit={(query) => void contract.draft(query)}
                activeScene={activeScene}
              />

              <ContractCard
                contract={contract.contract}
                loading={contract.loading}
                error={contract.error}
                onApprove={() => {
                  const hash = contract.contract?.contract_hash;
                  if (hash) void run.start(hash);
                }}
                onDiscard={() => {
                  contract.discard();
                  run.clear();
                }}
                approveLabel={run.streaming ? "Running" : "Approve and run"}
                approveDisabled={run.starting || run.streaming}
              />

              {run.error && (
                <p
                  role="alert"
                  className="panel px-4 py-3 text-[11px] leading-snug text-disagree"
                >
                  {run.error}
                </p>
              )}

              <AnimatePresence>
                {run.trace && (
                  <motion.div
                    key={run.trace.run_id}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="space-y-4"
                  >
                    {/*
                      One boundary per panel, not one around the run.
                      These are independent readings of the same run, so a fault
                      in any of them should cost that panel and nothing else. The
                      run once went blank as a whole because a single component
                      threw and React took the tree with it, which also destroyed
                      the trace that would have explained why.
                    */}
                    <PanelBoundary panel="The pipeline">
                      <AgentGraph
                        stages={run.trace.stages}
                        status={run.trace.status}
                        durationMs={run.trace.duration_ms}
                      />
                    </PanelBoundary>

                    {/*
                      Findings wait for the run to finish; the pipeline and the log
                      above and below them do not.

                      A stage that has not run yet has no result, so a panel
                      rendered next to it is either empty or half-built, and the
                      reader cannot tell which. Worse, it invites reading a partial
                      number as the answer. The live account of what is happening
                      is the pipeline and the log, which are built for it.
                    */}
                    {!settled ? (
                      <PanelBoundary panel="The findings">
                        <PendingFindings
                          stage={
                            run.trace.stages.find(
                              (item) => item.status === "running",
                            )?.label ?? null
                          }
                        />
                      </PanelBoundary>
                    ) : (
                      <>
                        <PanelBoundary panel="The map">
                          <MapWorkspace
                            layers={layers.layers}
                            loading={layers.loading}
                            error={layers.error}
                          />
                        </PanelBoundary>
                        <PanelBoundary panel="The evidence disagreement">
                          <EvidenceDisagreementPanel
                            disagreement={run.trace.disagreement}
                            layers={layers.layers}
                          />
                        </PanelBoundary>
                        {run.trace.verdict && (
                          <PanelBoundary panel="The verdict">
                            <VerdictCard verdict={run.trace.verdict} />
                          </PanelBoundary>
                        )}
                        <PanelBoundary panel="The imagery that would settle this">
                          <RemedyOffers
                            remedies={run.trace.remedies}
                            busyKey={samples.loadingKey}
                            disabled={contract.loading || run.streaming}
                            onTake={(offer) => void handleTakeRemedy(offer)}
                          />
                        </PanelBoundary>
                        <PanelBoundary panel="The sensor comparison">
                          <DisagreementPanel
                            measurements={run.trace.measurements}
                            layers={layers.layers}
                            notes={
                              run.trace.tool_runs.find(
                                (item) => item.tool === "optical-sar-fusion",
                              )?.notes ?? []
                            }
                          />
                        </PanelBoundary>
                        <PanelBoundary panel="The alternative explanations">
                          <ConfounderPanel tests={run.trace.confounders} />
                        </PanelBoundary>
                        {session && (
                          <PanelBoundary panel="The evidence packet">
                            <EvidencePacketCard
                              packet={run.trace.packet}
                              sessionId={session.session_id}
                            />
                          </PanelBoundary>
                        )}
                      </>
                    )}

                    <PanelBoundary panel="The execution trace">
                      <RunTracePanel
                        trace={run.trace}
                        log={run.log}
                        streaming={run.streaming}
                        onCancel={() => void run.cancel()}
                      />
                    </PanelBoundary>
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="space-y-3">
                <div className="flex items-baseline justify-between">
                  <h2 className="text-[10px] uppercase tracking-wider text-ink-faint">
                    Measured from the file
                  </h2>
                  <span className="tabular text-[10px] text-ink-faint">
                    {ingested.length} of {spec.slots.length} slots
                  </span>
                </div>
                {/*
                  Side by side when two full cards fit, one under the other when
                  they do not. auto-fit drops the unused track, so a single image
                  still gets the whole width instead of half of it.
                */}
                <div className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,28rem),1fr))] gap-4">
                  {spec.slots.map((slot) => {
                    const image = images[slot.role];
                    if (!image || !session) return null;
                    return (
                      <MetadataPanel
                        key={slot.role}
                        image={image}
                        thumbnailUrl={api.thumbnailUrl(
                          session.session_id,
                          slot.role,
                          image.sha256,
                        )}
                        onRemove={() => void removeFrom(slot.role)}
                      />
                    );
                  })}
                </div>
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
