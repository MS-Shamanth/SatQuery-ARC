import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { HealthPill } from "../components/HealthPill";
import { OrbitalScene } from "../components/OrbitalScene";
import { PipelineStrip } from "../components/PipelineStrip";
import { Starfield } from "../components/Starfield";
import { Wordmark } from "../components/Wordmark";
import { useHealth } from "../lib/useHealth";
import { useAuth } from "../lib/auth";
import { useNavigate } from "react-router-dom";

const VERDICTS = [
  { label: "Confirmed", cls: "text-agree border-agree/30 bg-agree/10" },
  { label: "Inconclusive", cls: "text-uncertain border-uncertain/30 bg-uncertain/10" },
  { label: "Rejected", cls: "text-disagree border-disagree/30 bg-disagree/10" },
];

const rise = (delay: number) => ({
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0 },
  transition: { delay, duration: 0.7, ease: [0.16, 1, 0.3, 1] as const },
});

export function Landing() {
  const { health, error, loading, refresh } = useHealth();
  const auth = useAuth();
  const navigate = useNavigate();

  return (
    <main className="grain relative min-h-screen overflow-hidden bg-void">
      <Starfield />

      {/* Horizon glow */}
      <div
        className="pointer-events-none absolute inset-x-0 bottom-0 h-[42vh]"
        style={{
          background:
            "radial-gradient(ellipse 120% 100% at 50% 130%, rgba(34,211,238,0.14), transparent 70%)",
        }}
        aria-hidden="true"
      />
      {/* Slow scanline, a nod to instrument displays */}
      <div
        className="pointer-events-none absolute inset-x-0 top-0 h-24 animate-[var(--animate-scanline)] opacity-[0.06]"
        style={{
          background:
            "linear-gradient(to bottom, transparent, var(--color-signal), transparent)",
        }}
        aria-hidden="true"
      />

      <div className="relative mx-auto flex min-h-screen max-w-7xl flex-col px-6 py-6 lg:px-10">
        {/* Positioned so the status panel has an anchor on a phone. */}
        <header className="relative flex items-center justify-between gap-3">
          <Wordmark />
          <div className="flex items-center gap-2.5">
            <HealthPill
              health={health}
              error={error}
              loading={loading}
              onRefresh={refresh}
            />
          </div>
        </header>

        <div className="grid flex-1 items-center gap-10 py-10 lg:grid-cols-[1.15fr_0.85fr]">
          <div>
            <motion.p
              {...rise(0.05)}
              className="mb-5 inline-flex items-center gap-2 rounded-full border border-edge bg-hull/60 px-3 py-1 text-[11px] text-ink-dim"
            >
              <span className="h-1.5 w-1.5 rounded-full bg-signal" />
              Archive Search &amp; Verified Change &middot; SIH26227 &middot; fully on-premises
            </motion.p>

            {/*
              Wrapped so the tour has one box to highlight: the claim, what backs
              it, and the three outcomes. Spotlighting the headline alone would
              cut off the part that explains it.
            */}
            <div data-tour="landing-hero">
              <motion.h1
                {...rise(0.12)}
                className="text-5xl font-semibold leading-[1.05] tracking-tight text-ink lg:text-6xl"
              >
                Search the archive
                <br />
                by meaning.
                <br />
                <span className="text-ink-dim">Verify every change.</span>
              </motion.h1>

              <motion.p
                {...rise(0.2)}
                className="mt-6 max-w-xl text-[15px] leading-relaxed text-ink-dim"
              >
                A sovereign multi-sensor satellite archive you can query by free
                text, an example tile, or &ldquo;find more like this&rdquo;. Every
                candidate change is checked for season, cloud, misregistration and
                other confounders before an analyst sees it &mdash; and no data
                leaves the building.
              </motion.p>

              <motion.div
                {...rise(0.28)}
                className="mt-7 flex flex-wrap items-center gap-2"
              >
                {VERDICTS.map((v) => (
                  <span
                    key={v.label}
                    className={`rounded-full border px-3 py-1 text-xs font-medium ${v.cls}`}
                  >
                    {v.label}
                  </span>
                ))}
                <span className="text-xs text-ink-faint">
                  &mdash; a forced yes/no is not one of the options
                </span>
              </motion.div>
            </div>

            <motion.div {...rise(0.36)} className="mt-9 flex flex-wrap items-center gap-3">
              {/*
                One button, no form. A demo that asks a reviewer to invent a
                password is a demo they abandon at the first screen, and there is
                nothing behind it to protect: this signs in a local identity and
                opens the Studio. See lib/auth.ts.
              */}
              {auth.signedIn ? (
                <Link
                  to="/studio"
                  data-tour="open-studio"
                  className="group relative overflow-hidden rounded-xl bg-signal px-5 py-3 text-sm font-semibold text-void transition hover:brightness-110"
                >
                  Open the Studio
                  <span className="ml-2 inline-block transition-transform group-hover:translate-x-0.5">
                    &rarr;
                  </span>
                </Link>
              ) : (
                <button
                  type="button"
                  data-tour="open-studio"
                  onClick={() => {
                    auth.signIn();
                    navigate("/studio");
                  }}
                  className="group relative overflow-hidden rounded-xl bg-signal px-5 py-3 text-sm font-semibold text-void transition hover:brightness-110"
                >
                  Sign in to the demo
                  <span className="ml-2 inline-block transition-transform group-hover:translate-x-0.5">
                    &rarr;
                  </span>
                </button>
              )}
              {!auth.signedIn && (
                <Link
                  to="/studio"
                  className="rounded-xl border border-edge px-5 py-3 text-sm text-ink-dim transition hover:border-edge-hi hover:text-ink"
                >
                  Skip, just open the Studio
                </Link>
              )}
              <a
                href="/api/docs"
                target="_blank"
                rel="noreferrer"
                className="rounded-xl border border-edge px-5 py-3 text-sm text-ink-dim transition hover:border-edge-hi hover:text-ink"
              >
                API reference
              </a>
            </motion.div>
          </div>

          <motion.div
            initial={{ opacity: 0, scale: 0.92 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.25, duration: 1.1, ease: [0.16, 1, 0.3, 1] }}
            className="flex justify-center lg:justify-end"
          >
            <OrbitalScene />
          </motion.div>
        </div>

        <div className="pb-2">
          <motion.p
            {...rise(0.5)}
            className="mb-2.5 text-[10px] uppercase tracking-wider text-ink-faint"
          >
            Archive &rarr; verified change pipeline
          </motion.p>
          <PipelineStrip delay={0.55} />
        </div>

        <motion.footer
          {...rise(1.1)}
          className="mt-6 flex flex-wrap items-center gap-x-5 gap-y-1.5 border-t border-edge pt-4 text-[11px] text-ink-faint"
        >
          <span>
            Retrieval:{" "}
            <a
              className="text-ink-dim underline decoration-edge-hi underline-offset-2 hover:text-signal"
              href="https://arxiv.org/abs/2306.11029"
              target="_blank"
              rel="noreferrer"
            >
              RemoteCLIP
            </a>{" "}
            &middot; Change:{" "}
            <a
              className="text-ink-dim underline decoration-edge-hi underline-offset-2 hover:text-signal"
              href="https://doi.org/10.1016/j.rse.2014.01.011"
              target="_blank"
              rel="noreferrer"
            >
              CCDC
            </a>
          </span>
          <span>Sentinel-1/2 &amp; Landsat &middot; PostGIS + pgvector HNSW</span>
          <span className="tabular">
            Runs fully offline &middot; measurements are computed, never generated
          </span>
        </motion.footer>
      </div>
    </main>
  );
}
