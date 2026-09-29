import { motion } from "framer-motion";
import { usePrefersReducedMotion } from "../lib/useReducedMotion";

/**
 * The hero graphic: a graticuled globe, two orbital tracks carrying an optical
 * and a SAR platform, and a radar sweep. Decorative, hence aria-hidden.
 *
 * The two platforms are visually distinct on purpose - the whole premise of the
 * product is that optical and SAR see different things and sometimes disagree.
 *
 * The platforms ride the tracks by construction, not by coincidence.
 *
 * They used to be positioned with a CSS rotation at a fixed offset from the
 * centre, which traces a circle. The tracks are flattened, tilted ellipses, so the
 * dot met its own orbit at exactly two points and spent the rest of the loop out
 * in open space, well clear of any line. Both now come from one path: the ellipse
 * is defined once, stroked as the track, and referenced by ``animateMotion`` as
 * the route. They cannot drift apart, because there is only one of them.
 */

/** An ellipse as a closed path, so it can be both stroked and animated along. */
function ellipsePath(cx: number, cy: number, rx: number, ry: number): string {
  return [
    `M ${cx - rx} ${cy}`,
    `a ${rx} ${ry} 0 1 0 ${rx * 2} 0`,
    `a ${rx} ${ry} 0 1 0 ${-rx * 2} 0`,
    "Z",
  ].join(" ");
}

interface Orbit {
  id: string;
  /** Semi-axes as functions of the half-size, so the scene scales. */
  rx: (c: number) => number;
  ry: (c: number) => number;
  /** Degrees. Applied to the track and the platform together. */
  tilt: number;
  track: string;
  opacity: number;
  colour: string;
  radius: number;
  seconds: number;
  /** Retrograde, so the two platforms visibly cross rather than pace each other. */
  reverse?: boolean;
}

const ORBITS: readonly Orbit[] = [
  {
    id: "orbit-optical",
    rx: (c) => c - 6,
    ry: (c) => (c - 6) * 0.34,
    tilt: 0,
    track: "var(--color-edge-hi)",
    opacity: 0.75,
    colour: "var(--color-signal)",
    radius: 5,
    seconds: 24,
  },
  {
    id: "orbit-sar",
    rx: (c) => c - 34,
    ry: (c) => (c - 34) * 0.62,
    tilt: -24,
    track: "var(--color-edge)",
    opacity: 0.6,
    colour: "var(--color-uncertain)",
    radius: 4,
    seconds: 42,
    reverse: true,
  },
] as const;

export function OrbitalScene({ size = 420 }: { size?: number }) {
  const c = size / 2;
  const reduceMotion = usePrefersReducedMotion();

  return (
    <div
      className="relative"
      style={{ width: size, height: size }}
      aria-hidden="true"
      data-testid="orbital-scene"
    >
      {/* Radar sweep, rotating behind the globe */}
      <div className="absolute inset-0 animate-[var(--animate-sweep)] [transform-origin:50%_50%]">
        <div
          className="absolute left-1/2 top-1/2 h-1/2 w-px origin-top"
          style={{
            background:
              "linear-gradient(to bottom, var(--color-signal), transparent)",
          }}
        />
      </div>

      <svg
        viewBox={`0 0 ${size} ${size}`}
        width={size}
        height={size}
        className="absolute inset-0"
      >
        <defs>
          <radialGradient id="globeFill" cx="38%" cy="32%" r="78%">
            <stop offset="0%" stopColor="#1b3a5c" />
            <stop offset="55%" stopColor="#0d1f38" />
            <stop offset="100%" stopColor="#060c18" />
          </radialGradient>
          <radialGradient id="limbGlow" cx="50%" cy="50%" r="50%">
            <stop offset="72%" stopColor="rgba(34,211,238,0)" />
            <stop offset="92%" stopColor="rgba(34,211,238,0.28)" />
            <stop offset="100%" stopColor="rgba(34,211,238,0)" />
          </radialGradient>
        </defs>

        {/* Orbital tracks, drawn from the same path the platform travels. */}
        {ORBITS.map((orbit) => (
          <g
            key={`track-${orbit.id}`}
            transform={`rotate(${orbit.tilt} ${c} ${c})`}
          >
            <path
              id={orbit.id}
              d={ellipsePath(c, c, orbit.rx(c), orbit.ry(c))}
              fill="none"
              stroke={orbit.track}
              strokeWidth="1"
              opacity={orbit.opacity}
            />
          </g>
        ))}

        {/* Atmospheric limb */}
        <circle cx={c} cy={c} r={c * 0.52} fill="url(#limbGlow)" />

        {/* Globe */}
        <circle cx={c} cy={c} r={c * 0.42} fill="url(#globeFill)" />
        <circle
          cx={c}
          cy={c}
          r={c * 0.42}
          fill="none"
          stroke="rgba(34,211,238,0.35)"
          strokeWidth="1"
        />

        {/* Graticule: parallels */}
        {[-0.28, -0.14, 0, 0.14, 0.28].map((offset) => {
          const r = c * 0.42;
          const y = c + r * offset;
          const half = Math.sqrt(Math.max(0, r * r - (r * offset) ** 2));
          return (
            <ellipse
              key={`par-${offset}`}
              cx={c}
              cy={y}
              rx={half}
              ry={half * 0.14}
              fill="none"
              stroke="rgba(34,211,238,0.16)"
              strokeWidth="0.75"
            />
          );
        })}

        {/* Graticule: meridians */}
        {[0.22, 0.48, 0.74, 1].map((factor, index) => (
          <ellipse
            key={`mer-${index}`}
            cx={c}
            cy={c}
            rx={c * 0.42 * factor}
            ry={c * 0.42}
            fill="none"
            stroke="rgba(34,211,238,0.14)"
            strokeWidth="0.75"
          />
        ))}

        {/*
          The platforms, drawn last so they pass in front of the globe.

          Inside the same rotated group as their track, so the tilt applies to both
          and there is no second copy of it to get wrong.
        */}
        {ORBITS.map((orbit) => (
          <g
            key={`sat-${orbit.id}`}
            transform={`rotate(${orbit.tilt} ${c} ${c})`}
          >
            <g
              // Parked at one end of the track when motion is not wanted. Still on
              // the line, which is the part that matters.
              transform={
                reduceMotion
                  ? `translate(${c + orbit.rx(c)} ${c})`
                  : undefined
              }
            >
              <circle r={orbit.radius * 2.2} fill={orbit.colour} opacity="0.18" />
              <circle r={orbit.radius} fill={orbit.colour} />
              {!reduceMotion && (
                <animateMotion
                  dur={`${orbit.seconds}s`}
                  repeatCount="indefinite"
                  calcMode="linear"
                  // keyPoints runs the path backwards for a retrograde orbit.
                  keyPoints={orbit.reverse ? "1;0" : "0;1"}
                  keyTimes="0;1"
                >
                  <mpath href={`#${orbit.id}`} />
                </animateMotion>
              )}
            </g>
          </g>
        ))}
      </svg>

      {/* Legend tying the two dots to the two modalities */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2, duration: 0.8 }}
        className="absolute -bottom-2 left-1/2 flex -translate-x-1/2 gap-4 text-[11px] text-ink-faint"
      >
        <span className="flex items-center gap-1.5">
          <span className="h-1.5 w-1.5 rounded-full bg-signal" />
          Optical / multispectral
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-1.5 w-1.5 rounded-full bg-uncertain" />
          SAR
        </span>
      </motion.div>
    </div>
  );
}
