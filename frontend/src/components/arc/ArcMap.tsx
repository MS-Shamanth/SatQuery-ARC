import "leaflet/dist/leaflet.css";
import { useEffect, useMemo } from "react";
import {
  CircleMarker,
  MapContainer,
  Rectangle,
  TileLayer,
  Tooltip,
  useMap,
} from "react-leaflet";
import type { LatLngBoundsExpression } from "leaflet";
import type { SearchResult } from "../../lib/arc";

const ESRI_TILES =
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const ESRI_ATTR = "Imagery &copy; Esri, Maxar, Earthstar Geographics";

// India bounding box, used to place the AOI dot on the inset silhouette.
const INDIA = { west: 67.5, east: 98.0, south: 6.5, north: 37.5 };

function toBounds(b: number[]): LatLngBoundsExpression {
  const [w, s, e, n] = b;
  return [
    [s, w],
    [n, e],
  ];
}

function FitToResults({ results }: { results: SearchResult[] }) {
  const map = useMap();
  useEffect(() => {
    if (results.length === 0) return;
    let [w, s, e, n] = results[0].bounds_wgs84;
    for (const r of results.slice(1)) {
      const [rw, rs, re, rn] = r.bounds_wgs84;
      w = Math.min(w, rw);
      s = Math.min(s, rs);
      e = Math.max(e, re);
      n = Math.max(n, rn);
    }
    // Pad a touch so single-AOI fits are not absurdly zoomed.
    const padLon = Math.max(0.4, (e - w) * 0.4);
    const padLat = Math.max(0.4, (n - s) * 0.4);
    map.fitBounds(
      [
        [s - padLat, w - padLon],
        [n + padLat, e + padLon],
      ],
      { padding: [24, 24], maxZoom: 11 },
    );
    setTimeout(() => map.invalidateSize(), 60);
  }, [results, map]);
  return null;
}

export function ArcMap({
  results,
  selectedKey,
  onSelect,
}: {
  results: SearchResult[];
  selectedKey: string | null;
  onSelect: (aoiKey: string) => void;
}) {
  const center = useMemo<[number, number]>(() => {
    if (results.length) return [results[0].lat, results[0].lon];
    return [23.5, 80.9];
  }, [results]);

  const selected = results.find((r) => r.aoi_key === selectedKey) ?? null;

  return (
    <div className="relative h-full w-full overflow-hidden rounded-xl border border-edge">
      <MapContainer
        center={center}
        zoom={5}
        scrollWheelZoom={false}
        zoomSnap={0.25}
        className="h-full w-full"
        style={{ background: "#05070d" }}
        attributionControl
      >
        <TileLayer url={ESRI_TILES} attribution={ESRI_ATTR} maxZoom={19} />
        <FitToResults results={results} />
        {results.map((r) => {
          const active = r.aoi_key === selectedKey;
          return (
            <Rectangle
              key={r.tile_id}
              bounds={toBounds(r.bounds_wgs84)}
              pathOptions={{
                color: active ? "#34d399" : "#22d3ee",
                weight: active ? 2.5 : 1.6,
                fillOpacity: active ? 0.12 : 0.05,
                dashArray: active ? "6 4" : undefined,
              }}
              eventHandlers={{ click: () => onSelect(r.aoi_key) }}
            >
              <Tooltip direction="top" className="satquery-tooltip" opacity={1}>
                {r.place} &middot; {r.match.toFixed(2)}
              </Tooltip>
            </Rectangle>
          );
        })}
        {selected && (
          <CircleMarker
            center={[selected.lat, selected.lon]}
            radius={5}
            pathOptions={{ color: "#34d399", fillColor: "#34d399", fillOpacity: 1 }}
          />
        )}
      </MapContainer>

      {/* Header label */}
      <div className="pointer-events-none absolute left-3 top-3 z-[500] rounded-lg border border-edge bg-void/70 px-3 py-1.5 text-[11px] text-ink-dim backdrop-blur">
        {selected ? (
          <>
            AOI &middot; <b className="text-ink">{selected.place}</b> &middot;{" "}
            <span className="tabular">
              {selected.lat.toFixed(3)}&deg;N, {selected.lon.toFixed(3)}&deg;E
            </span>
          </>
        ) : (
          <>
            <b className="text-ink">{results.length}</b> footprints &middot; select one to verify
          </>
        )}
      </div>

      {/* India inset */}
      <div className="absolute right-3 top-3 z-[500] h-28 w-24 rounded-lg border border-edge bg-void/80 p-1.5 backdrop-blur">
        <p className="mb-0.5 text-center text-[8px] uppercase tracking-wider text-ink-faint">
          India
        </p>
        <svg viewBox="0 0 100 110" className="h-[86px] w-full">
          <path
            d="M20 18 L38 12 L52 20 L70 16 L82 30 L74 44 L80 58 L64 74 L58 96 L48 86 L44 66 L32 58 L24 44 L30 34 Z"
            fill="rgba(34,211,238,0.08)"
            stroke="#2c4062"
            strokeWidth="1.2"
          />
          {results.map((r) => {
            const x = ((r.lon - INDIA.west) / (INDIA.east - INDIA.west)) * 100;
            const y = ((INDIA.north - r.lat) / (INDIA.north - INDIA.south)) * 110;
            const active = r.aoi_key === selectedKey;
            return (
              <circle
                key={r.tile_id}
                cx={x}
                cy={y}
                r={active ? 3 : 2}
                fill={active ? "#34d399" : "#22d3ee"}
              />
            );
          })}
        </svg>
      </div>
    </div>
  );
}
