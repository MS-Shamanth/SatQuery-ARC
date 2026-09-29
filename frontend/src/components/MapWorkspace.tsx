import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  GeoJSON,
  ImageOverlay,
  MapContainer,
  TileLayer,
  useMap,
} from "react-leaflet";
import { formatArea, formatInteger, formatLatLon } from "../lib/format";
import type { MapLayer } from "../lib/types";

/** Esri's imagery service, used as optional geographic context. */
const CONTEXT_TILES =
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const CONTEXT_ATTRIBUTION =
  "Context imagery &copy; Esri, Maxar, Earthstar Geographics";

type Bounds = [[number, number], [number, number]];

function toLeaflet(bounds: [number, number, number, number]): Bounds {
  const [west, south, east, north] = bounds;
  return [
    [south, west],
    [north, east],
  ];
}

function union(layers: MapLayer[]): Bounds | null {
  if (layers.length === 0) return null;
  let [w, s, e, n] = layers[0].bounds_wgs84;
  for (const layer of layers.slice(1)) {
    const [lw, ls, le, ln] = layer.bounds_wgs84;
    w = Math.min(w, lw);
    s = Math.min(s, ls);
    e = Math.max(e, le);
    n = Math.max(n, ln);
  }
  return toLeaflet([w, s, e, n]);
}

const FIT_PADDING: [number, number] = [16, 16];

/**
 * Keeps the viewport on the scene: fitted as layers arrive, and fitted again
 * whenever the map's box changes shape.
 *
 * The box changes when the panel's container query moves the layer list beside
 * the map or back under it. Leaflet re-measures only on a window resize and
 * never refits, so a map resized under it kept its old centre, left new ground
 * untiled, and cropped a scene that had been fitted to a bigger box.
 *
 * A reader who has moved the map keeps their view. Refitting stops at the first
 * press or key inside the map, and resumes with the next set of layers.
 */
function FitBounds({ bounds }: { bounds: Bounds | null }) {
  const map = useMap();
  const moved = useRef(false);

  useEffect(() => {
    moved.current = false;
    if (!bounds) return;
    map.fitBounds(bounds, { padding: FIT_PADDING });
  }, [bounds, map]);

  useEffect(() => {
    const container = map.getContainer();
    const mark = () => {
      moved.current = true;
    };
    container.addEventListener("pointerdown", mark);
    container.addEventListener("keydown", mark);

    // jsdom has no ResizeObserver, and a map that cannot watch itself still works.
    const observer =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(() => {
            map.invalidateSize();
            const size = map.getSize();
            // A box smaller than its own padding has no fit, and Leaflet would
            // derive a NaN zoom from it and throw.
            if (!bounds || moved.current || size.x <= 40 || size.y <= 40) return;
            map.fitBounds(bounds, { padding: FIT_PADDING, animate: false });
          });
    observer?.observe(container);

    return () => {
      observer?.disconnect();
      container.removeEventListener("pointerdown", mark);
      container.removeEventListener("keydown", mark);
    };
  }, [bounds, map]);

  return null;
}

/**
 * The map workspace.
 *
 * Two decisions worth stating. The scene under investigation is itself the base
 * layer, reprojected server-side, so the map is complete without a tile server
 * and a recording cannot be spoiled by a network hiccup; online imagery is
 * offered as context on top of that, not as the foundation. And every overlay is
 * placed from bounds derived by reprojecting the source grid, so a highlighted
 * region sits on exactly the pixels whose area is quoted beside it.
 */
export function MapWorkspace({
  layers,
  loading,
  error,
}: {
  layers: MapLayer[];
  loading: boolean;
  error: string | null;
}) {
  const bases = useMemo(
    () => layers.filter((layer) => layer.kind === "base"),
    [layers],
  );
  const masks = useMemo(
    () => layers.filter((layer) => layer.kind === "mask"),
    [layers],
  );

  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [showContext, setShowContext] = useState(false);
  const [showOutlines, setShowOutlines] = useState(true);
  const [activeBase, setActiveBase] = useState<string | null>(null);
  const [opacity, setOpacity] = useState(0.7);
  // Off by default. The formulas are the point of the system and they are one
  // click away, but printing all of them at once buries the map that they
  // describe.
  const [showFormulas, setShowFormulas] = useState(false);

  // Newly rendered layers start visible; a layer the user turned off stays off.
  useEffect(() => {
    if (bases.length > 0) {
      setActiveBase((current) =>
        current && bases.some((b) => b.key === current) ? current : bases[0].key,
      );
    }
  }, [bases]);

  const bounds = useMemo(() => union(layers), [layers]);
  const base = bases.find((layer) => layer.key === activeBase) ?? bases[0];

  if (error) {
    return (
      <section className="panel p-4" aria-label="Map workspace">
        <p role="alert" className="text-[11px] text-disagree">
          {error}
        </p>
      </section>
    );
  }

  if (layers.length === 0) {
    return (
      <section className="panel p-6 text-center" aria-label="Map workspace">
        <p className="text-[11px] text-ink-dim">
          {loading ? "Rendering map layers" : "No mapped result yet"}
        </p>
        <p className="mx-auto mt-1 max-w-sm text-[10px] leading-relaxed text-ink-faint">
          {loading
            ? "Each region is reprojected to latitude and longitude so it lands on the ground it was measured from."
            : "Approve a contract and any region a tool produces is drawn here, placed by its own georeferencing."}
        </p>
      </section>
    );
  }

  // Only a layer list is worth moving beside the map. The base switch and its
  // description are a line each, and a column holding only those would be empty.
  const beside = masks.length > 0;

  return (
    <section className="panel @container overflow-hidden" aria-label="Map workspace">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-edge px-4 py-2.5">
        <h2 className="text-[10px] uppercase tracking-wider text-ink-faint">
          Grounded result
        </h2>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-1.5 text-[10px] text-ink-faint">
            <span>Overlay</span>
            <input
              type="range"
              min={0.2}
              max={1}
              step={0.05}
              value={opacity}
              onChange={(event) => setOpacity(Number(event.target.value))}
              aria-label="Overlay opacity"
              className="h-1 w-20 accent-[var(--color-signal)]"
            />
          </label>
          <Toggle
            label="Outlines"
            on={showOutlines}
            onChange={setShowOutlines}
          />
          <Toggle label="Context" on={showContext} onChange={setShowContext} />
        </div>
      </div>

      {/*
        Beside the map on a wide panel, below it otherwise. Stacked on a monitor,
        a square scene sat in a strip several times as wide as it was tall, and
        each layer's name was a screen's width from its area. Beside it, the map
        gets the height it lacked and the list scrolls within that height rather
        than pushing the findings further down.

        The row takes its height from the grid, never from the list: min-h-0 on
        the column stops a long list, or one with its formulas open, from
        stretching the map to match it.
      */}
      <div
        className={
          beside
            ? "@4xl:grid @4xl:h-[clamp(28rem,62dvh,40rem)] @4xl:grid-cols-[minmax(0,1fr)_24rem]"
            : undefined
        }
      >
        {/* Scales with the viewport: a fixed 420px left almost no map on a laptop
            once the layer list below it was open. */}
        <div
          className={`h-[280px] w-full bg-abyss sm:h-[360px] lg:h-[440px] ${
            beside ? "@4xl:h-full" : ""
          }`}
        >
          <MapContainer
            bounds={bounds ?? undefined}
            // Quarter steps, so fitting the scene fills the box. Whole zoom levels
            // double the scale each time, which left the scene at about half the
            // height of a map made taller to show it.
            zoomSnap={0.25}
            className="h-full w-full"
            // The imagery is the subject, so a scroll gesture over the page should
            // not zoom it by accident.
            scrollWheelZoom={false}
            attributionControl
            style={{ background: "#05070d" }}
          >
            <FitBounds bounds={bounds} />

            {showContext && (
              <TileLayer
                url={CONTEXT_TILES}
                attribution={CONTEXT_ATTRIBUTION}
                maxZoom={19}
              />
            )}

            {base && (
              <ImageOverlay
                key={base.key}
                url={base.png_url}
                bounds={toLeaflet(base.bounds_wgs84)}
                opacity={1}
                zIndex={200}
              />
            )}

            {masks
              .filter((layer) => !hidden.has(layer.key))
              .map((layer) => (
                <ImageOverlay
                  key={layer.key}
                  url={layer.png_url}
                  bounds={toLeaflet(layer.bounds_wgs84)}
                  opacity={opacity}
                  zIndex={400}
                />
              ))}

            {showOutlines &&
              masks
                .filter((layer) => !hidden.has(layer.key) && layer.geojson_url)
                .map((layer) => (
                  <VectorOutline key={`${layer.key}-vector`} layer={layer} />
                ))}
          </MapContainer>
        </div>

        <div
          className={`divide-y divide-edge ${
            beside
              ? "@4xl:flex @4xl:min-h-0 @4xl:flex-col @4xl:border-l @4xl:border-edge"
              : ""
          }`}
        >
          {bases.length > 1 && (
            <div className="flex flex-wrap items-center gap-1.5 px-4 py-2.5">
              <span className="text-[10px] uppercase tracking-wider text-ink-faint">
                Base
              </span>
              {bases.map((layer) => (
                <button
                  key={layer.key}
                  type="button"
                  onClick={() => setActiveBase(layer.key)}
                  aria-pressed={layer.key === activeBase}
                  className={`rounded border px-2 py-0.5 text-[10px] transition ${
                    layer.key === activeBase
                      ? "border-signal/50 bg-signal/10 text-signal"
                      : "border-edge text-ink-faint hover:border-edge-hi"
                  }`}
                >
                  {layer.label}
                </button>
              ))}
            </div>
          )}

          {/*
            The layer list is capped and scrolls inside itself.

            A run produces a dozen masks, and each one carries its formula, its
            threshold and how that threshold was chosen. Printed in full that is
            several screens of text between the map and the verdict, which is how
            a page that should read top to bottom became a scroll hunt. So a row
            is one line until asked, and the list has a ceiling: the map stays
            next to the finding instead of being pushed off the top of it.

            Beside the map the ceiling is the map's own height, and this block is
            the part of the column that gives way: the base switch and the
            description keep their lines and the list scrolls.
          */}
          {masks.length > 0 && (
            <div className="@4xl:flex @4xl:min-h-0 @4xl:flex-col">
              <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2">
                <span className="text-[10px] uppercase tracking-wider text-ink-faint">
                  {masks.length} mapped layer{masks.length === 1 ? "" : "s"}
                </span>
                <div className="flex items-center gap-1.5">
                  <button
                    type="button"
                    onClick={() => setShowFormulas((current) => !current)}
                    aria-pressed={showFormulas}
                    className={`rounded border px-2 py-0.5 text-[10px] transition ${
                      showFormulas
                        ? "border-signal/50 bg-signal/10 text-signal"
                        : "border-edge text-ink-faint hover:border-edge-hi"
                    }`}
                  >
                    Formulas
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      setHidden((current) =>
                        current.size === masks.length
                          ? new Set()
                          : new Set(masks.map((layer) => layer.key)),
                      )
                    }
                    className="rounded border border-edge px-2 py-0.5 text-[10px] text-ink-faint transition hover:border-edge-hi"
                  >
                    {hidden.size === masks.length ? "Show all" : "Hide all"}
                  </button>
                </div>
              </div>

              <ul className="max-h-[17rem] divide-y divide-edge overflow-y-auto overscroll-contain border-t border-edge @4xl:max-h-none @4xl:min-h-0">
                {masks.map((layer) => (
                  <LayerRow
                    key={layer.key}
                    layer={layer}
                    visible={!hidden.has(layer.key)}
                    detail={showFormulas}
                    onToggle={() =>
                      setHidden((current) => {
                        const next = new Set(current);
                        if (next.has(layer.key)) next.delete(layer.key);
                        else next.add(layer.key);
                        return next;
                      })
                    }
                  />
                ))}
              </ul>
            </div>
          )}

          {base?.description && (
            <p className="px-4 py-2 text-[10px] leading-snug text-ink-faint">
              Base render: {base.description}
            </p>
          )}
        </div>
      </div>
    </section>
  );
}

function Toggle({
  label,
  on,
  onChange,
}: {
  label: string;
  on: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!on)}
      aria-pressed={on}
      className={`rounded border px-2 py-0.5 text-[10px] transition ${
        on
          ? "border-signal/50 bg-signal/10 text-signal"
          : "border-edge text-ink-faint hover:border-edge-hi"
      }`}
    >
      {label}
    </button>
  );
}

/**
 * One mapped layer: a swatch that toggles it, its name, and its area.
 *
 * The swatch and the disclosure are separate controls because they do different
 * things, and a single row that both hid a layer and unfolded its formula would
 * do the wrong one about half the time.
 *
 * The area is never folded away. It is the number the reader came for, and the
 * formula behind it is one click from it.
 */
function LayerRow({
  layer,
  visible,
  detail,
  onToggle,
}: {
  layer: MapLayer;
  visible: boolean;
  /** Forced open by the list's Formulas control. */
  detail: boolean;
  onToggle: () => void;
}) {
  const [open, setOpen] = useState(false);
  const expanded = detail || open;

  return (
    <li className="flex items-start gap-2.5 px-4 py-2">
      <button
        type="button"
        onClick={onToggle}
        aria-pressed={visible}
        aria-label={`${visible ? "Hide" : "Show"} ${layer.label}`}
        className="mt-1 h-3 w-3 shrink-0 rounded border transition"
        style={{
          backgroundColor: visible ? layer.colour : "transparent",
          borderColor: layer.colour,
        }}
      />
      <div className="min-w-0 flex-1">
        <button
          type="button"
          onClick={() => setOpen((current) => !current)}
          aria-expanded={expanded}
          className="flex w-full items-baseline justify-between gap-2 text-left"
        >
          <span
            className={`truncate text-[11px] ${
              visible ? "text-ink" : "text-ink-faint line-through"
            }`}
          >
            {layer.label}
          </span>
          <span className="tabular shrink-0 text-[11px] text-signal">
            {formatArea(layer.area_km2)}
          </span>
        </button>

        {expanded && (
          <div className="mt-1">
            <p className="text-[10px] leading-snug text-ink-faint">
              {layer.description}
            </p>
            <p className="tabular mt-0.5 break-words text-[9px] text-ink-faint">
              {formatInteger(layer.pixel_count)} px &middot; centre{" "}
              {formatLatLon([
                (layer.bounds_wgs84[0] + layer.bounds_wgs84[2]) / 2,
                (layer.bounds_wgs84[1] + layer.bounds_wgs84[3]) / 2,
              ])}
            </p>
          </div>
        )}
      </div>
    </li>
  );
}

/**
 * Region outlines, fetched as GeoJSON.
 *
 * The raster overlay shows the extent; the vector gives a crisp boundary and
 * carries each region's own measured area, so clicking a polygon says how big
 * that particular region is rather than only the total.
 */
function VectorOutline({ layer }: { layer: MapLayer }) {
  const [data, setData] = useState<unknown | null>(null);

  useEffect(() => {
    let active = true;
    if (!layer.geojson_url) return;
    void fetch(layer.geojson_url)
      .then((response) => (response.ok ? response.json() : null))
      .then((body) => {
        if (active) setData(body);
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, [layer.geojson_url]);

  if (!data) return null;

  return (
    <GeoJSON
      // Remounting on new data is how react-leaflet takes an updated collection.
      key={layer.key}
      data={data as never}
      style={() => ({
        color: layer.colour,
        weight: 1.5,
        opacity: 0.95,
        fillOpacity: 0,
      })}
      onEachFeature={(feature, leafletLayer) => {
        const area = feature.properties?.area_km2 as number | undefined;
        const rank = feature.properties?.rank as number | undefined;
        leafletLayer.bindTooltip(
          `Region ${rank ?? "?"} &middot; ${formatArea(area ?? null)}`,
          { sticky: true, className: "satquery-tooltip" },
        );
      }}
      pointToLayer={(_, latlng) => L.circleMarker(latlng, { radius: 4 })}
    />
  );
}
