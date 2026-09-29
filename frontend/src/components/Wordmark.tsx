/** The SatQuery ARC wordmark: an aperture glyph plus the product name. */
export function Wordmark({ compact = false }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-2.5">
      <svg width="26" height="26" viewBox="0 0 24 24" aria-hidden="true">
        <circle
          cx="12"
          cy="12"
          r="10.25"
          fill="none"
          stroke="var(--color-signal)"
          strokeWidth="1.25"
          opacity="0.8"
        />
        <circle cx="12" cy="12" r="5.5" fill="none" stroke="var(--color-signal)" strokeWidth="1" opacity="0.5" />
        <circle cx="12" cy="12" r="1.75" fill="var(--color-signal)" />
        <path d="M12 1.75v3.5M12 18.75v3.5M1.75 12h3.5M18.75 12h3.5" stroke="var(--color-signal)" strokeWidth="1.25" opacity="0.7" />
      </svg>
      <div className="flex flex-col leading-none">
        <span className="whitespace-nowrap text-[15px] font-semibold tracking-tight text-ink">
          SatQuery<span className="text-signal"> ARC</span>
        </span>
        {!compact && (
          <span className="mt-0.5 text-[10px] uppercase tracking-[0.14em] text-ink-faint">
            Search &middot; Analyse &middot; Verify &middot; Discover
          </span>
        )}
      </div>
    </div>
  );
}
