import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { initialsOf, useAuth } from "../../lib/auth";
import { usePrefersReducedMotion } from "../../lib/useReducedMotion";

type Section = "profile" | "settings" | null;

/**
 * The account menu for the SatQuery ARC workspace.
 *
 * Profile and Settings open in place rather than on their own routes: there is
 * little enough behind each that a page apiece would be empty, and leaving the
 * workspace to change a setting would drop the search and analysis on screen.
 *
 * Settings offers only what genuinely does something here. A reset that clears
 * the review queue, a way back to the classic investigator, and an honest report
 * of the motion setting; nothing painted on.
 */
export function AccountMenu({ onResetQueue }: { onResetQueue?: () => void | Promise<void> }) {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  const reduceMotion = usePrefersReducedMotion();
  const [open, setOpen] = useState(false);
  const [section, setSection] = useState<Section>(null);
  const holder = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (event: MouseEvent) => {
      if (!holder.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  if (!user) return null;

  const toggle = (next: Exclude<Section, null>) =>
    setSection((current) => (current === next ? null : next));

  return (
    <div className="relative" ref={holder}>
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        aria-haspopup="menu"
        aria-label={`Account menu for ${user.name}`}
        className="flex items-center gap-2 rounded-full border border-edge py-0.5 pl-0.5 pr-1 transition hover:border-signal/50 sm:pr-2.5"
      >
        <span
          className="tabular grid h-7 w-7 place-items-center rounded-full bg-signal/15 text-[11px] font-semibold text-signal"
          aria-hidden="true"
        >
          {initialsOf(user)}
        </span>
        <span className="hidden text-[11px] text-ink-dim md:inline">{user.name}</span>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            role="menu"
            aria-label="Account"
            initial={{ opacity: 0, y: -6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.98 }}
            transition={{ duration: reduceMotion ? 0 : 0.16 }}
            className="panel panel-raised absolute right-0 z-50 mt-2 w-[19rem] max-w-[calc(100vw-1.5rem)] overflow-hidden p-1.5 text-left"
          >
            <div className="border-b border-edge px-2.5 py-2">
              <p className="truncate text-xs font-medium text-ink">{user.name}</p>
              <p className="truncate text-[10px] text-ink-faint">{user.email}</p>
            </div>

            <MenuButton
              label="Profile"
              expanded={section === "profile"}
              onClick={() => toggle("profile")}
            />
            {section === "profile" && (
              <div className="mx-1 mb-1 rounded-lg border border-edge bg-hull/40 px-2.5 py-2">
                <Row label="Name" value={user.name} />
                <Row label="Role" value={user.role} />
                <Row label="Email" value={user.email} />
                <Row
                  label="Signed in"
                  value={new Date(user.signedInAt).toLocaleString("en-GB")}
                />
                <p className="mt-2 border-t border-edge pt-1.5 text-[10px] leading-snug text-uncertain">
                  Demo identity only. It lives in this browser, the server never
                  sees it, and it protects nothing.
                </p>
              </div>
            )}

            <MenuButton
              label="Settings"
              expanded={section === "settings"}
              onClick={() => toggle("settings")}
            />
            {section === "settings" && (
              <div className="mx-1 mb-1 space-y-2 rounded-lg border border-edge bg-hull/40 px-2.5 py-2">
                <SettingRow
                  label="Reset review queue"
                  hint="Clear every candidate and the audit log for a fresh session."
                  action="Reset"
                  onAct={onResetQueue ? () => void onResetQueue() : null}
                />
                <SettingRow
                  label="Classic investigator"
                  hint="Open the original claim-investigation workspace."
                  action="Open"
                  onAct={() => {
                    setOpen(false);
                    navigate("/investigator");
                  }}
                />
                <SettingRow
                  label="Motion"
                  hint="Animations follow your system reduced-motion setting."
                  action={reduceMotion ? "Reduced" : "Full"}
                  onAct={null}
                />
                <p className="border-t border-edge pt-1.5 text-[10px] leading-snug text-ink-faint">
                  Imagery, model choice and credentials are server settings, held
                  in the backend environment rather than here.
                </p>
              </div>
            )}

            <button
              type="button"
              role="menuitem"
              onClick={() => {
                signOut();
                setOpen(false);
                navigate("/");
              }}
              className="mt-0.5 flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-xs text-ink-dim transition hover:bg-panel-hi/50 hover:text-disagree"
            >
              <svg width="13" height="13" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <path d="M6 2H3.5A1.5 1.5 0 0 0 2 3.5v9A1.5 1.5 0 0 0 3.5 14H6" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
                <path d="M10 11l3-3-3-3M13 8H6" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Sign out
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function MenuButton({
  label,
  expanded,
  onClick,
}: {
  label: string;
  expanded: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      role="menuitem"
      onClick={onClick}
      aria-expanded={expanded}
      className="flex w-full items-center justify-between rounded-lg px-2.5 py-2 text-left text-xs text-ink-dim transition hover:bg-panel-hi/50 hover:text-ink"
    >
      {label}
      <span
        className={`text-[9px] text-ink-faint transition-transform ${expanded ? "rotate-90" : ""}`}
        aria-hidden="true"
      >
        &#9656;
      </span>
    </button>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <p className="flex items-baseline justify-between gap-2 text-[10px]">
      <span className="text-ink-faint">{label}</span>
      <span className="tabular min-w-0 truncate text-ink-dim">{value}</span>
    </p>
  );
}

function SettingRow({
  label,
  hint,
  action,
  onAct,
}: {
  label: string;
  hint: string;
  action: string;
  /** Null for a row that reports a state rather than offering a change. */
  onAct: (() => void) | null;
}) {
  const [done, setDone] = useState(false);
  return (
    <div className="flex items-start justify-between gap-2">
      <div className="min-w-0">
        <p className="text-[11px] text-ink">{label}</p>
        <p className="text-[10px] leading-snug text-ink-faint">{hint}</p>
      </div>
      {onAct ? (
        <button
          type="button"
          onClick={() => {
            onAct();
            setDone(true);
            window.setTimeout(() => setDone(false), 1600);
          }}
          className="shrink-0 rounded border border-edge px-2 py-0.5 text-[10px] text-ink-dim transition hover:border-signal/50 hover:text-signal"
        >
          {done ? "Done" : action}
        </button>
      ) : (
        <span className="shrink-0 rounded border border-edge px-2 py-0.5 text-[10px] text-ink-faint">
          {action}
        </span>
      )}
    </div>
  );
}
