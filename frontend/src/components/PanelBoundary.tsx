import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  /** What failed, in the user's terms, for the message. */
  panel: string;
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * Keeps one broken panel from taking the page with it.
 *
 * React unmounts the entire tree on an unhandled render error, so a single
 * component reading a field the server did not send turns the whole workspace
 * into a blank white page with nothing on it to explain why. That is exactly how
 * the approve-and-run flow failed: the run finished, a panel downstream of it
 * threw, and everything vanished including the trace that would have shown what
 * happened.
 *
 * A boundary per panel is the right granularity. The measurements, the map, the
 * verdict and the packet are independent readings of the same run, and losing one
 * is not a reason to lose the others. The failure is shown rather than swallowed,
 * because a panel that silently disappeared would be worse than one that says it
 * broke.
 */
export class PanelBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // Logged rather than reported home: this is a local development tool, and the
    // console is where the developer already is.
    console.error(`[${this.props.panel}] failed to render`, error, info);
  }

  render(): ReactNode {
    const { error } = this.state;
    if (error === null) return this.props.children;

    return (
      <section
        className="panel border border-disagree/40 bg-disagree/5 px-4 py-3"
        aria-label={`${this.props.panel} could not be displayed`}
      >
        <p className="text-[10px] uppercase tracking-wider text-disagree">
          {this.props.panel} could not be displayed
        </p>
        <p className="mt-1 text-[11px] leading-snug text-ink-dim">
          This panel hit an error while rendering. The rest of the run is
          unaffected and the measurements behind it are still in the trace below.
        </p>
        <p className="tabular mt-1.5 text-[10px] leading-snug text-ink-faint">
          {error.message}
        </p>
        <button
          type="button"
          onClick={() => this.setState({ error: null })}
          className="mt-2 rounded-lg border border-edge-hi bg-hull/60 px-3 py-1.5 text-[11px] text-ink transition hover:border-signal/50 hover:text-signal"
        >
          Try this panel again
        </button>
      </section>
    );
  }
}
