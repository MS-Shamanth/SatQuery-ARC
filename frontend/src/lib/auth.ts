import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";

/**
 * A demo identity, held in this browser and nowhere else.
 *
 * This is deliberately not authentication, and the distinction matters enough to
 * put at the top of the file rather than in a comment further down.
 *
 * There is no password, no token, no server call, and no check of any kind. The
 * backend does not know this exists and will serve every request whether or not
 * anyone has "signed in". Nothing here protects any route, any session, or any
 * piece of data, and it must never be mistaken for something that does: if
 * SatQuery ever needs real accounts, they belong on the server with the session
 * store, and this module should be deleted rather than extended.
 *
 * What it is for is the demonstration. A reviewer opening the app sees a way in
 * that looks like the product it would be, the Studio gets a profile menu in the
 * corner where one belongs, and the walkthrough has a beginning. One button, no
 * form, because a login form in a demo is a form nobody wants to fill in twice.
 */

const KEY = "satquery.demo.identity.v1";

export interface DemoUser {
  name: string;
  role: string;
  /** Initials for the avatar, derived rather than stored. */
  email: string;
  signedInAt: string;
}

/** The one account. Named for what it is, so no one mistakes it for a real user. */
export const DEMO_USER: Omit<DemoUser, "signedInAt"> = {
  name: "Demo Analyst",
  role: "Remote sensing analyst",
  email: "demo@satquery.local",
};

export function initialsOf(user: DemoUser): string {
  const parts = user.name.trim().split(/\s+/).slice(0, 2);
  return parts.map((part) => part[0]?.toUpperCase() ?? "").join("") || "DA";
}

/*
  localStorage access is wrapped because it throws outright in a Safari private
  window and in some embedded webviews. A demo identity that cannot be remembered
  is a minor annoyance; one that takes the page down on load is not.
*/
function read(): DemoUser | null {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<DemoUser>;
    if (!parsed?.name) return null;
    return {
      name: parsed.name,
      role: parsed.role ?? DEMO_USER.role,
      email: parsed.email ?? DEMO_USER.email,
      signedInAt: parsed.signedInAt ?? new Date().toISOString(),
    };
  } catch {
    return null;
  }
}

function write(user: DemoUser | null): void {
  try {
    if (user) window.localStorage.setItem(KEY, JSON.stringify(user));
    else window.localStorage.removeItem(KEY);
  } catch {
    /* storage unavailable; the identity just does not persist */
  }
}

export interface DemoAuth {
  user: DemoUser | null;
  signedIn: boolean;
  signIn: () => DemoUser;
  signOut: () => void;
}

export function useDemoAuth(): DemoAuth {
  const [user, setUser] = useState<DemoUser | null>(() => read());

  const signIn = useCallback(() => {
    const next: DemoUser = { ...DEMO_USER, signedInAt: new Date().toISOString() };
    setUser(next);
    write(next);
    return next;
  }, []);

  const signOut = useCallback(() => {
    setUser(null);
    write(null);
  }, []);

  return useMemo(
    () => ({ user, signedIn: user !== null, signIn, signOut }),
    [user, signIn, signOut],
  );
}

/*
  Shared across routes, like the tour. The default is inert rather than null so a
  page rendered on its own in a test does not need a provider to mount.
*/
const INERT: DemoAuth = {
  user: null,
  signedIn: false,
  signIn: () => ({ ...DEMO_USER, signedInAt: new Date().toISOString() }),
  signOut: () => {},
};

export const AuthContext = createContext<DemoAuth>(INERT);

export function useAuth(): DemoAuth {
  return useContext(AuthContext);
}
