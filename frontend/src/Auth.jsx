import { useEffect, useRef, useState } from "react";
import { api } from "./api";

const BEATS = [
  { word: "Find", line: "Search the literature by what you mean." },
  { word: "Library", line: "Every paper stays on your shelf." },
  { word: "Cite", line: "Ask — then open the source behind it." },
];

/** Brand hold → beats → unlock continue (ms). Skippable anytime. */
const INTRO_TIMING = {
  brandHold: 1200,
  beatGap: 1400,
  readyHold: 400,
};

const GIS_SCRIPT = "https://accounts.google.com/gsi/client";

function prefersReducedMotion() {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function loadGoogleScript() {
  if (typeof document === "undefined") return Promise.resolve();
  if (window.google?.accounts?.id) return Promise.resolve();
  const existing = document.querySelector(`script[src="${GIS_SCRIPT}"]`);
  if (existing) {
    return new Promise((resolve, reject) => {
      existing.addEventListener("load", () => resolve(), { once: true });
      existing.addEventListener("error", () => reject(new Error("Google script failed to load.")), {
        once: true,
      });
      if (window.google?.accounts?.id) resolve();
    });
  }
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = GIS_SCRIPT;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("Google script failed to load."));
    document.head.appendChild(script);
  });
}

export function Logo({ size = "nav" }) {
  return (
    <div className={`logo logo-${size}`} role="img" aria-label="Lexicon Gate">
      <span className="logo-mark" aria-hidden="true">
        <span className="gate-lintel" />
        <span className="gate-post" />
        <span className="gate-post" />
      </span>
      <span className="logo-word">
        <span className="logo-lexicon">Lexicon</span>
        <span className="logo-gate">Gate</span>
      </span>
    </div>
  );
}

export default function AuthScreen({ onSignedIn }) {
  const [mode, setMode] = useState("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [showIntro, setShowIntro] = useState(true);
  const [reduced] = useState(prefersReducedMotion);
  const [beatIndex, setBeatIndex] = useState(reduced ? BEATS.length : -1);
  const [introReady, setIntroReady] = useState(Boolean(reduced));
  const [leaving, setLeaving] = useState(false);
  const [googleClientId, setGoogleClientId] = useState("");
  const googleBtnRef = useRef(null);

  const isSignup = mode === "signup";
  const googleEnabled = Boolean(googleClientId) && !showIntro;

  function finishIntro() {
    if (leaving) return;
    setLeaving(true);
    window.setTimeout(() => setShowIntro(false), reduced ? 0 : 520);
  }

  useEffect(() => {
    let cancelled = false;
    api
      .config()
      .then((cfg) => {
        if (!cancelled) setGoogleClientId(String(cfg?.google_client_id || "").trim());
      })
      .catch(() => {
        if (!cancelled) setGoogleClientId("");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!showIntro || reduced) return undefined;

    const timers = [];
    let elapsed = INTRO_TIMING.brandHold;

    BEATS.forEach((_, index) => {
      timers.push(window.setTimeout(() => setBeatIndex(index), elapsed));
      elapsed += INTRO_TIMING.beatGap;
    });

    timers.push(
      window.setTimeout(() => {
        setBeatIndex(BEATS.length);
        setIntroReady(true);
      }, elapsed + INTRO_TIMING.readyHold)
    );

    return () => timers.forEach((id) => window.clearTimeout(id));
  }, [showIntro, reduced]);

  useEffect(() => {
    if (!showIntro) return undefined;
    function onKey(event) {
      if (event.key === "Enter" && introReady) {
        event.preventDefault();
        finishIntro();
      }
      if (event.key === "Escape") {
        event.preventDefault();
        finishIntro();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [showIntro, introReady, leaving]);

  useEffect(() => {
    if (!googleEnabled || !googleBtnRef.current) return undefined;
    let cancelled = false;

    async function mountGoogle() {
      try {
        await loadGoogleScript();
        if (cancelled || !window.google?.accounts?.id || !googleBtnRef.current) return;

        window.google.accounts.id.initialize({
          client_id: googleClientId,
          callback: async (response) => {
            if (!response?.credential) {
              setError("Google sign-in was cancelled.");
              return;
            }
            setBusy(true);
            setError("");
            try {
              await api.googleLogin({ credential: response.credential });
              const confirmed = await api.me();
              onSignedIn(confirmed);
            } catch (err) {
              setError(err.message || "Google sign-in failed.");
            } finally {
              setBusy(false);
            }
          },
          auto_select: false,
          cancel_on_tap_outside: true,
        });

        googleBtnRef.current.innerHTML = "";
        window.google.accounts.id.renderButton(googleBtnRef.current, {
          type: "standard",
          theme: "outline",
          size: "large",
          text: isSignup ? "signup_with" : "continue_with",
          shape: "pill",
          width: Math.min(340, googleBtnRef.current.clientWidth || 320),
          logo_alignment: "left",
        });
      } catch {
        if (!cancelled) setGoogleClientId("");
      }
    }

    mountGoogle();
    return () => {
      cancelled = true;
    };
  }, [googleEnabled, googleClientId, isSignup, onSignedIn]);

  async function submit(event) {
    event.preventDefault();
    setError("");
    if (isSignup && password !== confirm) {
      setError("Passwords do not match.");
      return;
    }
    setBusy(true);
    try {
      await (isSignup
        ? api.signup({ email, password, name })
        : api.login({ email, password }));
      const confirmed = await api.me();
      onSignedIn(confirmed);
    } catch (err) {
      setError(err.message || "Could not sign in.");
    } finally {
      setBusy(false);
    }
  }

  const pathProgress = introReady
    ? 1
    : Math.max(0, Math.min(1, (beatIndex + 1) / BEATS.length));
  const activeBeat = beatIndex >= 0 && beatIndex < BEATS.length ? BEATS[beatIndex] : null;
  const lineText = activeBeat
    ? activeBeat.line
    : introReady
      ? "Your research desk is ready."
      : "\u00a0";

  return (
    <div className="auth-screen">
      <div className="auth-orbs" aria-hidden="true">
        <span className="shape shape-sphere shape-a is-near" />
        <span className="shape shape-torus shape-c is-mid" />
        <span className="shape shape-sphere shape-b is-mid" />
        <span className="shape shape-torus shape-d is-near" />
        <span className="shape shape-sphere shape-e is-near" />
        <span className="shape shape-torus shape-intro-ring" />
      </div>
      {showIntro && (
        <div
          className={`intro-stage${reduced ? " is-reduced" : ""}${leaving ? " is-leaving" : ""}${introReady ? " is-ready" : ""}${beatIndex === 2 ? " is-cite" : ""}`}
          role="dialog"
          aria-label="Welcome to Lexicon Gate"
          aria-live="polite"
        >
          <Logo size="hero" />

          <h1 className="intro-headline">
            Where research
            <span className="intro-headline-line">keeps its sources</span>
          </h1>

          <div className="intro-path" aria-hidden="true">
            <span
              className="intro-path-fill"
              style={{ transform: `scaleX(${pathProgress})` }}
            />
            <ol className="intro-path-nodes">
              {BEATS.map((beat, index) => {
                const state =
                  beatIndex > index || introReady
                    ? "is-done"
                    : beatIndex === index
                      ? "is-active"
                      : "";
                return (
                  <li key={beat.word} className={`intro-path-node ${state}`}>
                    <span className="intro-path-dot" />
                    <span className="intro-path-label">{beat.word}</span>
                  </li>
                );
              })}
            </ol>
          </div>

          <p className="intro-line" key={activeBeat?.word || (introReady ? "ready" : "wait")}>
            {lineText}
          </p>

          <div className="intro-actions">
            {introReady ? (
              <button type="button" className="primary intro-continue" onClick={finishIntro}>
                <span>Enter Lexicon Gate</span>
                <span className="intro-continue-arrow" aria-hidden="true" />
              </button>
            ) : (
              <button type="button" className="ghost intro-skip" onClick={finishIntro}>
                Skip intro
              </button>
            )}
          </div>
        </div>
      )}
      <div
        className={`auth-card ${isSignup ? "is-signup" : "is-login"}${showIntro ? " is-behind" : ""}`}
        aria-hidden={showIntro}
      >
        <Logo size="hero" />
        <p className="muted auth-lead">
          {isSignup ? "Create an account to use your library." : "Log in to continue."}
        </p>

        {googleClientId ? (
          <div className="auth-google-block">
            <div
              ref={googleBtnRef}
              className="auth-google-btn"
              aria-label={isSignup ? "Sign up with Google" : "Continue with Google"}
            />
            <div className="auth-divider" role="presentation">
              <span>or</span>
            </div>
          </div>
        ) : null}

        <form className="auth-form" onSubmit={submit}>
          <div className={`fold ${isSignup ? "open" : ""}`}>
            <div className="fold-inner">
              <label>
                Name
                <input
                  autoComplete="name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Optional"
                  tabIndex={isSignup && !showIntro ? 0 : -1}
                />
              </label>
            </div>
          </div>
          <label>
            Email
            <input
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              tabIndex={showIntro ? -1 : 0}
            />
          </label>
          <label>
            Password
            <input
              type="password"
              autoComplete={isSignup ? "new-password" : "current-password"}
              required
              minLength={8}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              tabIndex={showIntro ? -1 : 0}
            />
          </label>
          <div className={`fold ${isSignup ? "open" : ""}`}>
            <div className="fold-inner">
              <label>
                Confirm password
                <input
                  type="password"
                  autoComplete="new-password"
                  required={isSignup}
                  minLength={8}
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  tabIndex={isSignup && !showIntro ? 0 : -1}
                />
              </label>
            </div>
          </div>
          {error && <p className="auth-error">{error}</p>}
          <button className={`primary${busy ? " is-busy" : ""}`} type="submit" disabled={busy || showIntro}>
            {busy ? "Please wait…" : isSignup ? "Sign up" : "Log in"}
          </button>
        </form>
        <p className="auth-switch">
          {isSignup ? "Already have an account?" : "Need an account?"}{" "}
          <button
            type="button"
            className="linkish"
            tabIndex={showIntro ? -1 : 0}
            onClick={() => {
              setError("");
              setMode(isSignup ? "login" : "signup");
            }}
          >
            {isSignup ? "Log in" : "Sign up"}
          </button>
        </p>
      </div>
    </div>
  );
}
