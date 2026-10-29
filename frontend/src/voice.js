/**
 * Browser voice helpers — Web Speech API only, no extra deps.
 *
 * Chrome quirks this module works around:
 * - speak() should start from a user click (no await before first speak)
 * - synth can sit paused/stuck with no audio until resume()
 * - cancel()+speak in the same tick can drop the new utterance
 * - assigning a bad voice can produce silence with no error
 * - long utterances can die mid-way (chunk + keep-alive)
 */

let keepAliveTimer = null;
let watchdogTimer = null;
let activeSession = null;
let voicesWarmed = false;

export function getSpeechRecognition() {
  if (typeof window === "undefined") return null;
  return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

export function speechRecognitionSupported() {
  return Boolean(getSpeechRecognition());
}

export function speechSynthesisSupported() {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

/** Strip citation markers and trim for clearer TTS. */
export function answerForSpeech(text) {
  return String(text || "")
    .replace(/\[\d+\]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function clearKeepAlive() {
  if (keepAliveTimer != null) {
    window.clearInterval(keepAliveTimer);
    keepAliveTimer = null;
  }
}

function clearWatchdog() {
  if (watchdogTimer != null) {
    window.clearTimeout(watchdogTimer);
    watchdogTimer = null;
  }
}

function startKeepAlive() {
  clearKeepAlive();
  if (!speechSynthesisSupported()) return;
  keepAliveTimer = window.setInterval(() => {
    try {
      const synth = window.speechSynthesis;
      if (!synth.speaking) return;
      synth.pause();
      synth.resume();
    } catch {
      /* ignore */
    }
  }, 10000);
}

/** Call early (and on first click) so getVoices() is usually populated. */
export function warmSpeechVoices() {
  if (!speechSynthesisSupported()) return;
  try {
    const synth = window.speechSynthesis;
    synth.getVoices();
    if (!voicesWarmed) {
      voicesWarmed = true;
      synth.addEventListener("voiceschanged", () => {
        synth.getVoices();
      });
    }
  } catch {
    /* ignore */
  }
}

/** Nudge autoplay / audio policies before TTS (same user gesture). */
export function unlockAudio() {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (AudioCtx) {
      const ctx = new AudioCtx();
      if (ctx.state === "suspended") {
        void ctx.resume();
      }
      // Tiny silent buffer helps some Chromium builds treat the tab as audible.
      const buffer = ctx.createBuffer(1, 1, 22050);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      source.start(0);
      window.setTimeout(() => {
        try {
          ctx.close();
        } catch {
          /* ignore */
        }
      }, 500);
    }
  } catch {
    /* ignore */
  }
}

export function stopSpeaking() {
  if (!speechSynthesisSupported()) return;
  if (activeSession) {
    activeSession.stopped = true;
    activeSession = null;
  }
  clearKeepAlive();
  clearWatchdog();
  try {
    window.speechSynthesis.cancel();
  } catch {
    /* ignore */
  }
}

function pickVoice(voices, { localOnly = false } = {}) {
  if (!voices?.length) return null;
  const english = voices.filter((v) => /^en([-_]|$)/i.test(v.lang || ""));
  const pool = english.length ? english : [...voices];
  const local = pool.filter((v) => v.localService);
  if (localOnly) return local[0] || null;
  return (
    local.find((v) => /en(-|_)?US/i.test(v.lang || "")) ||
    local[0] ||
    pool.find((v) => /Samantha|Alex|Daniel|Karen|Moira|Rishi|Microsoft/i.test(v.name || "")) ||
    pool[0] ||
    null
  );
}

/** Split into sentence-ish chunks under maxLen for Chrome's long-utterance cutoff. */
export function chunkForSpeech(text, maxLen = 160) {
  const clean = answerForSpeech(text);
  if (!clean) return [];
  if (clean.length <= maxLen) return [clean];

  const parts = clean.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [clean];
  const chunks = [];
  let buf = "";
  for (const part of parts) {
    const next = part.trim();
    if (!next) continue;
    if (!buf) {
      buf = next;
      continue;
    }
    if (`${buf} ${next}`.length <= maxLen) {
      buf = `${buf} ${next}`;
    } else {
      chunks.push(buf);
      buf = next;
    }
  }
  if (buf) chunks.push(buf);

  const out = [];
  for (const chunk of chunks) {
    if (chunk.length <= maxLen) {
      out.push(chunk);
      continue;
    }
    let i = 0;
    while (i < chunk.length) {
      let end = Math.min(i + maxLen, chunk.length);
      if (end < chunk.length) {
        const space = chunk.lastIndexOf(" ", end);
        if (space > i + Math.floor(maxLen * 0.4)) end = space;
      }
      const piece = chunk.slice(i, end).trim();
      if (piece) out.push(piece);
      i = end;
      while (i < chunk.length && chunk[i] === " ") i += 1;
    }
  }
  return out;
}

function makeUtterance(text, voice) {
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1;
  utterance.pitch = 1;
  utterance.volume = 1;
  utterance.lang = voice?.lang || "en-US";
  if (voice) utterance.voice = voice;
  return utterance;
}

function kickSynth(synth) {
  try {
    synth.resume();
  } catch {
    /* ignore */
  }
}

/**
 * Speak text. Invoke directly from a click/tap handler.
 * Returns { stop } or null.
 *
 * onStart fires only when audio actually begins (utterance.onstart).
 */
export function speakText(text, { onStart, onEnd, onError } = {}) {
  if (!speechSynthesisSupported()) {
    onError?.(new Error("Speech playback is not supported in this browser. Try Chrome or Edge."));
    return null;
  }

  warmSpeechVoices();
  unlockAudio();

  const chunks = chunkForSpeech(text);
  if (!chunks.length) {
    onError?.(new Error("Nothing to read aloud."));
    return null;
  }

  const synth = window.speechSynthesis;

  if (activeSession) {
    activeSession.stopped = true;
    activeSession = null;
  }
  clearKeepAlive();
  clearWatchdog();

  // Only cancel if something is already queued — cancel()+speak same tick
  // often drops the new utterance on Chrome.
  try {
    if (synth.speaking || synth.pending) synth.cancel();
  } catch {
    /* ignore */
  }

  const session = {
    id: Symbol("speak"),
    stopped: false,
    started: false,
    attempt: 0,
  };
  activeSession = session;

  const finish = (err) => {
    if (session.stopped) return;
    session.stopped = true;
    if (activeSession === session) activeSession = null;
    clearKeepAlive();
    clearWatchdog();
    try {
      synth.cancel();
    } catch {
      /* ignore */
    }
    if (err) onError?.(err);
    onEnd?.();
  };

  const handle = {
    stop() {
      if (session.stopped) return;
      session.stopped = true;
      if (activeSession === session) activeSession = null;
      clearKeepAlive();
      clearWatchdog();
      try {
        synth.cancel();
      } catch {
        /* ignore */
      }
      onEnd?.();
    },
  };

  let index = 0;
  let activeVoice = null;

  const speakNext = () => {
    if (session.stopped || activeSession !== session) return;
    if (index >= chunks.length) {
      finish();
      return;
    }

    const utterance = makeUtterance(chunks[index], activeVoice);

    utterance.onstart = () => {
      if (session.stopped || activeSession !== session) return;
      if (!session.started) {
        session.started = true;
        clearWatchdog();
        startKeepAlive();
        onStart?.();
      }
      kickSynth(synth);
    };

    utterance.onend = () => {
      if (session.stopped || activeSession !== session) return;
      index += 1;
      if (index >= chunks.length) {
        finish();
        return;
      }
      speakNext();
    };

    utterance.onerror = (event) => {
      if (session.stopped || activeSession !== session) return;
      const code = event?.error || "";
      if (code === "canceled" || code === "interrupted") {
        return;
      }
      const message =
        code === "not-allowed"
          ? "Browser blocked speech. Click Read aloud again, and allow sound for this site."
          : code === "synthesis-failed" || code === "synthesis-unavailable"
            ? "Could not play speech on this device. Try Chrome or Safari."
            : `Could not read aloud${code ? ` (${code})` : ""}.`;
      finish(new Error(message));
    };

    try {
      synth.speak(utterance);
      kickSynth(synth);
    } catch (err) {
      finish(err instanceof Error ? err : new Error("Could not read aloud."));
    }
  };

  const startAttempt = (voice) => {
    if (session.stopped || activeSession !== session) return;
    index = 0;
    activeVoice = voice;
    speakNext();
    kickSynth(synth);
  };

  // Attempt 0: browser default voice (most reliable).
  startAttempt(null);

  // If audio never starts, unstick Chrome and retry with a local voice.
  watchdogTimer = window.setTimeout(() => {
    if (session.stopped || activeSession !== session) return;
    if (session.started || synth.speaking) {
      kickSynth(synth);
      return;
    }

    session.attempt += 1;
    try {
      synth.cancel();
    } catch {
      /* ignore */
    }

    const local =
      pickVoice(synth.getVoices(), { localOnly: true }) || pickVoice(synth.getVoices());
    startAttempt(local);

    watchdogTimer = window.setTimeout(() => {
      if (session.stopped || activeSession !== session) return;
      if (session.started || synth.speaking) {
        kickSynth(synth);
        return;
      }
      finish(
        new Error(
          "No audio from this browser’s speech engine. Unmute the tab, check system volume, and try Chrome or Safari.",
        ),
      );
    }, 800);
  }, 500);

  return handle;
}

/**
 * Create a recognition session. Caller must stop() on unmount.
 * onResult({ interim, final, committed, display })
 */
export function createRecognition({
  lang = "en-US",
  continuous = true,
  interimResults = true,
  onResult,
  onError,
  onEnd,
} = {}) {
  const SpeechRecognition = getSpeechRecognition();
  if (!SpeechRecognition) return null;

  const recognition = new SpeechRecognition();
  recognition.lang = lang;
  recognition.continuous = continuous;
  recognition.interimResults = interimResults;
  let committed = "";

  recognition.onresult = (event) => {
    let interim = "";
    let finalChunk = "";
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const piece = event.results[i][0]?.transcript || "";
      if (event.results[i].isFinal) finalChunk += piece;
      else interim += piece;
    }
    if (finalChunk) {
      committed = `${committed} ${finalChunk}`.replace(/\s+/g, " ").trim();
    }
    onResult?.({
      interim: interim.trim(),
      final: finalChunk.trim(),
      committed,
      display: `${committed}${interim ? ` ${interim}` : ""}`.replace(/\s+/g, " ").trim(),
    });
  };

  recognition.onerror = (event) => {
    if (event.error === "aborted" || event.error === "no-speech") return;
    const messages = {
      "not-allowed": "Microphone permission denied. Allow mic access to speak your question.",
      "service-not-allowed": "Speech recognition is blocked in this browser or context.",
      network: "Speech recognition needs a network connection in this browser.",
      "audio-capture": "No microphone found.",
    };
    onError?.(new Error(messages[event.error] || `Voice input error: ${event.error}`));
  };

  recognition.onend = () => onEnd?.();

  return {
    start() {
      committed = "";
      recognition.start();
    },
    stop() {
      try {
        recognition.stop();
      } catch {
        /* already stopped */
      }
    },
    abort() {
      try {
        recognition.abort();
      } catch {
        /* already stopped */
      }
    },
  };
}

if (typeof window !== "undefined") {
  warmSpeechVoices();
}

/**
 * Play answer audio via the backend (macOS say → WAV). Reliable when browser
 * SpeechSynthesis is silent. Returns { stop } or throws.
 */
export async function speakViaServer(text, { onStart, onEnd, onError } = {}) {
  const { api } = await import("./api");
  const clean = answerForSpeech(text);
  if (!clean) {
    const err = new Error("Nothing to read aloud.");
    onError?.(err);
    throw err;
  }

  stopSpeaking();

  let audio = null;
  let objectUrl = null;
  let stopped = false;

  const cleanup = () => {
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }
    audio = null;
  };

  const handle = {
    stop() {
      if (stopped) return;
      stopped = true;
      try {
        audio?.pause();
      } catch {
        /* ignore */
      }
      cleanup();
      onEnd?.();
    },
  };

  try {
    const blob = await api.tts(clean);
    if (stopped) return handle;
    objectUrl = URL.createObjectURL(blob);
    audio = new Audio(objectUrl);
    audio.onended = () => {
      if (stopped) return;
      stopped = true;
      cleanup();
      onEnd?.();
    };
    audio.onerror = () => {
      if (stopped) return;
      stopped = true;
      cleanup();
      const err = new Error("Could not play the generated audio.");
      onError?.(err);
      onEnd?.();
    };
    onStart?.();
    await audio.play();
    return handle;
  } catch (err) {
    cleanup();
    const error = err instanceof Error ? err : new Error("Speech synthesis failed");
    onError?.(error);
    throw error;
  }
}
