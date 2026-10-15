import { useCallback, useEffect, useRef, useState } from 'react';

// Browser-native Web Speech API (no AI provider, no API key).
// Chrome/Edge ship webkitSpeechRecognition; Firefox/Safari support varies.
const SpeechRecognitionCtor =
  typeof window !== 'undefined'
    ? window.SpeechRecognition || window.webkitSpeechRecognition
    : undefined;

const NO_SPEECH_MSG = 'No speech detected. Type instead.';
const SILENCE_MS = 5000; // auto-stop after ~5s without new speech
const TICK_MS = 250;     // how often the silence timer checks the clock

/**
 * Speech-to-text hook used by both voice workflows (report retrieval and
 * clinical dictation).
 *
 * - `start(seed?)` begins listening; any pre-typed text is used as a seed so
 *   spoken words append to it and both stay visible in the input.
 * - Speech auto-stops after ~5s of silence: the last activity timestamp is
 *   updated on every recognition result, and a timer flushes any in-progress
 *   interim words once (in `onend`) before the existing auto-submit fires.
 * - The stop button stays as a manual override; it also ends the session and
 *   submits the full transcript.
 * - Fatal errors (permission denied, network, …) and empty/no-speech sessions
 *   never trigger a submit; they surface via `error` so the typed-input
 *   fallback stays primary.
 */
export function useSpeechRecognition({ onFinalTranscript, lang = 'en-IN' } = {}) {
  const supported = Boolean(SpeechRecognitionCtor);

  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState(''); // finalised text this session
  const [interim, setInterim] = useState('');       // live partial words
  const [error, setError] = useState(null);

  const recRef = useRef(null);        // live SpeechRecognition instance
  const finalRef = useRef('');        // seed + finalised chunks
  const interimRef = useRef('');      // live interim words (for the silence flush)
  const lastActivityRef = useRef(0);  // timestamp of the most recent result
  const timerRef = useRef(null);      // silence-timer interval id
  const spokenRef = useRef(false);    // at least one final speech chunk arrived
  const abortedRef = useRef(false);   // fatal error / cancel: never auto-submit
  const finalizeRef = useRef(null);

  // Keep the latest callback available to recognition handlers that were
  // created earlier (they must see the freshest closure, not a stale one).
  useEffect(() => {
    finalizeRef.current = onFinalTranscript;
  }, [onFinalTranscript]);

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const start = useCallback((seed = '') => {
    if (!SpeechRecognitionCtor || recRef.current) return;
    setError(null);
    const seeded = seed.trim();
    finalRef.current = seeded;
    interimRef.current = '';
    spokenRef.current = false;
    abortedRef.current = false;
    lastActivityRef.current = Date.now();
    setTranscript(seeded);
    setInterim('');

    const rec = new SpeechRecognitionCtor();
    rec.lang = lang;
    rec.continuous = true;
    rec.interimResults = true;

    rec.onresult = (event) => {
      lastActivityRef.current = Date.now(); // any result resets the silence clock
      let finalPart = '';
      let interimPart = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) finalPart += result[0].transcript;
        else interimPart += result[0].transcript;
      }
      if (finalPart.trim()) {
        spokenRef.current = true;
        finalRef.current = `${finalRef.current} ${finalPart}`.trim();
      }
      interimRef.current = interimPart.trim();
      setTranscript(finalRef.current);
      setInterim(interimPart.trim());
    };

    rec.onerror = (event) => {
      if (event.error === 'aborted') return;  // intentional stop; onend handles it
      if (event.error === 'no-speech') return; // handled as an empty session in onend
      abortedRef.current = true;               // fatal; never auto-submit
      clearTimer();
      if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
        setError('Microphone permission denied. Type instead.');
      } else if (event.error === 'network') {
        setError('Speech service unavailable. Type instead.');
      } else {
        setError('Speech recognition error. Type instead.');
      }
    };

    rec.onend = () => {
      clearTimer();
      // Flush in-progress interim words exactly once. If the browser already
      // finalised them via onresult, interimRef is empty and this is a no-op
      // (no duplication); if it ended without finalising, nothing is lost.
      const pending = interimRef.current.trim();
      if (pending) {
        finalRef.current = `${finalRef.current} ${pending}`.trim();
        spokenRef.current = true; // interim words count as heard speech
        interimRef.current = '';
        setTranscript(finalRef.current);
      }
      const text = finalRef.current.trim();
      const wasAborted = abortedRef.current;
      const heardSpeech = spokenRef.current;
      recRef.current = null;
      setListening(false);
      setInterim('');
      if (wasAborted) return;
      if (text && heardSpeech) {
        finalizeRef.current?.(text); // speech ended → caller auto-submits
      } else {
        setError(NO_SPEECH_MSG);
      }
    };

    recRef.current = rec;
    setListening(true);
    try {
      rec.start();
    } catch {
      recRef.current = null;
      setListening(false);
      clearTimer();
      setError('Could not start the microphone. Type instead.');
      return;
    }

    // Silence auto-stop: after SILENCE_MS with no new result, stop the session.
    // onend then flushes any interim text and auto-submits the full transcript.
    clearTimer();
    timerRef.current = setInterval(() => {
      if (!recRef.current) {
        clearTimer();
        return;
      }
      if (Date.now() - lastActivityRef.current >= SILENCE_MS) {
        clearTimer();
        try {
          recRef.current?.stop(); // → onend flushes interim and submits
        } catch {
          // Recognition already stopped
        }
      }
    }, TICK_MS);
  }, [lang, clearTimer]);

  const stop = useCallback(() => {
    // Manual override: graceful stop; onend flushes interim and auto-submits.
    clearTimer();
    recRef.current?.stop();
  }, [clearTimer]);

  const cancel = useCallback(() => {
    // Abort without submitting (keeps whatever was typed intact).
    clearTimer();
    abortedRef.current = true;
    try {
      recRef.current?.abort();
    } catch {
      // Recognition already stopped
    }
    recRef.current = null;
    interimRef.current = '';
    setListening(false);
    setInterim('');
  }, [clearTimer]);

  // Abort any live session on unmount; never submit from a dead panel.
  useEffect(() => () => {
    clearTimer();
    abortedRef.current = true;
    try {
      recRef.current?.abort();
    } catch {
      // Recognition already stopped
    }
    recRef.current = null;
  }, [clearTimer]);

  return { supported, listening, transcript, interim, error, start, stop, cancel };
}
