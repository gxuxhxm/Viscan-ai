import { useCallback, useEffect, useRef, useState } from 'react';

// Browser-native Web Speech API (no AI provider, no API key).
// Chrome/Edge ship webkitSpeechRecognition; Firefox/Safari support varies.
const SpeechRecognitionCtor =
  typeof window !== 'undefined'
    ? window.SpeechRecognition || window.webkitSpeechRecognition
    : undefined;

const NO_SPEECH_MSG = 'No speech detected — type instead.';

/**
 * Speech-to-text hook used by both voice workflows (report retrieval and
 * clinical dictation).
 *
 * - `start(seed?)` begins listening; any pre-typed text is used as a seed so
 *   spoken words append to it and both stay visible in the input.
 * - When the user stops speaking (silence, browser auto-end, or the stop
 *   button), the session ends gracefully and `onFinalTranscript(fullText)`
 *   fires once with the complete transcript — callers auto-submit from there.
 * - Fatal errors (permission denied, network, …) never trigger a submit;
 *   they surface via `error` so the typed-input fallback stays primary.
 */
export function useSpeechRecognition({ onFinalTranscript, lang = 'en-IN' } = {}) {
  const supported = Boolean(SpeechRecognitionCtor);

  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState(''); // finalised text this session
  const [interim, setInterim] = useState('');       // live partial words
  const [error, setError] = useState(null);

  const recRef = useRef(null);      // live SpeechRecognition instance
  const finalRef = useRef('');      // seed + finalised chunks
  const spokenRef = useRef(false);  // at least one final speech chunk arrived
  const abortedRef = useRef(false); // fatal error / cancel — no auto-submit
  const finalizeRef = useRef(null);

  // Keep the latest callback available to recognition handlers that were
  // created earlier (they must see the freshest closure, not a stale one).
  useEffect(() => {
    finalizeRef.current = onFinalTranscript;
  }, [onFinalTranscript]);

  const start = useCallback((seed = '') => {
    if (!SpeechRecognitionCtor || recRef.current) return;
    setError(null);
    const seeded = seed.trim();
    finalRef.current = seeded;
    spokenRef.current = false;
    abortedRef.current = false;
    setTranscript(seeded);
    setInterim('');

    const rec = new SpeechRecognitionCtor();
    rec.lang = lang;
    rec.continuous = true;
    rec.interimResults = true;

    rec.onresult = (event) => {
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
      setTranscript(finalRef.current);
      setInterim(interimPart.trim());
    };

    rec.onerror = (event) => {
      if (event.error === 'aborted') return;  // intentional stop — onend handles it
      if (event.error === 'no-speech') return; // handled as an empty session in onend
      abortedRef.current = true;               // fatal — never auto-submit
      if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
        setError('Microphone permission denied — type instead.');
      } else if (event.error === 'network') {
        setError('Speech service unavailable — type instead.');
      } else {
        setError('Speech recognition error — type instead.');
      }
    };

    rec.onend = () => {
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
      setError('Could not start the microphone — type instead.');
    }
  }, [lang]);

  const stop = useCallback(() => {
    // Graceful stop: onend fires and the transcript is auto-submitted.
    recRef.current?.stop();
  }, []);

  const cancel = useCallback(() => {
    // Abort without submitting (keeps whatever was typed intact).
    abortedRef.current = true;
    try {
      recRef.current?.abort();
    } catch {
      // Recognition already stopped
    }
    recRef.current = null;
    setListening(false);
    setInterim('');
  }, []);

  // Abort any live session on unmount — never submit from a dead panel.
  useEffect(() => () => {
    abortedRef.current = true;
    try {
      recRef.current?.abort();
    } catch {
      // Recognition already stopped
    }
    recRef.current = null;
  }, []);

  return { supported, listening, transcript, interim, error, start, stop, cancel };
}
