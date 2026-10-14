import { useState, useCallback, useRef, useEffect } from 'react';
import {
  Search, ChevronRight, FileText, User, MessageSquare,
  Mic, Send, AlertTriangle, CheckCircle, Loader2,
  X, ChevronDown, Edit3, Zap, Shield,
  Activity, Calendar, Stethoscope, Radio, Info, Square,
} from 'lucide-react';
import { api } from './api.js';
import { useSpeechRecognition } from './useSpeechRecognition.js';

// ── Utility helpers ──────────────────────────────────────────────────────────

function initials(name) {
  return name ? name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2) : '?';
}

function reportTypeBadgeClass(type) {
  const map = { OPG: 'badge-opg', PERIAPICAL: 'badge-periapical', BITEWING: 'badge-bitewing', CBCT: 'badge-cbct' };
  return map[type] || '';
}

function severityClass(sev) {
  const map = { none: 'severity-none', mild: 'severity-mild', moderate: 'severity-moderate', severe: 'severity-severe' };
  return map[(sev || '').toLowerCase()] || 'severity-unspecified';
}

function formatDate(d) {
  if (!d) return '';
  const dt = new Date(d + 'T00:00:00');
  return dt.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
}

// ── Tiny LoadingDots ─────────────────────────────────────────────────────────
function LoadingDots() {
  return (
    <span style={{ display: 'inline-flex', gap: 4, alignItems: 'center', padding: '2px 0' }}>
      {[0, 1, 2].map(i => (
        <span key={i} className="pulse-dot" style={{
          width: 6, height: 6, borderRadius: '50%', background: 'var(--color-ink-muted)',
          animationDelay: `${i * 0.2}s`,
        }} />
      ))}
    </span>
  );
}

// ── Skeleton loader ──────────────────────────────────────────────────────────
function Skeleton({ h = 16, w = '100%', mb = 8 }) {
  return <div className="skeleton" style={{ height: h, width: w, marginBottom: mb }} />;
}

// ── Demo mode badge ──────────────────────────────────────────────────────────
function DemoBanner() {
  return (
    <div className="demo-banner">
      <strong>Demo Mode</strong> — No API key configured. LLM features use deterministic heuristics.
      Set <code>GEMINI_API_KEY</code> in <code>backend/.env</code> to enable Gemini AI.
    </div>
  );
}

// ── Report type label ────────────────────────────────────────────────────────
function ReportTypeBadge({ type }) {
  return (
    <span className={`type-eyebrow ${reportTypeBadgeClass(type)}`}
      style={{ padding: '2px 8px', borderRadius: 'var(--radius-xs)', display: 'inline-block' }}>
      {type}
    </span>
  );
}

// ── Finding card ─────────────────────────────────────────────────────────────
function FindingCard({ finding }) {
  const sev = (finding.severity || 'unspecified').toLowerCase();
  return (
    <div className={`report-finding ${sev}`} style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', gap: 8, justifyContent: 'space-between', flexWrap: 'wrap', marginBottom: 4 }}>
        <span className="type-eyebrow" style={{ color: 'var(--color-ink-muted)' }}>{finding.site || 'Unspecified site'}</span>
        <span className={`type-caption ${severityClass(finding.severity)}`}
          style={{ padding: '1px 6px', borderRadius: 'var(--radius-xs)' }}>
          {finding.severity || 'unspecified'}
        </span>
      </div>
      <p className="type-body-sm" style={{ margin: 0, color: 'var(--color-ink)' }}>{finding.observation}</p>
    </div>
  );
}

// ── Full Report Detail ───────────────────────────────────────────────────────
function ReportDetail({ report, patient, onFollowUpClick }) {
  const [showDraft, setShowDraft] = useState(false);

  if (!report) return null;

  return (
    <div className="animate-fade-up" style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 20, height: '100%' }}>
      {/* Header */}
      <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap' }}>
        <div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 4, flexWrap: 'wrap' }}>
            <ReportTypeBadge type={report.type} />
            <span className="type-caption" style={{ color: 'var(--color-ink-muted)' }}>#{report.id}</span>
            <span className={`type-caption severity-${report.status === 'final' ? 'none' : 'mild'}`}
              style={{ padding: '1px 6px', borderRadius: 'var(--radius-xs)' }}>
              {report.status}
            </span>
          </div>
          <h2 className="type-card-title" style={{ margin: 0 }}>
            {report.type} Report
            {patient && <span style={{ fontWeight: 400, color: 'var(--color-ink-muted)' }}> — {patient.name}</span>}
          </h2>
          <div style={{ display: 'flex', gap: 16, marginTop: 6, flexWrap: 'wrap' }}>
            <span className="type-body-sm" style={{ color: 'var(--color-ink-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
              <Calendar size={13} /> {formatDate(report.date)}
            </span>
            <span className="type-body-sm" style={{ color: 'var(--color-ink-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
              <Stethoscope size={13} /> {report.radiologist}
            </span>
          </div>
        </div>
        <button className="btn-fin" onClick={onFollowUpClick} style={{ gap: 6 }}>
          <MessageSquare size={15} /> Ask AI
        </button>
      </div>

      {/* Findings */}
      <div>
        <h3 className="type-eyebrow" style={{ marginBottom: 10, color: 'var(--color-ink-muted)', textTransform: 'none' }}>
          Findings ({report.findings?.length || 0})
        </h3>
        {report.findings?.map((f, i) => <FindingCard key={i} finding={f} />)}
      </div>

      {/* Impression */}
      <div className="card" style={{ borderLeft: '3px solid var(--color-ink)', borderRadius: '0 var(--radius-lg) var(--radius-lg) 0' }}>
        <div className="type-eyebrow" style={{ marginBottom: 6, color: 'var(--color-ink-muted)' }}>Impression</div>
        <p className="type-body" style={{ margin: 0 }}>{report.impression}</p>
      </div>

      {/* Draft Report Toggle */}
      <div>
        <button className="btn-secondary" onClick={() => setShowDraft(s => !s)} style={{ width: '100%', justifyContent: 'space-between' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <FileText size={15} /> Draft Report
          </span>
          <ChevronDown size={15} style={{ transform: showDraft ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
        </button>
        {showDraft && (
          <div className="animate-fade-up" style={{
            marginTop: 8, background: 'var(--color-surface-1)', border: '1px solid var(--color-hairline)',
            borderRadius: 'var(--radius-md)', padding: 16,
          }}>
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 8 }}>
              <span className="type-caption" style={{ color: 'var(--color-fin-orange)', display: 'flex', alignItems: 'center', gap: 4 }}>
                <Shield size={11} /> Clinician review required before use
              </span>
            </div>
            <pre style={{
              fontFamily: 'var(--font-mono)', fontSize: 12, lineHeight: 1.7,
              color: 'var(--color-ink)', margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
            }}>
              {report.draft_report}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Patient sidebar ──────────────────────────────────────────────────────────
function PatientSidebar({ patients, loading, selectedPatientId, onSelect, searchValue, onSearchChange }) {
  return (
    <div className="panel-left">
      {/* Search bar */}
      <div style={{ padding: '14px 16px', borderBottom: '1px solid var(--color-hairline-soft)' }}>
        <div style={{ position: 'relative' }}>
          <Search size={15} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--color-ink-tertiary)' }} />
          <input
            className="input-text"
            style={{ paddingLeft: 34, fontSize: 14 }}
            placeholder="Search patients…"
            value={searchValue}
            onChange={e => onSearchChange(e.target.value)}
            aria-label="Search patients"
          />
        </div>
      </div>

      {/* List */}
      <div className="scroll-y" style={{ flex: 1, padding: '8px 8px' }}>
        {loading ? (
          Array.from({ length: 5 }, (_, i) => (
            <div key={i} style={{ padding: '10px 16px', display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
              <Skeleton h={36} w={36} mb={0} />
              <div style={{ flex: 1 }}>
                <Skeleton h={12} w="70%" mb={6} />
                <Skeleton h={10} w="50%" mb={0} />
              </div>
            </div>
          ))
        ) : patients.map(p => (
          <div
            key={p.id}
            className={`patient-item ${selectedPatientId === p.id ? 'active' : ''}`}
            onClick={() => onSelect(p)}
            role="button"
            tabIndex={0}
            onKeyDown={e => e.key === 'Enter' && onSelect(p)}
          >
            <div className="patient-avatar">{initials(p.name)}</div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="type-body-sm" style={{ fontWeight: 500, color: 'var(--color-ink)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {p.name}
              </div>
              <div className="type-caption" style={{ color: 'var(--color-ink-muted)' }}>
                {p.gender} · DOB {formatDate(p.dob)}
              </div>
            </div>
            <ChevronRight size={14} style={{ color: 'var(--color-ink-tertiary)', flexShrink: 0 }} />
          </div>
        ))}
        {!loading && patients.length === 0 && (
          <div style={{ padding: '32px 16px', textAlign: 'center', color: 'var(--color-ink-muted)' }}>
            <User size={32} style={{ marginBottom: 8, opacity: 0.4 }} />
            <p className="type-body-sm">No patients found</p>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Conversation panel ───────────────────────────────────────────────────────
function ConversationPanel({ messages, loading, activeReportId, onSend }) {
  const [input, setInput] = useState('');
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  // Voice input: when speech ends, auto-submit the transcript as the query.
  const handleVoiceSubmit = useCallback((text) => {
    const q = text.trim();
    if (!q || loading) return;
    setInput('');
    onSend(q);
  }, [loading, onSend]);

  const speech = useSpeechRecognition({ onFinalTranscript: handleVoiceSubmit });

  // While listening, the live transcript (seeded with any typed text) is
  // shown in the input itself, so the user always sees what will be submitted.
  const voiceText = speech.listening
    ? `${speech.transcript} ${speech.interim}`.trim()
    : null;
  const displayValue = voiceText ?? input;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const send = () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput('');
    onSend(q);
  };

  const handleKey = e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  };

  const suggestions = [
    "Summarise this report",
    "What are the key findings?",
    "What treatment is recommended?",
    "Are there any urgent concerns?",
  ];

  return (
    <div className="panel-right" style={{ display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <div style={{ padding: '14px 16px', borderBottom: '1px solid var(--color-hairline-soft)', flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--color-fin-orange)' }} />
          <span className="type-eyebrow" style={{ color: 'var(--color-ink)' }}>Viscan AI</span>
        </div>
        {activeReportId && (
          <div className="type-caption" style={{ color: 'var(--color-ink-muted)', marginTop: 2, display: 'flex', alignItems: 'center', gap: 4 }}>
            <Radio size={11} /> Context: {activeReportId}
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="scroll-y" style={{ flex: 1, padding: '16px', display: 'flex', flexDirection: 'column', gap: 12 }}>
        {messages.length === 0 && (
          <div style={{ textAlign: 'center', padding: '32px 0', color: 'var(--color-ink-muted)' }}>
            <Zap size={32} style={{ marginBottom: 10, color: 'var(--color-fin-orange)', opacity: 0.7 }} />
            <p className="type-body-sm" style={{ marginBottom: 16 }}>
              Open a report to start asking questions, or type a patient query below.
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {suggestions.map(s => (
                <button key={s} className="btn-secondary" style={{ width: '100%', fontSize: 13, padding: '8px 12px', justifyContent: 'flex-start' }}
                  onClick={() => { setInput(s); inputRef.current?.focus(); }}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className="animate-fade-up" style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {msg.role === 'user' ? (
              <div className="chat-user">{msg.content}</div>
            ) : (
              <div>
                <div className="chat-ai">
                  <p style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{msg.content}</p>
                </div>
                {msg.evidence?.length > 0 && (
                  <div className="evidence-block animate-fade-up" style={{ marginTop: 6 }}>
                    <div className="type-eyebrow" style={{ marginBottom: 4, color: '#15803d', display: 'flex', alignItems: 'center', gap: 4 }}>
                      <CheckCircle size={12} /> Source evidence
                    </div>
                    {msg.evidence.map((e, j) => (
                      <div key={j} className="type-caption" style={{ marginTop: 2 }}>• {e}</div>
                    ))}
                  </div>
                )}
                {msg.source_report_id && (
                  <div className="type-caption" style={{ marginTop: 4, color: 'var(--color-ink-tertiary)', display: 'flex', alignItems: 'center', gap: 4 }}>
                    <Info size={10} /> Report {msg.source_report_id}
                  </div>
                )}
                {msg.demo_mode && (
                  <div className="type-caption" style={{ marginTop: 2, color: 'var(--color-ink-tertiary)' }}>
                    Demo mode response
                  </div>
                )}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="chat-ai" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <LoadingDots />
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div style={{ padding: '12px 16px', borderTop: '1px solid var(--color-hairline-soft)', flexShrink: 0 }}>
        {!activeReportId && (
          <div className="type-caption" style={{ color: 'var(--color-ink-tertiary)', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 4 }}>
            <AlertTriangle size={11} /> Open a report first for context-aware answers
          </div>
        )}
        <div style={{ display: 'flex', gap: 8 }}>
          <input
            ref={inputRef}
            id="chat-input"
            className="input-text"
            style={{ flex: 1, fontSize: 14, minHeight: 44 }}
            placeholder={activeReportId ? "Ask about this report…" : "Open Priya's latest OPG…"}
            value={displayValue}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            aria-label="Chat input"
            disabled={loading || speech.listening}
          />
          <button
            type="button"
            className="btn-secondary"
            onClick={speech.listening ? speech.stop : () => speech.start(input)}
            disabled={!speech.supported || loading}
            aria-label={speech.listening ? 'Stop voice input and submit' : 'Start voice input'}
            aria-pressed={speech.listening}
            title={
              !speech.supported
                ? 'Speech recognition unavailable — type instead'
                : speech.listening
                  ? 'Stop and submit'
                  : 'Speak your request'
            }
            style={{
              padding: '10px 14px', minHeight: 44,
              ...(speech.listening ? {
                borderColor: 'var(--color-semantic-error)',
                color: 'var(--color-semantic-error)',
              } : {}),
            }}
          >
            {speech.listening
              ? <Square size={14} fill="currentColor" />
              : <Mic size={15} />}
          </button>
          <button className="btn-fin" onClick={send} disabled={!input.trim() || loading || speech.listening} style={{ padding: '10px 14px' }}>
            {loading ? <Loader2 size={15} className="animate-spin" /> : <Send size={15} />}
          </button>
        </div>

        {/* Voice status: listening / errors / unsupported fallback */}
        <div aria-live="polite" style={{ marginTop: 6, minHeight: 16 }}>
          {speech.listening && (
            <span className="type-caption" style={{ color: 'var(--color-semantic-error)', display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="pulse-dot" style={{
                width: 6, height: 6, borderRadius: '50%', background: 'var(--color-semantic-error)',
              }} />
              Listening… speak now, then tap stop to submit
            </span>
          )}
          {!speech.listening && speech.error && (
            <span className="type-caption" style={{ color: 'var(--color-semantic-error)', display: 'flex', alignItems: 'center', gap: 4 }}>
              <AlertTriangle size={11} /> {speech.error}
            </span>
          )}
          {!speech.supported && (
            <span className="type-caption" style={{ color: 'var(--color-ink-tertiary)', display: 'flex', alignItems: 'center', gap: 4 }}>
              <Mic size={11} /> Voice input unavailable in this browser — type your request instead.
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Dictation panel ──────────────────────────────────────────────────────────
function DictationPanel({ result, loading, onSubmit }) {
  const [transcript, setTranscript] = useState('');
  // Local edits override the latest draft; null = show the latest draft as-is.
  // Reset on submit so a fresh result replaces any stale edits (no effect needed).
  const [editedDraft, setEditedDraft] = useState(null);

  // Voice dictation: when speech ends, the transcript is auto-processed —
  // no extra button click. The transcript itself is never overwritten by the
  // structured result, so the original wording stays visible and editable.
  const handleVoiceSubmit = useCallback((text) => {
    const t = text.trim();
    if (!t || loading) return;
    setTranscript(t);
    setEditedDraft(null);
    onSubmit(t);
  }, [loading, onSubmit]);

  const speech = useSpeechRecognition({ onFinalTranscript: handleVoiceSubmit });

  // Live transcript (seeded with any typed text) shows in the textarea
  // while listening; otherwise the typed/stored transcript shows as-is.
  const voiceText = speech.listening
    ? `${speech.transcript} ${speech.interim}`.trim()
    : null;
  const displayTranscript = voiceText ?? transcript;

  const handleSubmit = () => {
    if (!transcript.trim() || loading) return;
    setEditedDraft(null);
    onSubmit(transcript);
  };

  return (
    <div style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 20, height: '100%', overflowY: 'auto' }}>
      <div>
        <h2 className="type-card-title" style={{ margin: 0, marginBottom: 4 }}>Clinical Dictation</h2>
        <p className="type-body-sm" style={{ color: 'var(--color-ink-muted)', margin: 0, display: 'flex', alignItems: 'center', gap: 4 }}>
          <Mic size={13} /> Type, paste, or dictate clinical findings below
        </p>
      </div>

      {/* Review warning */}
      <div className="review-banner">
        <AlertTriangle size={16} style={{ flexShrink: 0, marginTop: 1 }} />
        <div>
          <strong>Clinician review required.</strong> AI-structured output must be verified before use. Never use unsupervised output in clinical documentation.
        </div>
      </div>

      {/* Transcript input */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6, gap: 8, flexWrap: 'wrap' }}>
          <label className="type-eyebrow" style={{ color: 'var(--color-ink-muted)' }}>
            Dictation transcript
          </label>
          <button
            type="button"
            className="btn-secondary"
            onClick={speech.listening ? speech.stop : () => speech.start(transcript)}
            disabled={!speech.supported || loading}
            aria-label={speech.listening ? 'Stop dictation and process transcript' : 'Start voice dictation'}
            aria-pressed={speech.listening}
            title={
              !speech.supported
                ? 'Speech recognition unavailable — type instead'
                : speech.listening
                  ? 'Stop and process'
                  : 'Dictate observations'
            }
            style={{
              fontSize: 13, padding: '6px 12px', minHeight: 40,
              ...(speech.listening ? {
                borderColor: 'var(--color-semantic-error)',
                color: 'var(--color-semantic-error)',
              } : {}),
            }}
          >
            {speech.listening
              ? <><Square size={13} fill="currentColor" /> Stop &amp; process</>
              : <><Mic size={13} /> Dictate</>}
          </button>
        </div>
        <textarea
          className="textarea-text"
          rows={6}
          placeholder="e.g. Patient shows periapical radiolucency at 26, approximately 4mm. Mild bone loss at 46…"
          value={displayTranscript}
          onChange={e => setTranscript(e.target.value)}
          disabled={loading || speech.listening}
          aria-label="Clinical dictation transcript"
          style={{ minHeight: 130 }}
        />

        {/* Voice status: listening / errors / unsupported fallback */}
        <div aria-live="polite" style={{ marginTop: 6, minHeight: 16 }}>
          {speech.listening && (
            <span className="type-caption" style={{ color: 'var(--color-semantic-error)', display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="pulse-dot" style={{
                width: 6, height: 6, borderRadius: '50%', background: 'var(--color-semantic-error)',
              }} />
              Dictating… speak naturally, then tap stop to structure automatically
            </span>
          )}
          {!speech.listening && speech.error && (
            <span className="type-caption" style={{ color: 'var(--color-semantic-error)', display: 'flex', alignItems: 'center', gap: 4 }}>
              <AlertTriangle size={11} /> {speech.error}
            </span>
          )}
          {!speech.supported && (
            <span className="type-caption" style={{ color: 'var(--color-ink-tertiary)', display: 'flex', alignItems: 'center', gap: 4 }}>
              <Mic size={11} /> Voice dictation unavailable in this browser — type your observations instead.
            </span>
          )}
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 8, flexWrap: 'wrap', gap: 8 }}>
          <span className="type-caption" style={{ color: 'var(--color-ink-tertiary)' }}>
            {displayTranscript.length}/10000 characters
          </span>
          <button className="btn-primary" onClick={handleSubmit}
            disabled={!transcript.trim() || loading || speech.listening}>
            {loading
              ? <><Loader2 size={14} className="animate-spin" /> Structuring…</>
              : <><Activity size={14} /> Structure findings</>}
          </button>
        </div>
      </div>

      {/* Result */}
      {loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <Skeleton h={14} w="60%" />
          <Skeleton h={80} />
          <Skeleton h={14} w="40%" />
          <Skeleton h={120} />
        </div>
      )}

      {result && !loading && (
        <div className="animate-fade-up" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Structured findings */}
          <div>
            <h3 className="type-eyebrow" style={{ marginBottom: 8, color: 'var(--color-ink-muted)' }}>
              Structured Findings
            </h3>
            {result.findings?.map((f, i) => <FindingCard key={i} finding={f} />)}
          </div>

          {/* Anatomical site */}
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span className="type-eyebrow" style={{ color: 'var(--color-ink-muted)' }}>Primary region:</span>
            <span className="type-body-sm">{result.anatomical_site}</span>
          </div>

          {/* Uncertainties */}
          {result.uncertainties?.length > 0 && (
            <div style={{ background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 'var(--radius-md)', padding: '10px 14px' }}>
              <div className="type-eyebrow" style={{ marginBottom: 6, color: '#92400e', display: 'flex', alignItems: 'center', gap: 4 }}>
                <AlertTriangle size={13} /> Uncertainties
              </div>
              {result.uncertainties.map((u, i) => (
                <div key={i} className="type-body-sm" style={{ color: '#92400e', marginBottom: 2 }}>• {u}</div>
              ))}
            </div>
          )}

          {/* Editable draft */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
              <h3 className="type-eyebrow" style={{ color: 'var(--color-ink-muted)', margin: 0, display: 'flex', alignItems: 'center', gap: 4 }}>
                <Edit3 size={13} /> Editable Draft Report
              </h3>
              <span className="type-caption" style={{ color: 'var(--color-fin-orange)', display: 'flex', alignItems: 'center', gap: 4 }}>
                <Shield size={11} /> Review before use
              </span>
            </div>
            <textarea
              className="textarea-text type-mono"
              rows={12}
              value={editedDraft ?? result?.draft_report ?? ''}
              onChange={e => setEditedDraft(e.target.value)}
              aria-label="Editable draft report"
              style={{ minHeight: 220, fontSize: 12, lineHeight: 1.7 }}
            />
            <p className="type-caption" style={{ color: 'var(--color-ink-tertiary)', marginTop: 4 }}>
              This draft was AI-generated from your dictation and is editable. It must be reviewed and approved by a qualified radiologist before clinical use.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Ambiguity resolver ───────────────────────────────────────────────────────
function AmbiguityCard({ patients, query, onSelect, onDismiss }) {
  return (
    <div className="ambiguity-card animate-fade-up" style={{ margin: 24 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
            <AlertTriangle size={16} style={{ color: 'var(--color-fin-orange)' }} />
            <span className="type-eyebrow" style={{ color: 'var(--color-ink)' }}>Multiple patients match</span>
          </div>
          <p className="type-body-sm" style={{ color: 'var(--color-ink-muted)', margin: 0 }}>
            Found {patients.length} patients matching "{query}". Please select the correct patient.
          </p>
        </div>
        <button className="btn-icon" onClick={onDismiss}><X size={15} /></button>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {patients.map(p => (
          <button key={p.id} className="btn-secondary" onClick={() => onSelect(p)}
            style={{ width: '100%', justifyContent: 'flex-start', gap: 10 }}>
            <div className="patient-avatar" style={{ width: 32, height: 32, fontSize: 12 }}>{initials(p.name)}</div>
            <div style={{ textAlign: 'left' }}>
              <div className="type-body-sm" style={{ fontWeight: 500 }}>{p.name}</div>
              <div className="type-caption" style={{ color: 'var(--color-ink-muted)' }}>{p.gender} · DOB {formatDate(p.dob)}</div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

// ── Report list for a patient ────────────────────────────────────────────────
function ReportList({ reports, loadingReports, onSelect, activeReportId }) {
  if (loadingReports) {
    return (
      <div style={{ padding: 24 }}>
        {Array.from({ length: 3 }, (_, i) => <div key={i} style={{ marginBottom: 12 }}><Skeleton h={72} /></div>)}
      </div>
    );
  }

  if (!reports?.length) {
    return (
      <div style={{ padding: 48, textAlign: 'center', color: 'var(--color-ink-muted)' }}>
        <FileText size={32} style={{ marginBottom: 8, opacity: 0.4 }} />
        <p className="type-body-sm">No reports found for this patient.</p>
      </div>
    );
  }

  return (
    <div style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 10 }}>
      <h3 className="type-eyebrow" style={{ color: 'var(--color-ink-muted)', marginBottom: 4 }}>Reports ({reports.length})</h3>
      {reports.map(r => (
        <div key={r.id}
          className={`card animate-fade-up`}
          style={{
            cursor: 'pointer', padding: '14px 16px',
            outline: activeReportId === r.id ? '2px solid var(--color-ink)' : 'none',
            outlineOffset: 2,
          }}
          onClick={() => onSelect(r)}
          role="button"
          tabIndex={0}
          onKeyDown={e => e.key === 'Enter' && onSelect(r)}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 4, flexWrap: 'wrap' }}>
                <ReportTypeBadge type={r.type} />
                <span className="type-caption" style={{ color: 'var(--color-ink-muted)', display: 'flex', alignItems: 'center', gap: 3 }}>
                  <Calendar size={11} /> {formatDate(r.date)}
                </span>
              </div>
              <p className="type-caption" style={{
                color: 'var(--color-ink-muted)', margin: 0,
                display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden',
              }}>
                {r.impression}
              </p>
            </div>
            <ChevronRight size={15} style={{ color: 'var(--color-ink-tertiary)', flexShrink: 0, marginTop: 2 }} />
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  // State
  const [patients, setPatients] = useState([]);
  const [patientsLoading, setPatientsLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [reports, setReports] = useState([]);
  const [reportsLoading, setReportsLoading] = useState(false);
  const [activeReport, setActiveReport] = useState(null);
  const [activeReportFull, setActiveReportFull] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);

  const [centerTab, setCenterTab] = useState('reports'); // 'reports' | 'report-detail' | 'dictation'
  const [ambiguity, setAmbiguity] = useState(null); // { patients, query }

  const [chatMessages, setChatMessages] = useState([]);
  const [chatLoading, setChatLoading] = useState(false);

  const [dictResult, setDictResult] = useState(null);
  const [dictLoading, setDictLoading] = useState(false);

  const [demoMode, setDemoMode] = useState(false);
  const [backendError, setBackendError] = useState(null);

  // Load patients on mount
  useEffect(() => {
    api.listPatients()
      .then(data => { setPatients(data); setPatientsLoading(false); })
      .catch(() => {
        setPatientsLoading(false);
        setBackendError('Cannot reach backend. Make sure the API is running on port 8000.');
      });
    api.health().then(h => setDemoMode(h.demo_mode)).catch(() => {});
  }, []);

  const filteredPatients = patients.filter(p =>
    p.name.toLowerCase().includes(search.toLowerCase())
  );

  const handleSelectPatient = useCallback(async (p) => {
    setSelectedPatient(p);
    setActiveReport(null);
    setActiveReportFull(null);
    setAmbiguity(null);
    setCenterTab('reports');
    setReportsLoading(true);
    try {
      const reps = await api.getReports(p.id);
      setReports(reps);
    } catch {
      setReports([]);
    } finally {
      setReportsLoading(false);
    }
  }, []);

  const handleSelectReport = useCallback(async (r) => {
    setActiveReport(r);
    setCenterTab('report-detail');
    setReportLoading(true);
    try {
      const full = await api.getReport(r.id);
      setActiveReportFull(full);
    } catch {
      setActiveReportFull(null);
    } finally {
      setReportLoading(false);
    }
  }, []);

  const handleConversationSend = useCallback(async (question) => {
    // Add user message
    setChatMessages(m => [...m, { role: 'user', content: question }]);
    setChatLoading(true);

    try {
      // Always interpret the query WITH session context so intent routing stays
      // correct even while another report is open (e.g. "Open Rohan's bitewing"
      // must open Rohan's report, not be answered as a follow-up about Priya).
      const res = await api.query({
        text: question,
        session_context: activeReportFull?.id ? { active_report_id: activeReportFull.id } : null,
      });

      if (res.ambiguous) {
        // Ambiguous identity → disambiguation card, never a silent pick
        setAmbiguity({ patients: res.matched_patients, query: question });
        setChatMessages(m => [...m, {
          role: 'ai',
          content: res.message,
          evidence: [],
          demo_mode: res.demo_mode,
        }]);
      } else if (res.intent === 'follow_up' && res.report) {
        // Contextual follow-up: answer from the active report, with evidence
        const fu = await api.followUp({ question, report_id: res.report.id });
        setChatMessages(m => [...m, {
          role: 'ai',
          content: fu.answer,
          evidence: fu.evidence,
          source_report_id: fu.source_report_id,
          demo_mode: fu.demo_mode,
        }]);
      } else if (res.report) {
        // Retrieval: auto-open the returned report
        setActiveReportFull(res.report);
        setActiveReport(res.report);
        setCenterTab('report-detail');
        if (res.matched_patients?.[0]) setSelectedPatient(res.matched_patients[0]);
        setChatMessages(m => [...m, {
          role: 'ai',
          content: res.message,
          evidence: [],
          source_report_id: res.report.id,
          demo_mode: res.demo_mode,
        }]);
      } else {
        setChatMessages(m => [...m, {
          role: 'ai',
          content: res.message,
          evidence: [],
          demo_mode: res.demo_mode,
        }]);
      }
    } catch (err) {
      setChatMessages(m => [...m, {
        role: 'ai',
        content: `Error: ${err.message}. Please check the backend is running.`,
        evidence: [],
      }]);
    } finally {
      setChatLoading(false);
    }
  }, [activeReportFull]);

  const handleDictationSubmit = useCallback(async (transcript) => {
    setDictLoading(true);
    setDictResult(null);
    try {
      const res = await api.dictation({ transcript });
      setDictResult(res);
    } catch (err) {
      setDictResult({
        findings: [],
        anatomical_site: 'Error',
        draft_report: `Failed to process dictation: ${err.message}`,
        uncertainties: ['Backend error'],
        requires_review: true,
        demo_mode: true,
      });
    } finally {
      setDictLoading(false);
    }
  }, []);

  const handleAmbiguitySelect = useCallback(async (patient) => {
    setAmbiguity(null);
    await handleSelectPatient(patient);
  }, [handleSelectPatient]);

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <>
      {/* Top nav */}
      <nav className="top-nav">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 'var(--radius-sm)',
            background: 'var(--color-fin-orange)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Radio size={15} color="white" />
          </div>
          <span className="type-eyebrow" style={{ color: 'var(--color-ink)' }}>Viscan AI</span>
          <span className="type-caption" style={{ color: 'var(--color-ink-tertiary)', marginLeft: 4 }}>Dental Radiology Assistant</span>
        </div>

        <div style={{ flex: 1 }} />

        {/* Tab group for center panel */}
        <div className="tab-group">
          <button className={`tab-item ${centerTab !== 'dictation' ? 'active' : ''}`}
            onClick={() => setCenterTab(activeReport ? 'report-detail' : 'reports')}
            style={{ gap: 4 }}>
            <FileText size={13} /> Reports
          </button>
          <button className={`tab-item ${centerTab === 'dictation' ? 'active' : ''}`}
            onClick={() => setCenterTab('dictation')}
            style={{ gap: 4 }}>
            <Mic size={13} /> Dictation
          </button>
        </div>

        <div style={{ flex: 1 }} />

        <div className="type-caption" style={{ color: 'var(--color-ink-tertiary)', display: 'flex', alignItems: 'center', gap: 4 }}>
          <Shield size={11} />
          <span>POC — Not for clinical use</span>
        </div>
      </nav>

      {/* Demo banner */}
      {demoMode && <DemoBanner />}

      {/* Backend error */}
      {backendError && (
        <div style={{ background: '#fef2f2', borderBottom: '1px solid #fecaca', padding: '8px 24px', color: '#991b1b', fontSize: 13, display: 'flex', alignItems: 'center', gap: 8 }}>
          <AlertTriangle size={14} /> {backendError}
        </div>
      )}

      {/* Main 3-panel layout */}
      <div className="app-layout">
        {/* LEFT: Patient list */}
        <PatientSidebar
          patients={filteredPatients}
          loading={patientsLoading}
          selectedPatientId={selectedPatient?.id}
          onSelect={handleSelectPatient}
          searchValue={search}
          onSearchChange={setSearch}
        />

        {/* CENTER: Reports / detail / dictation */}
        <div className="panel-center">
          {/* Patient header */}
          {selectedPatient && centerTab !== 'dictation' && (
            <div style={{
              padding: '12px 24px', borderBottom: '1px solid var(--color-hairline-soft)',
              display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0,
              background: 'var(--color-surface-1)',
            }}>
              <div className="patient-avatar">{initials(selectedPatient.name)}</div>
              <div>
                <div className="type-body" style={{ fontWeight: 500 }}>{selectedPatient.name}</div>
                <div className="type-caption" style={{ color: 'var(--color-ink-muted)' }}>
                  {selectedPatient.gender} · DOB {formatDate(selectedPatient.dob)}
                  {selectedPatient.phone && <> · {selectedPatient.phone}</>}
                </div>
              </div>
              {activeReport && (
                <button className="btn-secondary" onClick={() => { setActiveReport(null); setActiveReportFull(null); setCenterTab('reports'); }}
                  style={{ marginLeft: 'auto', fontSize: 13, padding: '6px 12px' }}>
                  ← All reports
                </button>
              )}
            </div>
          )}

          {/* Ambiguity resolver */}
          {ambiguity && (
            <AmbiguityCard
              patients={ambiguity.patients}
              query={ambiguity.query}
              onSelect={handleAmbiguitySelect}
              onDismiss={() => setAmbiguity(null)}
            />
          )}

          {/* Content */}
          <div className="scroll-y" style={{ flex: 1 }}>
            {centerTab === 'dictation' && (
              <DictationPanel result={dictResult} loading={dictLoading} onSubmit={handleDictationSubmit} />
            )}

            {centerTab === 'reports' && !selectedPatient && (
              <div style={{ padding: 48, textAlign: 'center', color: 'var(--color-ink-muted)' }}>
                <div style={{
                  width: 64, height: 64, borderRadius: 'var(--radius-xl)',
                  background: 'var(--color-surface-1)', border: '1px solid var(--color-hairline)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  margin: '0 auto 16px',
                }}>
                  <User size={28} style={{ color: 'var(--color-ink-tertiary)' }} />
                </div>
                <h2 className="type-headline" style={{ marginBottom: 8 }}>Select a patient</h2>
                <p className="type-body" style={{ color: 'var(--color-ink-muted)', marginBottom: 24 }}>
                  Choose a patient from the sidebar, or type a query in the AI panel.
                </p>
                <p className="type-body-sm" style={{ color: 'var(--color-ink-tertiary)' }}>
                  Try: "Open Priya's latest OPG report"
                </p>
              </div>
            )}

            {centerTab === 'reports' && selectedPatient && (
              <ReportList
                reports={reports}
                loadingReports={reportsLoading}
                onSelect={handleSelectReport}
                activeReportId={activeReport?.id}
              />
            )}

            {centerTab === 'report-detail' && (
              reportLoading
                ? <div style={{ padding: 24 }}>
                    <Skeleton h={20} w="60%" mb={12} />
                    <Skeleton h={14} w="40%" mb={24} />
                    <Skeleton h={100} mb={12} />
                    <Skeleton h={80} mb={12} />
                    <Skeleton h={200} />
                  </div>
                : <ReportDetail
                    report={activeReportFull || activeReport}
                    patient={selectedPatient}
                    onFollowUpClick={() => document.getElementById('chat-input')?.focus()}
                  />
            )}
          </div>
        </div>

        {/* RIGHT: Conversation */}
        <ConversationPanel
          messages={chatMessages}
          loading={chatLoading}
          activeReportId={activeReportFull?.id}
          onSend={handleConversationSend}
        />
      </div>
    </>
  );
}
