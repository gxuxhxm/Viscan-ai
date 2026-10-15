/**
 * Regression tests for the unusable-dictation gate.
 *
 * A transcript with no clinical content ("what the dark report for 2:36@") must
 * never come back as an editable draft with a Save button: the panel shows an
 * explicit "Nothing to structure" notice instead, so nothing can be filed.
 * A genuine clinical dictation still gets its editable draft and save action.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from '../App.jsx';
import { api } from '../api.js';

vi.mock('../api.js', () => ({
  api: {
    health: vi.fn(),
    listPatients: vi.fn(),
    getReports: vi.fn(),
    getReport: vi.fn(),
    query: vi.fn(),
    dictation: vi.fn(),
    followUp: vi.fn(),
    saveReport: vi.fn(),
  },
}));

const PATIENTS = [
  { id: 'P001', name: 'Priya Sharma', gender: 'Female', dob: '1992-03-14', phone: '9810000001' },
];

// Shapes mirror the backend's /dictation contract (see backend/main.py).
const UNUSABLE_RESULT = {
  findings: [],
  anatomical_site: 'Not determinable: transcript not clinically usable',
  draft_report: '',
  uncertainties: [
    'No clinical content detected in the transcript; nothing to structure',
    'Dictate a clinical observation (site or tooth, finding, severity) and submit again. Nothing was drafted from this transcript.',
  ],
  requires_review: true,
  demo_mode: false,
  fallback_reason: 'No clinical content detected in the transcript; nothing to structure',
  metadata: {},
  transcript: 'what the dark report for 2:36@',
  usable: false,
};

const USABLE_RESULT = {
  findings: [
    {
      site: 'lower left molar',
      observation: 'Mild bone loss noted at lower left molar.',
      severity: 'mild',
      context: 'current',
    },
  ],
  anatomical_site: 'lower left molar',
  draft_report: 'CLINICAL DICTATION DRAFT\n\nFINDINGS:\n- Mild bone loss noted at lower left molar.',
  uncertainties: ['AI-structured output: clinician verification required'],
  requires_review: true,
  demo_mode: false,
  fallback_reason: null,
  metadata: {},
  transcript: 'Mild bone loss noted at lower left molar.',
  usable: true,
};

beforeEach(() => {
  vi.clearAllMocks();
  Element.prototype.scrollIntoView = vi.fn();
  api.listPatients.mockResolvedValue(PATIENTS);
  api.health.mockResolvedValue({ demo_mode: false });
  api.getReports.mockResolvedValue([]);
  api.getReport.mockResolvedValue(null);
  api.query.mockResolvedValue({
    intent: 'retrieve_report',
    patient_name: null,
    report_type: null,
    ambiguous: false,
    matched_patients: [],
    report: null,
    message: 'No report opened.',
    demo_mode: false,
    fallback_reason: null,
  });
});

async function renderDictationPanel() {
  render(<App />);
  await waitFor(() => expect(api.listPatients).toHaveBeenCalled());
  fireEvent.click(screen.getByRole('button', { name: /Dictation/ }));
  expect(await screen.findByLabelText('Clinical dictation transcript')).toBeInTheDocument();
}

async function submitDictation(text) {
  fireEvent.change(screen.getByLabelText('Clinical dictation transcript'), {
    target: { value: text },
  });
  fireEvent.click(screen.getByRole('button', { name: /Create AI Report/ }));
  await waitFor(() => expect(api.dictation).toHaveBeenCalledWith({ transcript: text }));
}

describe('dictation: unusable transcript', () => {
  it('shows no editable draft and no save action for a transcript with no clinical content', async () => {
    api.dictation.mockResolvedValue(UNUSABLE_RESULT);

    await renderDictationPanel();
    await submitDictation('what the dark report for 2:36@');

    expect(await screen.findByText('Nothing to structure')).toBeInTheDocument();
    // The reason is surfaced …
    expect(screen.getByText(/No clinical content detected in the transcript/)).toBeInTheDocument();
    // … but nothing can be edited or filed.
    expect(screen.queryByLabelText('Editable draft report')).toBeNull();
    expect(screen.queryByRole('button', { name: /Save Report/ })).toBeNull();
    expect(screen.queryByText('Structured Findings')).toBeNull();
    // The provenance chip never claims AI output.
    expect(screen.getByText(/Not structured/)).toBeInTheDocument();
  });

  it('keeps the transcript editable so the clinician can dictate again', async () => {
    api.dictation.mockResolvedValue(UNUSABLE_RESULT);

    await renderDictationPanel();
    await submitDictation('what the dark report for 2:36@');

    expect(await screen.findByText('Nothing to structure')).toBeInTheDocument();
    const input = screen.getByLabelText('Clinical dictation transcript');
    expect(input).toHaveValue('what the dark report for 2:36@');
    expect(screen.getByRole('button', { name: /Create AI Report/ })).toBeEnabled();
  });
});

describe('dictation: usable transcript', () => {
  it('still offers the editable draft and the save action', async () => {
    api.dictation.mockResolvedValue(USABLE_RESULT);

    await renderDictationPanel();
    await submitDictation('Mild bone loss noted at lower left molar.');

    expect(await screen.findByLabelText('Editable draft report')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Save Report/ })).toBeEnabled();
    expect(screen.queryByText('Nothing to structure')).toBeNull();
    // The dictated observation reaches the structured findings panel.
    expect(screen.getAllByText('Mild bone loss noted at lower left molar.').length).toBeGreaterThan(0);
  });
});
