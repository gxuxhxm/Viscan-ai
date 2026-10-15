/**
 * Regression tests for the patient ambiguity selection panel.
 *
 * A query that matches several patients opens the "Multiple patients match"
 * panel in the main area. Selecting a patient there, or clarifying the name
 * through the chatbot, must resolve the pending ambiguity, but only when the
 * patient's report actually opens: an ambiguous or failed clarification keeps
 * the panel on screen so the user can retry.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
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

// ── Fixtures ─────────────────────────────────────────────────────────────────
const PRIYA = { id: 'P001', name: 'Priya Sharma', gender: 'Female', dob: '1992-03-14', phone: '9810000001' };
const PRIYANKA = { id: 'P002', name: 'Priyanka Sharma', gender: 'Female', dob: '1994-07-02', phone: '9810000002' };
const PATIENTS = [PRIYA, PRIYANKA];

const PRIYA_REPORT = {
  id: 'R001',
  patient_id: 'P001',
  type: 'OPG',
  date: '2024-05-11',
  radiologist: 'Unassigned',
  status: 'final',
  tooth_number: null,
  encounter_type: null,
  billing_codes: null,
  findings: [],
  impression: 'Normal adult dentition with no active disease.',
  draft_report: '',
};

// Shapes mirror the backend's /query contract (see backend/main.py).
const ambiguousResponse = (matched = PATIENTS, name = 'Priya') => ({
  intent: 'retrieve_report',
  patient_name: name,
  report_type: null,
  ambiguous: true,
  matched_patients: matched,
  report: null,
  message: `Multiple patients match '${name}'. Please select the correct patient from the list.`,
  demo_mode: false,
  fallback_reason: null,
});

const resolvedWithReportResponse = (patient = PRIYA, report = PRIYA_REPORT) => ({
  intent: 'retrieve_report',
  patient_name: patient.name,
  report_type: null,
  ambiguous: false,
  matched_patients: [patient],
  report,
  message: `Showing the latest ${report.type} report for ${patient.name} dated ${report.date}.`,
  demo_mode: false,
  fallback_reason: null,
});

const resolvedWithoutReportResponse = (patient = PRIYA) => ({
  intent: 'retrieve_report',
  patient_name: patient.name,
  report_type: null,
  ambiguous: false,
  matched_patients: [patient],
  report: null,
  message: `No reports found for ${patient.name}.`,
  demo_mode: false,
  fallback_reason: null,
});

// ── Helpers ──────────────────────────────────────────────────────────────────
const panel = () => document.querySelector('.ambiguity-card');

async function renderApp() {
  render(<App />);
  // The sidebar patient list is loaded on mount; wait for the app to settle.
  await waitFor(() => expect(api.listPatients).toHaveBeenCalled());
  await waitFor(() => expect(screen.getAllByText('Priya Sharma').length).toBeGreaterThan(0));
}

async function sendChat(text) {
  const input = screen.getByLabelText('Chat input');
  fireEvent.change(input, { target: { value: text } });
  fireEvent.keyDown(input, { key: 'Enter' });
}

beforeEach(() => {
  vi.clearAllMocks();
  Element.prototype.scrollIntoView = vi.fn();
  api.listPatients.mockResolvedValue(PATIENTS);
  api.health.mockResolvedValue({ demo_mode: false });
  api.getReports.mockResolvedValue([PRIYA_REPORT]);
  api.getReport.mockResolvedValue(PRIYA_REPORT);
  api.query.mockResolvedValue(ambiguousResponse());
});

// ── Tests ────────────────────────────────────────────────────────────────────
describe('ambiguity selection panel: chatbot clarification', () => {
  it('closes the panel and opens the report when the clarification resolves', async () => {
    api.query
      .mockResolvedValueOnce(ambiguousResponse())
      .mockResolvedValueOnce(resolvedWithReportResponse());

    await renderApp();

    await sendChat("Open Priya's report");
    await waitFor(() => expect(panel()).not.toBeNull());
    expect(screen.getByText('Multiple patients match')).toBeInTheDocument();

    await sendChat('Priya Sharma');

    // The correct report opens in the main panel …
    expect(await screen.findByRole('heading', { name: /OPG Report/ })).toBeInTheDocument();
    // … and the pending ambiguity panel closes with it.
    await waitFor(() => expect(panel()).toBeNull());
    expect(screen.queryByText('Multiple patients match')).toBeNull();

    // Chatbot answer and report context stay in sync with the main panel.
    expect(screen.getByText(/Showing the latest OPG report for Priya Sharma/)).toBeInTheDocument();
    expect(screen.getByText(/Context: R001/)).toBeInTheDocument();

    // The selected patient's own report list is loaded, so navigating back
    // shows this patient rather than a stale list.
    await waitFor(() => expect(api.getReports).toHaveBeenCalledWith('P001'));
    fireEvent.click(screen.getByRole('button', { name: /All reports/ }));
    expect(await screen.findByText(/Normal adult dentition/)).toBeInTheDocument();
  });
});

describe('ambiguity selection panel: unsuccessful clarifications', () => {
  it('keeps the panel open when the clarification is still ambiguous', async () => {
    api.query
      .mockResolvedValueOnce(ambiguousResponse())
      .mockResolvedValueOnce(ambiguousResponse(PATIENTS, 'Priya Sharma'));

    await renderApp();

    await sendChat('Open Priya report');
    await waitFor(() => expect(panel()).not.toBeNull());

    await sendChat('Priya Sharma');

    // Panel stays, refreshed with the latest candidates, so nothing opened.
    await waitFor(() =>
      expect(screen.getByText(/Found 2 patients matching "Priya Sharma"/)).toBeInTheDocument()
    );
    expect(panel()).not.toBeNull();
    expect(screen.queryByRole('heading', { name: /OPG Report/ })).toBeNull();
    expect(api.getReports).not.toHaveBeenCalled();
  });

  it('keeps the panel open when no report could be opened for the resolved patient', async () => {
    api.query
      .mockResolvedValueOnce(ambiguousResponse())
      .mockResolvedValueOnce(resolvedWithoutReportResponse());

    await renderApp();

    await sendChat('Open Priya report');
    await waitFor(() => expect(panel()).not.toBeNull());

    await sendChat('Priya Sharma');

    expect(await screen.findByText(/No reports found for Priya Sharma/)).toBeInTheDocument();
    expect(panel()).not.toBeNull();
    expect(screen.queryByRole('heading', { name: /OPG Report/ })).toBeNull();
  });

  it('keeps the panel open when the clarification request fails', async () => {
    api.query
      .mockResolvedValueOnce(ambiguousResponse())
      .mockRejectedValueOnce(new Error('HTTP 500'));

    await renderApp();

    await sendChat('Open Priya report');
    await waitFor(() => expect(panel()).not.toBeNull());

    await sendChat('Priya Sharma');

    expect(await screen.findByText(/Error: HTTP 500/)).toBeInTheDocument();
    expect(panel()).not.toBeNull();
    expect(screen.queryByRole('heading', { name: /OPG Report/ })).toBeNull();
    expect(api.getReports).not.toHaveBeenCalled();
  });
});

describe('ambiguity selection panel: manual selection', () => {
  it('closes the panel and loads that patient’s reports when picked directly', async () => {
    api.query.mockResolvedValueOnce(ambiguousResponse());

    await renderApp();

    await sendChat('Open Priya report');
    await waitFor(() => expect(panel()).not.toBeNull());

    fireEvent.click(within(panel()).getByRole('button', { name: /Priya Sharma/ }));

    await waitFor(() => expect(panel()).toBeNull());
    expect(await screen.findByText(/Normal adult dentition/)).toBeInTheDocument();
    expect(api.getReports).toHaveBeenCalledWith('P001');
    expect(screen.getAllByText('Priya Sharma').length).toBeGreaterThan(0);
    // No report detail is opened from the panel; it only picks the patient.
    expect(screen.queryByRole('heading', { name: /OPG Report/ })).toBeNull();
  });
});
