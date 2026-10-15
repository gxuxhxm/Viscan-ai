/**
 * Regression tests for new-patient entry in the Confirm Clinical Review modal.
 *
 * The dialog must offer both entry modes — the existing manual form and a
 * validated JSON editor bound to the *actual* patients.json schema — and both
 * must normalise to the same patient payload on the same save endpoint, with
 * no data lost while switching modes and nothing saved when the JSON is
 * invalid. Existing-patient selection and the dictation → report payload
 * (findings, transcript, explicit confirmation) are covered here too.
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

const PATIENTS = [
  { id: 'P001', name: 'Priya Sharma', dob: '1988-03-15', gender: 'F', phone: '+91-98765-43210', email: 'priya@example.com', report_count: 3 },
  { id: 'P003', name: 'Arjun Mehta', dob: '1975-11-08', gender: 'M', phone: null, email: null, report_count: 2 },
];

const DICTATION_TEXT = 'Mild bone loss noted at lower left molar.';

// Shape of the backend's /dictation response (see backend/main.py).
const DICTATION_RESULT = {
  findings: [
    { site: 'lower left molar', observation: 'Mild bone loss noted at lower left molar.', severity: 'mild', context: 'current' },
  ],
  anatomical_site: 'lower left molar',
  draft_report: 'CLINICAL DICTATION DRAFT\n\nFINDINGS:\n- Mild bone loss noted at lower left molar.',
  uncertainties: ['AI-structured output: clinician verification required'],
  requires_review: true,
  demo_mode: false,
  fallback_reason: null,
  metadata: { tooth: '' },
  transcript: DICTATION_TEXT,
  usable: true,
};

const SAVED_REPORT = {
  id: 'R013',
  patient_id: 'P007',
  type: 'OPG',
  date: '2026-10-15',
  radiologist: 'Unassigned',
  status: 'final',
  findings: DICTATION_RESULT.findings,
  impression: 'Mild bone loss noted at lower left molar.',
  draft_report: DICTATION_RESULT.draft_report,
  clinician_confirmed: true,
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
  api.dictation.mockResolvedValue(DICTATION_RESULT);
  api.saveReport.mockResolvedValue(SAVED_REPORT);
});

/** Render the app, dictate a transcript and open the save confirmation modal. */
async function openSaveModal() {
  render(<App />);
  await waitFor(() => expect(api.listPatients).toHaveBeenCalled());
  fireEvent.click(screen.getByRole('button', { name: /Dictation/ }));
  const transcript = await screen.findByLabelText('Clinical dictation transcript');
  fireEvent.change(transcript, { target: { value: DICTATION_TEXT } });
  fireEvent.click(screen.getByRole('button', { name: /Create AI Report/ }));
  await screen.findByLabelText('Editable draft report');
  fireEvent.click(screen.getByRole('button', { name: /Save Report/ }));
  await screen.findByText('Confirm Clinical Review');
}

function selectNewPatient() {
  fireEvent.change(screen.getByLabelText(/File report under/), { target: { value: '__new__' } });
}

function patientModeGroup() {
  return screen.getByRole('group', { name: 'New patient entry mode' });
}

function switchTo(mode) {
  const label = mode === 'json' ? 'JSON' : 'Manual form';
  fireEvent.click(within(patientModeGroup()).getByRole('button', { name: label }));
}

function jsonEditor() {
  return screen.getByLabelText('New patient JSON');
}

function fillManualForm() {
  fireEvent.change(screen.getByLabelText(/Full name/), { target: { value: 'Asha Rao' } });
  fireEvent.change(screen.getByLabelText(/Date of birth/), { target: { value: '1990-05-14' } });
  fireEvent.change(screen.getByLabelText(/Gender/), { target: { value: 'F' } });
  fireEvent.change(screen.getByLabelText(/Phone/), { target: { value: '+91-90000-11111' } });
  fireEvent.change(screen.getByLabelText(/Email/), { target: { value: 'asha@example.com' } });
}

function confirmSave() {
  fireEvent.click(screen.getByRole('button', { name: /Yes, Confirm/ }));
}

// Both the save dialog and the JSON editor raise role="alert" notices, so
// assertions read the whole message area.
function alertText() {
  return screen.getAllByRole('alert').map(el => el.textContent).join('\n');
}

// ── Existing patient selection ────────────────────────────────────────────────

describe('save report: existing patient', () => {
  it('files the structured report under the selected existing patient', async () => {
    await openSaveModal();

    fireEvent.change(screen.getByLabelText(/File report under/), { target: { value: 'P003' } });
    confirmSave();

    await waitFor(() => expect(api.saveReport).toHaveBeenCalledTimes(1));
    const payload = api.saveReport.mock.calls[0][0];
    expect(payload.patient_id).toBe('P003');
    expect(payload.patient).toBeUndefined();
    // Dictation output is saved with the existing report schema…
    expect(payload.findings).toEqual(DICTATION_RESULT.findings);
    expect(payload.transcript).toBe(DICTATION_TEXT);
    expect(payload.anatomical_site).toBe('lower left molar');
    expect(payload.draft_report).toContain('Mild bone loss noted at lower left molar.');
    // …and only an explicit confirmation marks it clinician-confirmed.
    expect(payload.clinician_confirmed).toBe(true);
    expect(payload.report_type).toBe('OPG');

    expect(await screen.findByText(/Saved as report #R013/)).toBeInTheDocument();
  });

  it('keeps the report-type selector with the supported modalities', async () => {
    await openSaveModal();
    const select = screen.getByLabelText(/Report type/);
    expect(Array.from(select.options).map(o => o.value)).toEqual(
      ['OPG', 'PERIAPICAL', 'BITEWING', 'CBCT']
    );
  });
});

// ── Manual form ───────────────────────────────────────────────────────────────

describe('save report: manual new patient', () => {
  it('creates a new patient from the manual form, without an id', async () => {
    await openSaveModal();
    selectNewPatient();

    // The manual form is the default mode with the original fields
    expect(screen.getByLabelText(/Full name/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Date of birth/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Gender/)).toBeInTheDocument();
    fillManualForm();
    confirmSave();

    await waitFor(() => expect(api.saveReport).toHaveBeenCalledTimes(1));
    expect(api.saveReport.mock.calls[0][0].patient).toEqual({
      name: 'Asha Rao',
      dob: '1990-05-14',
      gender: 'F',
      phone: '+91-90000-11111',
      email: 'asha@example.com',
    });
  });

  it('still refuses to save an incomplete manual form', async () => {
    await openSaveModal();
    selectNewPatient();
    confirmSave();

    expect(api.saveReport).not.toHaveBeenCalled();
    expect(screen.getByRole('alert')).toHaveTextContent(/name, date of birth and gender/);
  });
});

// ── JSON mode ─────────────────────────────────────────────────────────────────

describe('save report: JSON new patient', () => {
  it('saves a valid JSON patient normalised to the same payload as the form', async () => {
    await openSaveModal();
    selectNewPatient();
    switchTo('json');

    fireEvent.change(jsonEditor(), {
      target: {
        value: JSON.stringify({
          name: '  Vikram Rao ',
          dob: '1987-06-11',
          gender: 'M',
          phone: null,
          email: 'vikram@example.com',
        }),
      },
    });
    confirmSave();

    await waitFor(() => expect(api.saveReport).toHaveBeenCalledTimes(1));
    expect(api.saveReport.mock.calls[0][0].patient).toEqual({
      name: 'Vikram Rao',
      dob: '1987-06-11',
      gender: 'M',
      phone: null,
      email: 'vikram@example.com',
    });
  });

  it('shows a schema example generated from the existing patient schema', async () => {
    await openSaveModal();
    selectNewPatient();
    switchTo('json');

    // The editor placeholder is the template…
    expect(jsonEditor().placeholder).toContain('"dob"');
    expect(jsonEditor().placeholder).toContain('"gender"');
    expect(jsonEditor().placeholder).not.toContain('"id"');

    // …and the button fills the editor with that same example.
    fireEvent.click(screen.getByRole('button', { name: /Insert example/ }));
    expect(jsonEditor().value).toContain('"asha.rao@example.com"');
    expect(() => JSON.parse(jsonEditor().value)).not.toThrow();
  });

  it('refuses to save the untouched example template', async () => {
    await openSaveModal();
    selectNewPatient();
    switchTo('json');
    fireEvent.click(screen.getByRole('button', { name: /Insert example/ }));
    confirmSave();

    expect(api.saveReport).not.toHaveBeenCalled();
    expect(alertText()).toMatch(/example template/);
  });

  it('reports JSON syntax errors and saves nothing', async () => {
    await openSaveModal();
    selectNewPatient();
    switchTo('json');
    const broken = '{ "name": "Asha Rao", "dob": }';
    fireEvent.change(jsonEditor(), { target: { value: broken } });
    confirmSave();

    await waitFor(() => expect(screen.getAllByRole('alert').length).toBeGreaterThan(0));
    expect(api.saveReport).not.toHaveBeenCalled();
    expect(screen.getAllByRole('alert').some(el => /Invalid JSON/.test(el.textContent))).toBe(true);
    // The broken text is still there to fix.
    expect(jsonEditor().value).toBe(broken);
  });

  it('reports missing required fields', async () => {
    await openSaveModal();
    selectNewPatient();
    switchTo('json');
    fireEvent.change(jsonEditor(), { target: { value: JSON.stringify({ name: 'Asha Rao', gender: 'F' }) } });
    confirmSave();

    expect(api.saveReport).not.toHaveBeenCalled();
    expect(alertText()).toContain('"dob" is required.');
  });

  it('reports wrong data types', async () => {
    await openSaveModal();
    selectNewPatient();
    switchTo('json');
    fireEvent.change(jsonEditor(), {
      target: { value: JSON.stringify({ name: 12, dob: '1990-01-01', gender: 'F', phone: 9876543210 }) },
    });
    confirmSave();

    expect(api.saveReport).not.toHaveBeenCalled();
    const alerts = alertText();
    expect(alerts).toContain('"name" must be a string.');
    expect(alerts).toContain('"phone" must be a string or null.');
  });

  it('reports unsupported fields instead of silently dropping them', async () => {
    await openSaveModal();
    selectNewPatient();
    switchTo('json');
    fireEvent.change(jsonEditor(), {
      target: { value: JSON.stringify({ name: 'Asha Rao', dob: '1990-01-01', gender: 'F', address: '21 Main St' }) },
    });
    confirmSave();

    expect(api.saveReport).not.toHaveBeenCalled();
    expect(alertText()).toContain('Unsupported field: address.');
  });

  it('rejects a caller-supplied patient id', async () => {
    await openSaveModal();
    selectNewPatient();
    switchTo('json');
    fireEvent.change(jsonEditor(), {
      target: { value: JSON.stringify({ id: 'P001', name: 'Asha Rao', dob: '1990-01-01', gender: 'F' }) },
    });
    confirmSave();

    expect(api.saveReport).not.toHaveBeenCalled();
    expect(alertText()).toMatch(/"id" is generated by the server/);
  });

  it('rejects a malformed dob', async () => {
    await openSaveModal();
    selectNewPatient();
    switchTo('json');
    fireEvent.change(jsonEditor(), {
      target: { value: JSON.stringify({ name: 'Asha Rao', dob: '1990-02-31', gender: 'F' }) },
    });
    confirmSave();

    expect(api.saveReport).not.toHaveBeenCalled();
    expect(alertText()).toMatch(/not a real calendar date/);
  });
});

// ── Switching modes keeps the entered data ────────────────────────────────────

describe('save report: switching entry modes', () => {
  it('Manual → JSON seeds the editor with the current form values', async () => {
    await openSaveModal();
    selectNewPatient();
    fillManualForm();
    switchTo('json');

    const parsed = JSON.parse(jsonEditor().value);
    expect(parsed).toEqual({
      name: 'Asha Rao',
      dob: '1990-05-14',
      gender: 'F',
      phone: '+91-90000-11111',
      email: 'asha@example.com',
    });
  });

  it('JSON → Manual fills the form from valid JSON, then back again', async () => {
    await openSaveModal();
    selectNewPatient();
    switchTo('json');
    fireEvent.change(jsonEditor(), {
      target: { value: JSON.stringify({ name: 'Meera Iyer', dob: '1985-12-01', gender: 'Other', phone: null, email: 'meera@example.com' }) },
    });
    switchTo('manual');

    expect(screen.getByLabelText(/Full name/)).toHaveValue('Meera Iyer');
    expect(screen.getByLabelText(/Date of birth/)).toHaveValue('1985-12-01');
    expect(screen.getByLabelText(/Gender/)).toHaveValue('Other');
    expect(screen.getByLabelText(/Phone/)).toHaveValue('');
    expect(screen.getByLabelText(/Email/)).toHaveValue('meera@example.com');

    // Round-tripping back to JSON keeps the same data (empty → null).
    switchTo('json');
    expect(JSON.parse(jsonEditor().value)).toEqual({
      name: 'Meera Iyer',
      dob: '1985-12-01',
      gender: 'Other',
      phone: null,
      email: 'meera@example.com',
    });
  });

  it('returns from an untouched JSON editor to the form without friction', async () => {
    await openSaveModal();
    selectNewPatient();
    switchTo('json');
    // The editor is seeded from the (empty) form, i.e. the schema skeleton
    expect(JSON.parse(jsonEditor().value)).toEqual({
      name: '', dob: '', gender: '', phone: null, email: null,
    });
    switchTo('manual');

    // Nothing was typed in JSON mode, so the switch needs no validation
    expect(screen.getByLabelText(/Full name/)).toBeInTheDocument();
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('invalid JSON is kept in the editor and the switch is refused with a reason', async () => {
    await openSaveModal();
    selectNewPatient();
    switchTo('json');
    const broken = '{ "name": "Asha Rao" ';
    fireEvent.change(jsonEditor(), { target: { value: broken } });
    switchTo('manual');

    // Still in JSON mode, nothing discarded, and the reason is on screen.
    expect(screen.getByLabelText('New patient JSON')).toBeInTheDocument();
    expect(jsonEditor().value).toBe(broken);
    expect(screen.getByRole('alert')).toHaveTextContent(/Invalid JSON/);
    expect(screen.queryByLabelText(/Full name/)).toBeNull();
  });
});

// ── Failure recovery ──────────────────────────────────────────────────────────

describe('save report: failure recovery', () => {
  it('keeps the JSON patient and the editable draft when the save fails', async () => {
    api.saveReport.mockRejectedValue(new Error('patients.json is locked'));
    await openSaveModal();
    selectNewPatient();
    switchTo('json');
    const json = JSON.stringify({ name: 'Asha Rao', dob: '1990-01-01', gender: 'F', phone: null, email: null });
    fireEvent.change(jsonEditor(), { target: { value: json } });
    confirmSave();

    await waitFor(() => expect(api.saveReport).toHaveBeenCalledTimes(1));
    expect(await screen.findByText(/patients\.json is locked/)).toBeInTheDocument();
    // Dialog, entered JSON and the draft are all still there for a retry.
    expect(screen.getByText('Confirm Clinical Review')).toBeInTheDocument();
    expect(jsonEditor().value).toBe(json);
    expect(screen.getByLabelText('Editable draft report')).toHaveValue(DICTATION_RESULT.draft_report);
    expect(screen.queryByText(/Saved as report/)).toBeNull();

    // Retrying after the failure re-saves the same data, not a duplicate form.
    api.saveReport.mockResolvedValue(SAVED_REPORT);
    confirmSave();
    await waitFor(() => expect(api.saveReport).toHaveBeenCalledTimes(2));
    expect(api.saveReport.mock.calls[1][0].patient).toEqual({
      name: 'Asha Rao',
      dob: '1990-01-01',
      gender: 'F',
      phone: null,
      email: null,
    });
  });

  it('keeps the manual form when the save fails', async () => {
    api.saveReport.mockRejectedValue(new Error('disk full'));
    await openSaveModal();
    selectNewPatient();
    fillManualForm();
    confirmSave();

    await waitFor(() => expect(screen.getByText(/disk full/)).toBeInTheDocument());
    expect(screen.getByLabelText(/Full name/)).toHaveValue('Asha Rao');
    expect(screen.getByLabelText(/Date of birth/)).toHaveValue('1990-05-14');
    expect(screen.getByLabelText(/Email/)).toHaveValue('asha@example.com');
  });
});
