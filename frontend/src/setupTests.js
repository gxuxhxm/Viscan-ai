import '@testing-library/jest-dom/vitest';
import { afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';

// Vitest runs without injected globals, so Testing Library's automatic
// cleanup never registers itself, so mount one clean App per test.
afterEach(() => cleanup());

// jsdom does not implement scrollIntoView, which the chat panel calls to keep
// the newest message in view; stub it so component effects never throw.
Element.prototype.scrollIntoView = vi.fn();
