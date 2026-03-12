import { writable, derived } from 'svelte/store';
import type { TraceMode, TraceData, Meta } from './types';

export const currentMode = writable<TraceMode>('streaming');
export const currentTab = writable<string>('overview');
export const meta = writable<Meta>({
	models: [],
	modes: [],
	colors: {},
	phase_colors: {}
});
export const traceData = writable<TraceData>({
	requests: [],
	forwardPasses: [],
	join: [],
	kvCache: [],
	perToken: []
});
export const loading = writable<boolean>(false);

export const isStreaming = derived(currentMode, ($m) => $m === 'streaming');
export const isMultiturn = derived(currentMode, ($m) => $m === 'multiturn');
export const hasPerToken = derived(currentMode, ($m) => $m === 'streaming' || $m === 'multiturn');

export const availableTabs = derived(currentMode, ($m) => {
	const tabs = [
		{ id: 'overview', label: 'Overview', icon: 'chart-bar' },
		{ id: 'forward-pass', label: 'Forward Pass', icon: 'bolt' },
	];
	if ($m === 'streaming' || $m === 'multiturn') {
		tabs.push({ id: 'per-request', label: 'Per-Request', icon: 'magnifying-glass' });
	}
	if ($m === 'multiturn') {
		tabs.push({ id: 'kv-cache', label: 'KV Cache & Turns', icon: 'database' });
	}
	tabs.push({ id: 'fwd-compare', label: 'Fwd Compare', icon: 'compare' });
	tabs.push({ id: 'data-export', label: 'Data & Export', icon: 'table' });
	return tabs;
});
