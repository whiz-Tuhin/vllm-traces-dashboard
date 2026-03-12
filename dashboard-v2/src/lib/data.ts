import { env } from '$env/dynamic/public';
import type { TraceMode, TraceData, Meta } from './types';

const cache = new Map<string, unknown>();

/** Base path: /api/data when using Blob proxy, /data when using static files */
function getDataBase(): string {
	return env.PUBLIC_DATA_SOURCE === 'blob' ? '/api/data' : '/data';
}

function dataUrl(relativePath: string): string {
	return `${getDataBase()}/${relativePath.replace(/^\//, '')}`;
}

async function fetchJson<T>(url: string): Promise<T> {
	if (cache.has(url)) return cache.get(url) as T;
	const res = await fetch(url);
	const data = await res.json();
	cache.set(url, data);
	return data as T;
}

export async function loadMeta(): Promise<Meta> {
	return fetchJson<Meta>(dataUrl('meta.json'));
}

export async function loadTraceData(mode: TraceMode): Promise<TraceData> {
	const [requests, forwardPasses, join] = await Promise.all([
		fetchJson<TraceData['requests']>(dataUrl(`${mode}/requests.json`)),
		fetchJson<TraceData['forwardPasses']>(dataUrl(`${mode}/forward_passes.json`)),
		fetchJson<TraceData['join']>(dataUrl(`${mode}/join.json`))
	]);

	let kvCache: TraceData['kvCache'] = [];
	if (mode === 'multiturn') {
		try {
			kvCache = await fetchJson<TraceData['kvCache']>(dataUrl(`${mode}/kv_cache.json`));
		} catch { /* no kv cache data */ }
	}

	let perToken: TraceData['perToken'] = [];
	if (mode === 'streaming' || mode === 'multiturn') {
		try {
			perToken = await fetchJson<TraceData['perToken']>(dataUrl(`${mode}/per_token.json`));
		} catch { /* no per-token data */ }
	}

	return { requests, forwardPasses, join, kvCache, perToken };
}

export async function loadAllTraceDataBundle(): Promise<{
	meta: Meta;
	modes: Record<TraceMode, TraceData>;
}> {
	const m = await loadMeta();
	const modeEntries = await Promise.all(
		(m.modes as TraceMode[]).map(async (mode) => [mode, await loadTraceData(mode)] as const)
	);
	const modes = Object.fromEntries(modeEntries) as Record<TraceMode, TraceData>;
	return { meta: m, modes };
}
