import { env } from '$env/dynamic/public';
import type { TraceMode, TraceData, Meta } from './types';

const cache = new Map<string, unknown>();

/** Base path: /api/data (Blob proxy) by default; /data (static) only when PUBLIC_DATA_SOURCE=static */
function getDataBase(): string {
	return env.PUBLIC_DATA_SOURCE === 'static' ? '/data' : '/api/data';
}

function fullUrl(path: string): string {
	if (typeof window !== 'undefined') {
		return new URL(path, window.location.origin).href;
	}
	return path;
}

function dataUrl(relativePath: string): string {
	const p = relativePath.replace(/^\//, '');
	return fullUrl(`${getDataBase()}/${p}`);
}

function staticDataUrl(relativePath: string): string {
	return fullUrl(`/data/${relativePath.replace(/^\//, '')}`);
}

const FETCH_TIMEOUT_MS = 30_000;

async function fetchWithTimeout(url: string): Promise<Response> {
	const controller = new AbortController();
	const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
	try {
		const res = await fetch(url, { signal: controller.signal });
		return res;
	} finally {
		clearTimeout(timeout);
	}
}

async function fetchJson<T>(url: string, altUrl?: string): Promise<T> {
	const urlsToTry = altUrl ? [url, altUrl] : [url];
	for (const u of urlsToTry) {
		if (cache.has(u)) return cache.get(u) as T;
	}
	let lastError: Error | null = null;
	for (const u of urlsToTry) {
		try {
			const res = await fetchWithTimeout(u);
			if (!res.ok) {
				const text = await res.text();
				throw new Error(`Failed to fetch ${u}: ${res.status} ${res.statusText}${text ? ` - ${text.slice(0, 150)}` : ''}`);
			}
			const data = await res.json();
			cache.set(u, data);
			return data as T;
		} catch (e) {
			lastError = e instanceof Error ? e : new Error(String(e));
			if (e instanceof Error && e.name === 'AbortError') {
				lastError = new Error(`Request timed out after ${FETCH_TIMEOUT_MS / 1000}s: ${u}`);
			}
			// Try next URL if available
			if (urlsToTry.indexOf(u) < urlsToTry.length - 1) continue;
			throw lastError;
		}
	}
	throw lastError ?? new Error('Unknown fetch error');
}

export async function loadMeta(): Promise<Meta> {
	const apiUrl = dataUrl('meta.json');
	const staticUrl = staticDataUrl('meta.json');
	return fetchJson<Meta>(apiUrl, staticUrl);
}

export async function loadTraceData(mode: TraceMode): Promise<TraceData> {
	const apiUrl = (p: string) => dataUrl(`${mode}/${p}`);
	const staticUrl = (p: string) => staticDataUrl(`${mode}/${p}`);
	const [requests, forwardPasses, join] = await Promise.all([
		fetchJson<TraceData['requests']>(apiUrl('requests.json'), staticUrl('requests.json')),
		fetchJson<TraceData['forwardPasses']>(apiUrl('forward_passes.json'), staticUrl('forward_passes.json')),
		fetchJson<TraceData['join']>(apiUrl('join.json'), staticUrl('join.json'))
	]);

	let kvCache: TraceData['kvCache'] = [];
	if (mode === 'multiturn') {
		try {
			kvCache = await fetchJson<TraceData['kvCache']>(apiUrl('kv_cache.json'), staticUrl('kv_cache.json'));
		} catch { /* no kv cache data */ }
	}

	let perToken: TraceData['perToken'] = [];
	if (mode === 'streaming' || mode === 'multiturn') {
		try {
			perToken = await fetchJson<TraceData['perToken']>(apiUrl('per_token.json'), staticUrl('per_token.json'));
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
