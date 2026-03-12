<script lang="ts">
	import { onMount } from 'svelte';
	import Chart from '../Chart.svelte';
	import MetricCard from '../MetricCard.svelte';
	import { median, percentile, mean } from '$lib/stats';
	import { loadMeta, loadTraceData } from '$lib/data';
	import type { TraceMode } from '$lib/types';

	interface FwdEntry { fwd_id: number; duration_ms: number; batch_size: number; total_tokens: number; start_s: number }

	let traceA = $state<{ name: string; fwds: FwdEntry[] } | null>(null);
	let traceB = $state<{ name: string; fwds: FwdEntry[] } | null>(null);
	let threshold = $state(2);
	let alignMode = $state<'fwd_id' | 'index'>('fwd_id');
	let uploadError = $state('');
	type PresetSource = { id: string; label: string; fwds: FwdEntry[] };
	let presetSources = $state<PresetSource[]>([]);
	let presetA = $state('');
	let presetB = $state('');

	function parseFwdFromPerfetto(events: Record<string, unknown>[]): FwdEntry[] {
		const fwdMap = new Map<number, FwdEntry>();
		for (const e of events) {
			if (e.cat === 'forward_pass' && e.ph === 'X') {
				const name = String(e.name || '');
				const match = name.match(/fwd_(\d+)/);
				if (match) {
					const fid = parseInt(match[1]);
					const dur = (e.dur as number) / 1000;
					const ts = (e.ts as number) / 1_000_000;
					const args = (e.args || {}) as Record<string, number>;
					if (!fwdMap.has(fid) || dur > (fwdMap.get(fid)?.duration_ms || 0)) {
						fwdMap.set(fid, { fwd_id: fid, duration_ms: dur, batch_size: args.batch_size || 0, total_tokens: args.total_tokens || 0, start_s: ts });
					}
				}
			}
		}
		return [...fwdMap.values()].sort((a, b) => a.fwd_id - b.fwd_id);
	}

	function parseFwdFromRaw(data: Record<string, unknown>[]): FwdEntry[] {
		return data.map(d => ({
			fwd_id: d.fwd_id as number,
			duration_ms: d.duration_ms as number,
			batch_size: d.batch_size as number || (d.req_ids as string[])?.length || 0,
			total_tokens: d.total_tokens as number || 0,
			start_s: d.rel_start_s as number || 0,
		})).sort((a, b) => a.fwd_id - b.fwd_id);
	}

	function parseJsonl(text: string): Record<string, unknown>[] {
		return text.trim().split('\n').filter(l => l.trim()).map(l => JSON.parse(l));
	}

	function loadPreset(slot: 'A' | 'B') {
		const id = slot === 'A' ? presetA : presetB;
		if (!id) return;
		const preset = presetSources.find(p => p.id === id);
		if (!preset) return;
		if (slot === 'A') traceA = { name: preset.label, fwds: preset.fwds };
		else traceB = { name: preset.label, fwds: preset.fwds };
		uploadError = '';
	}

	function handleUpload(slot: 'A' | 'B', event: Event) {
		uploadError = '';
		const input = event.target as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) return;
		const reader = new FileReader();
		reader.onload = () => {
			try {
				const text = reader.result as string;
				let data: unknown;
				try {
					data = JSON.parse(text);
				} catch {
					const lines = parseJsonl(text);
					if (lines.length > 0) {
						data = lines;
					} else {
						uploadError = `Trace ${slot}: "${file.name}" is not valid JSON or JSONL.`;
						return;
					}
				}

				let fwds: FwdEntry[];
				if (data && typeof data === 'object' && !Array.isArray(data) && 'traceEvents' in (data as Record<string, unknown>)) {
					fwds = parseFwdFromPerfetto((data as { traceEvents: Record<string, unknown>[] }).traceEvents);
				} else if (Array.isArray(data) && data.length > 0 && 'fwd_id' in data[0]) {
					fwds = parseFwdFromRaw(data);
				} else if (Array.isArray(data) && data.length > 0) {
					fwds = parseFwdFromPerfetto(data);
				} else {
					uploadError = `Trace ${slot}: Unrecognized format in "${file.name}". Expected Perfetto JSON, forward_passes.json array, or _fwd.jsonl file.`;
					return;
				}
				if (fwds.length === 0) {
					uploadError = `Trace ${slot}: No forward pass data found in "${file.name}". Ensure it contains forward pass events or entries with fwd_id.`;
					return;
				}
				uploadError = '';
				if (slot === 'A') traceA = { name: file.name, fwds };
				else traceB = { name: file.name, fwds };
			} catch (e) {
				uploadError = `Trace ${slot}: Failed to parse "${file.name}" — ${e instanceof Error ? e.message : 'unknown error'}`;
			}
		};
		reader.readAsText(file);
	}

	interface CompRow { idx: number; fwdIdA: number; fwdIdB: number; durA: number; durB: number; diff: number; pctDiff: number; highlighted: boolean }

	let comparison = $derived.by(() => {
		if (!traceA || !traceB) return [];
		const rows: CompRow[] = [];
		if (alignMode === 'fwd_id') {
			const allIds = new Set([...traceA.fwds.map(f => f.fwd_id), ...traceB.fwds.map(f => f.fwd_id)]);
			const mapA = new Map(traceA.fwds.map(f => [f.fwd_id, f]));
			const mapB = new Map(traceB.fwds.map(f => [f.fwd_id, f]));
			for (const fid of [...allIds].sort((a, b) => a - b)) {
				const a = mapA.get(fid);
				const b = mapB.get(fid);
				const durA = a?.duration_ms ?? 0;
				const durB = b?.duration_ms ?? 0;
				const diff = Math.abs(durA - durB);
				const base = Math.max(durA, durB, 0.001);
				rows.push({ idx: fid, fwdIdA: a?.fwd_id ?? -1, fwdIdB: b?.fwd_id ?? -1, durA, durB, diff, pctDiff: (diff / base) * 100, highlighted: diff > threshold });
			}
		} else {
			const len = Math.max(traceA.fwds.length, traceB.fwds.length);
			for (let i = 0; i < len; i++) {
				const a = traceA.fwds[i];
				const b = traceB.fwds[i];
				const durA = a?.duration_ms ?? 0;
				const durB = b?.duration_ms ?? 0;
				const diff = Math.abs(durA - durB);
				const base = Math.max(durA, durB, 0.001);
				rows.push({ idx: i, fwdIdA: a?.fwd_id ?? -1, fwdIdB: b?.fwd_id ?? -1, durA, durB, diff, pctDiff: (diff / base) * 100, highlighted: diff > threshold });
			}
		}
		return rows;
	});

	let summaryStats = $derived.by(() => {
		if (!comparison.length) return null;
		const diffs = comparison.map(r => r.diff);
		const pctDiffs = comparison.filter(r => r.durA > 0 && r.durB > 0).map(r => r.pctDiff);
		const highlighted = comparison.filter(r => r.highlighted).length;
		return {
			total: comparison.length,
			highlighted,
			withinThreshold: comparison.length - highlighted,
			meanDiff: mean(diffs).toFixed(2),
			medianDiff: median(diffs).toFixed(2),
			p95Diff: percentile(diffs, 95).toFixed(2),
			maxDiff: Math.max(...diffs).toFixed(2),
			meanPctDiff: pctDiffs.length ? mean(pctDiffs).toFixed(1) : '–',
		};
	});

	function buildDurationOverlay() {
		if (!comparison.length) return null;
		return {
			backgroundColor: 'transparent',
			title: { text: 'Forward Pass Duration Overlay', textStyle: { color: '#E8EAF0', fontSize: 14, fontWeight: 500 }, left: 16, top: 12 },
			tooltip: {
				trigger: 'axis', backgroundColor: '#1C2030', borderColor: 'rgba(255,255,255,0.09)', textStyle: { color: '#E8EAF0' },
			},
			legend: { bottom: 8, textStyle: { color: '#9BA1B0' } },
			grid: { left: 70, right: 30, top: 55, bottom: 50 },
			xAxis: { type: 'value', name: alignMode === 'fwd_id' ? 'Forward Pass ID' : 'Index', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			yAxis: { type: 'value', name: 'Duration (ms)', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 5, height: 20, borderColor: 'rgba(255,255,255,0.06)' }],
			series: [
				{
					name: traceA?.name || 'Trace A',
					type: 'line',
					data: comparison.map(r => [r.idx, r.durA]),
					lineStyle: { width: 2.75, color: '#8FB8FF' },
					symbol: 'circle',
					showSymbol: false,
					itemStyle: { color: '#8FB8FF' },
					emphasis: { focus: 'series' },
				},
				{
					name: traceB?.name || 'Trace B',
					type: 'line',
					data: comparison.map(r => [r.idx, r.durB]),
					lineStyle: { width: 2.75, color: '#FF9B7A' },
					symbol: 'circle',
					showSymbol: false,
					itemStyle: { color: '#FF9B7A' },
					emphasis: { focus: 'series' },
				},
			],
		} as echarts.EChartsOption;
	}

	function buildDiffChart() {
		if (!comparison.length) return null;
		return {
			backgroundColor: 'transparent',
			title: { text: 'Duration Difference (|A − B|)', textStyle: { color: '#E8EAF0', fontSize: 14, fontWeight: 500 }, left: 16, top: 12 },
			tooltip: {
				trigger: 'axis', backgroundColor: '#1C2030', borderColor: 'rgba(255,255,255,0.09)', textStyle: { color: '#E8EAF0' },
				formatter: (params: { value: [number, number] }[]) => {
					const idx = params[0]?.value[0];
					const row = comparison.find(r => r.idx === idx);
					if (!row) return '';
					return `<b>${alignMode === 'fwd_id' ? 'fwd_' + idx : 'idx ' + idx}</b><br>A: ${row.durA.toFixed(1)} ms<br>B: ${row.durB.toFixed(1)} ms<br>Diff: ${row.diff.toFixed(2)} ms<br>${row.highlighted ? '<span style="color:#FF9B7A">⚠ Exceeds threshold</span>' : '<span style="color:#7ED7AB">✓ Within threshold</span>'}`;
				}
			},
			grid: { left: 70, right: 30, top: 55, bottom: 50 },
			xAxis: { type: 'value', name: alignMode === 'fwd_id' ? 'Forward Pass ID' : 'Index', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			yAxis: { type: 'value', name: 'Diff (ms)', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 5, height: 20, borderColor: 'rgba(255,255,255,0.06)' }],
			series: [
				{
					type: 'bar',
					data: comparison.map(r => ({
						value: [r.idx, r.diff],
						itemStyle: { color: r.highlighted ? 'rgba(255,155,122,0.9)' : 'rgba(126,215,171,0.65)' },
					})),
					barMaxWidth: 8,
				},
				{
					type: 'line',
					data: [[comparison[0].idx, threshold], [comparison[comparison.length - 1].idx, threshold]],
					lineStyle: { type: 'dashed', color: '#F4CD78', width: 2 },
					symbol: 'none',
					name: `Threshold (${threshold} ms)`,
				},
			],
		} as echarts.EChartsOption;
	}

	function buildScatterCorrelation() {
		if (!comparison.length) return null;
		const valid = comparison.filter(r => r.durA > 0 && r.durB > 0);
		return {
			backgroundColor: 'transparent',
			title: { text: 'A vs B Correlation', textStyle: { color: '#E8EAF0', fontSize: 14, fontWeight: 500 }, left: 16, top: 12 },
			tooltip: {
				backgroundColor: '#1C2030', borderColor: 'rgba(255,255,255,0.09)', textStyle: { color: '#E8EAF0' },
				formatter: (p: { value: number[] }) => `A: ${p.value[0].toFixed(1)} ms<br>B: ${p.value[1].toFixed(1)} ms<br>Diff: ${Math.abs(p.value[0] - p.value[1]).toFixed(2)} ms`,
			},
			grid: { left: 70, right: 30, top: 55, bottom: 50 },
			xAxis: { type: 'value', name: 'A Duration (ms)', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			yAxis: { type: 'value', name: 'B Duration (ms)', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			series: [
				{
					type: 'scatter',
					data: valid.map(r => ({
						value: [r.durA, r.durB],
						itemStyle: {
							color: r.highlighted ? 'rgba(255,155,122,0.92)' : 'rgba(143,184,255,0.9)',
							borderColor: r.highlighted ? '#FFD0C2' : '#D8E7FF',
							borderWidth: 1,
						},
					})),
					symbolSize: 8,
				},
				{
					type: 'line',
					data: (() => {
						const maxVal = Math.max(...valid.map(r => Math.max(r.durA, r.durB)));
						return [[0, 0], [maxVal, maxVal]];
					})(),
					lineStyle: { type: 'dashed', color: 'rgba(255,255,255,0.15)', width: 1 },
					symbol: 'none',
					name: 'y = x',
				},
			],
		} as echarts.EChartsOption;
	}

	onMount(async () => {
		try {
			const m = await loadMeta();
			const modes = m.modes as TraceMode[];
			const presets: PresetSource[] = [];
			for (const mode of modes) {
				const data = await loadTraceData(mode);
				const byModel = new Map<string, FwdEntry[]>();
				for (const f of data.forwardPasses) {
					if (!byModel.has(f.model_label)) byModel.set(f.model_label, []);
					byModel.get(f.model_label)!.push({
						fwd_id: f.fwd_id,
						duration_ms: f.duration_ms,
						batch_size: f.batch_size,
						total_tokens: f.total_tokens,
						start_s: f.rel_start_s,
					});
				}
				for (const [modelLabel, fwds] of byModel.entries()) {
					const id = `${mode}:${modelLabel}`;
					presets.push({
						id,
						label: `${mode} • ${modelLabel} (${fwds.length} fwds)`,
						fwds: fwds.sort((a, b) => a.fwd_id - b.fwd_id),
					});
				}
			}
			presetSources = presets;
			if (presets.length > 0) {
				presetA = presets[0].id;
				presetB = presets[Math.min(1, presets.length - 1)].id;
			}
		} catch {
			/* ignore preset preload failures */
		}
	});
</script>

<style>
	input[type="file"]::file-selector-button {
		background: var(--accent-subtle);
		color: var(--accent-primary);
	}
	input[type="file"]::file-selector-button:hover {
		background: rgba(124, 147, 219, 0.2);
	}
	tr.compare-row:hover {
		background: var(--surface-hover) !important;
	}
</style>

<div class="space-y-8" style="padding: 1.75rem 2rem;">
	<div>
		<h2 class="font-bold text-[1.25rem]" style="color: var(--text-primary)">Forward Pass Comparison</h2>
		<p class="text-[0.8rem] mt-1" style="color: var(--text-muted)">Compare forward pass durations between two runs (vLLM vs vLLM, vLLM vs MIST, vLLM vs SGLang, etc.)</p>
	</div>

	<div class="rounded-lg p-6 space-y-4" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
		<p class="text-xs" style="color: var(--text-muted)">Upload Perfetto JSON (exported from this dashboard), preprocessed forward_passes.json, or raw _fwd.jsonl files. Supports .json and .jsonl formats.</p>

		<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
			<div>
				<label for="trace-a-upload" class="block text-xs font-medium mb-1.5" style="color: var(--text-secondary)">
					Trace A {traceA ? `— ${traceA.name} (${traceA.fwds.length} fwd passes)` : ''}
				</label>
				<input id="trace-a-upload" type="file" accept=".json,.jsonl,application/json" onchange={(e) => handleUpload('A', e)} class="block w-full text-sm file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-sm file:font-medium file:cursor-pointer" style="color: var(--text-secondary)" />
			</div>
			<div>
				<label for="trace-b-upload" class="block text-xs font-medium mb-1.5" style="color: var(--text-secondary)">
					Trace B {traceB ? `— ${traceB.name} (${traceB.fwds.length} fwd passes)` : ''}
				</label>
				<input id="trace-b-upload" type="file" accept=".json,.jsonl,application/json" onchange={(e) => handleUpload('B', e)} class="block w-full text-sm file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-sm file:font-medium file:cursor-pointer" style="color: var(--text-secondary)" />
			</div>
		</div>

		<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
			<div class="space-y-2">
				<label for="preset-a" class="block text-xs font-medium mb-1.5" style="color: var(--text-secondary)">
					Preset A (existing traces)
				</label>
				<div class="flex gap-2">
					<select id="preset-a" bind:value={presetA} class="w-full px-3 py-2 rounded-lg text-sm" style="background: var(--surface-overlay); border: 1px solid var(--border-default); color: var(--text-primary);">
						<option value="" disabled>Select preset</option>
						{#each presetSources as p}
							<option value={p.id}>{p.label}</option>
						{/each}
					</select>
					<button
						class="px-3 py-2 rounded-lg text-xs font-medium"
						style="background: var(--accent-subtle); color: var(--accent-primary); border: 1px solid rgba(124,147,219,0.18);"
						onclick={() => loadPreset('A')}
					>
						Load
					</button>
				</div>
			</div>
			<div class="space-y-2">
				<label for="preset-b" class="block text-xs font-medium mb-1.5" style="color: var(--text-secondary)">
					Preset B (existing traces)
				</label>
				<div class="flex gap-2">
					<select id="preset-b" bind:value={presetB} class="w-full px-3 py-2 rounded-lg text-sm" style="background: var(--surface-overlay); border: 1px solid var(--border-default); color: var(--text-primary);">
						<option value="" disabled>Select preset</option>
						{#each presetSources as p}
							<option value={p.id}>{p.label}</option>
						{/each}
					</select>
					<button
						class="px-3 py-2 rounded-lg text-xs font-medium"
						style="background: var(--accent-subtle); color: var(--accent-primary); border: 1px solid rgba(124,147,219,0.18);"
						onclick={() => loadPreset('B')}
					>
						Load
					</button>
				</div>
			</div>
		</div>

		{#if uploadError}
			<div class="rounded-lg px-4 py-2.5 text-sm" style="background: rgba(212,129,107,0.08); border: 1px solid rgba(212,129,107,0.2); color: var(--color-danger);">
				{uploadError}
			</div>
		{/if}

		<div class="flex items-center gap-6 flex-wrap">
			<div class="flex items-center gap-3">
				<label for="fwd-threshold" class="text-sm" style="color: var(--text-secondary)">Highlight threshold:</label>
				<input id="fwd-threshold" type="range" min="0.1" max="50" step="0.1" bind:value={threshold} class="w-48" />
				<span class="text-sm tabular-nums" style="color: var(--text-muted)">{threshold.toFixed(1)} ms</span>
			</div>
			<div class="flex items-center gap-3">
				<span class="text-sm" style="color: var(--text-secondary)">Align by:</span>
				<button onclick={() => alignMode = 'fwd_id'} class="px-3 py-1 rounded-full text-xs font-medium transition-all border" style={alignMode === 'fwd_id' ? 'background: var(--accent-subtle); color: var(--accent-primary); border: 1px solid rgba(124,147,219,0.18);' : 'color: var(--text-muted); border: 1px solid var(--border-default);'}>
					Forward Pass ID
				</button>
				<button onclick={() => alignMode = 'index'} class="px-3 py-1 rounded-full text-xs font-medium transition-all border" style={alignMode === 'index' ? 'background: var(--accent-subtle); color: var(--accent-primary); border: 1px solid rgba(124,147,219,0.18);' : 'color: var(--text-muted); border: 1px solid var(--border-default);'}>
					Sequential Index
				</button>
			</div>
		</div>
	</div>

	{#if comparison.length > 0 && summaryStats}
		<div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3">
			<MetricCard label="Total Passes" value={String(summaryStats.total)} />
			<MetricCard label="Highlighted" value={String(summaryStats.highlighted)} />
			<MetricCard label="Within Threshold" value={String(summaryStats.withinThreshold)} />
			<MetricCard label="Mean Diff" value="{summaryStats.meanDiff} ms" />
			<MetricCard label="Median Diff" value="{summaryStats.medianDiff} ms" />
			<MetricCard label="p95 Diff" value="{summaryStats.p95Diff} ms" />
			<MetricCard label="Max Diff" value="{summaryStats.maxDiff} ms" />
			<MetricCard label="Mean % Diff" value="{summaryStats.meanPctDiff}%" />
		</div>

		{#if buildDurationOverlay()}
			<div class="rounded-lg p-2" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
				<Chart options={buildDurationOverlay()!} height="420px" />
			</div>
		{/if}

		<div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
			{#if buildDiffChart()}
				<div class="rounded-lg p-2" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
					<Chart options={buildDiffChart()!} height="400px" />
				</div>
			{/if}

			{#if buildScatterCorrelation()}
				<div class="rounded-lg p-2" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
					<Chart options={buildScatterCorrelation()!} height="400px" />
				</div>
			{/if}
		</div>

		<div class="rounded-lg overflow-hidden" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
			<div class="px-4 py-3 flex items-center justify-between" style="border-bottom: 1px solid var(--border-subtle);">
				<h3 class="text-sm font-semibold" style="color: var(--text-secondary)">Per-Forward-Pass Comparison</h3>
				<span class="text-xs" style="color: var(--text-muted)">Highlighted rows exceed {threshold.toFixed(1)} ms threshold</span>
			</div>
			<div class="overflow-auto max-h-96">
				<table class="w-full text-sm">
					<thead class="sticky top-0" style="background: var(--surface-overlay);">
						<tr>
							<th class="px-3 py-2 text-left text-xs font-medium uppercase" style="color: var(--text-muted); border-bottom: 1px solid var(--border-subtle);">{alignMode === 'fwd_id' ? 'Fwd ID' : 'Index'}</th>
							<th class="px-3 py-2 text-right text-xs font-medium uppercase" style="color: var(--text-muted); border-bottom: 1px solid var(--border-subtle);">A (ms)</th>
							<th class="px-3 py-2 text-right text-xs font-medium uppercase" style="color: var(--text-muted); border-bottom: 1px solid var(--border-subtle);">B (ms)</th>
							<th class="px-3 py-2 text-right text-xs font-medium uppercase" style="color: var(--text-muted); border-bottom: 1px solid var(--border-subtle);">Diff (ms)</th>
							<th class="px-3 py-2 text-right text-xs font-medium uppercase" style="color: var(--text-muted); border-bottom: 1px solid var(--border-subtle);">% Diff</th>
							<th class="px-3 py-2 text-center text-xs font-medium uppercase" style="color: var(--text-muted); border-bottom: 1px solid var(--border-subtle);">Status</th>
						</tr>
					</thead>
					<tbody>
						{#each comparison as row, i}
							<tr class="compare-row transition-colors" style="background: {row.highlighted ? 'rgba(212,129,107,0.08)' : i % 2 === 0 ? 'var(--surface-base)' : 'var(--surface-raised)'};">
								<td class="px-3 py-1.5 tabular-nums" style="color: var(--text-secondary)">{row.idx}</td>
								<td class="px-3 py-1.5 text-right tabular-nums" style="color: var(--text-secondary)">{row.durA > 0 ? row.durA.toFixed(1) : '–'}</td>
								<td class="px-3 py-1.5 text-right tabular-nums" style="color: var(--text-secondary)">{row.durB > 0 ? row.durB.toFixed(1) : '–'}</td>
								<td class="px-3 py-1.5 text-right tabular-nums {row.highlighted ? 'font-medium' : ''}" style="color: {row.highlighted ? 'var(--color-danger)' : 'var(--text-muted)'}">{row.diff.toFixed(2)}</td>
								<td class="px-3 py-1.5 text-right tabular-nums" style="color: {row.highlighted ? 'var(--color-danger)' : 'var(--text-muted)'}">{row.pctDiff.toFixed(1)}%</td>
								<td class="px-3 py-1.5 text-center">{row.highlighted ? '⚠️' : '✓'}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</div>
	{:else if traceA || traceB}
		<div class="text-center py-12" style="color: var(--text-muted)">
			<p class="text-lg">Upload both traces to start comparison</p>
			<p class="text-sm mt-1">{traceA ? 'Trace A loaded' : 'Waiting for Trace A'} · {traceB ? 'Trace B loaded' : 'Waiting for Trace B'}</p>
		</div>
	{/if}
</div>
