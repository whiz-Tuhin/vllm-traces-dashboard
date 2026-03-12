<script lang="ts">
	import { traceData, meta, currentMode } from '$lib/stores';
	import { loadAllTraceDataBundle } from '$lib/data';
	import DataTable from '../DataTable.svelte';
	import Chart from '../Chart.svelte';

	let modelFilter = $state<string[]>([]);
	let uploadedPerfettoA = $state<{ name: string; events: Record<string, unknown>[] } | null>(null);
	let uploadedPerfettoB = $state<{ name: string; events: Record<string, unknown>[] } | null>(null);
	let comparisonResult = $state<{ fwdId: number; durA: number; durB: number; diff: number; highlighted: boolean }[]>([]);
	let compThreshold = $state(2);

	$effect(() => {
		if ($meta.models.length && !modelFilter.length) {
			modelFilter = [...$meta.models];
		}
	});

	function toggleModel(m: string) {
		if (modelFilter.includes(m)) {
			modelFilter = modelFilter.filter(x => x !== m);
		} else {
			modelFilter = [...modelFilter, m];
		}
	}

	let filteredReqs = $derived($traceData.requests.filter(r => modelFilter.includes(r.model_label)));
	let filteredFwd = $derived($traceData.forwardPasses.filter(f => modelFilter.includes(f.model_label)));

	const reqColumns = [
		{ key: 'model_label', label: 'Model' },
		{ key: 'prompt_id', label: 'Prompt ID' },
		{ key: 'prompt_tokens', label: 'Prompt Tok' },
		{ key: 'output_tokens', label: 'Output Tok' },
		{ key: 'ttft_ms', label: 'TTFT (ms)', format: (v: unknown) => (v as number).toFixed(1) },
		{ key: 'tpot_ms', label: 'TPOT (ms)', format: (v: unknown) => (v as number).toFixed(2) },
		{ key: 'total_latency_ms', label: 'E2E (ms)', format: (v: unknown) => (v as number).toFixed(0) },
		{ key: 'scheduling_overhead_ms', label: 'Sched (ms)', format: (v: unknown) => (v as number).toFixed(1) },
	];

	const fwdColumns = [
		{ key: 'model_label', label: 'Model' },
		{ key: 'fwd_id', label: 'Fwd ID' },
		{ key: 'rel_start_s', label: 'Start (s)', format: (v: unknown) => (v as number).toFixed(3) },
		{ key: 'duration_ms', label: 'Duration (ms)', format: (v: unknown) => (v as number).toFixed(1) },
		{ key: 'batch_size', label: 'Batch' },
		{ key: 'total_tokens', label: 'Tokens' },
	];

	function downloadCsv(data: Record<string, unknown>[], filename: string) {
		if (!data.length) return;
		const keys = Object.keys(data[0]);
		const csv = [keys.join(','), ...data.map(row => keys.map(k => {
			const v = row[k];
			if (typeof v === 'string' && v.includes(',')) return `"${v}"`;
			return String(v ?? '');
		}).join(','))].join('\n');
		const blob = new Blob([csv], { type: 'text/csv' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = filename;
		a.click();
		URL.revokeObjectURL(url);
	}

	function buildPerfettoJson() {
		const events: Record<string, unknown>[] = [];
		const reqs = filteredReqs.sort((a, b) => a.api_receive_ts - b.api_receive_ts);
		if (!reqs.length) return '';
		const globalT0 = Math.min(...reqs.map(r => r.api_receive_ts));
		const us = (ts: number) => (ts - globalT0) * 1_000_000;

		const reqIdToPid = new Map<string, number>();

		for (let pidIdx = 0; pidIdx < reqs.length; pidIdx++) {
			const req = reqs[pidIdx];
			const pid = pidIdx + 1;
			reqIdToPid.set(req.request_id, pid);
			events.push({ ph: 'M', pid, tid: 0, name: 'process_name', args: { name: `${req.prompt_id} (${req.request_id})` } });
			events.push({ ph: 'M', pid, tid: 1, name: 'thread_name', args: { name: 'Phases' } });
			events.push({ ph: 'M', pid, tid: 2, name: 'thread_name', args: { name: 'Forward Passes' } });
			events.push({ ph: 'M', pid, tid: 3, name: 'thread_name', args: { name: 'Tokens' } });

			const schedTs = us(req.api_receive_ts);
			const engineTs = us(req.engine_add_request_ts);
			const firstTs = us(req.first_token_ts);
			const compTs = us(req.completion_ts);

			events.push({ ph: 'X', pid, tid: 1, name: 'Scheduling', ts: schedTs, dur: engineTs - schedTs, cat: 'phase' });
			events.push({ ph: 'X', pid, tid: 1, name: 'Prefill', ts: engineTs, dur: firstTs - engineTs, cat: 'phase' });
			events.push({ ph: 'X', pid, tid: 1, name: 'Decode', ts: firstTs, dur: compTs - firstTs, cat: 'phase' });
		}

		const joinByReq = new Map<string, typeof $traceData.join>();
		for (const j of $traceData.join.filter(j => modelFilter.includes(j.model_label))) {
			if (!joinByReq.has(j.request_id)) joinByReq.set(j.request_id, []);
			joinByReq.get(j.request_id)!.push(j);
		}
		for (const [rid, joins] of joinByReq) {
			const pid = reqIdToPid.get(rid);
			if (!pid) continue;
			const req = reqs.find(r => r.request_id === rid);
			if (!req) continue;
			for (const j of joins.sort((a, b) => a.rel_start_s - b.rel_start_s)) {
				const fwdTs = us(req.api_receive_ts + j.rel_start_s);
				const fwdDur = j.duration_ms * 1000;
				events.push({
					ph: 'X', pid, tid: 2,
					name: `fwd_${j.fwd_id}`,
					ts: fwdTs, dur: fwdDur,
					cat: 'forward_pass',
					args: { batch_size: j.batch_size, total_tokens: j.total_tokens, tokens_in_pass: j.tokens_in_pass },
				});
			}
		}

		const perTokenByReq = new Map<string, typeof $traceData.perToken>();
		for (const t of $traceData.perToken.filter(t => modelFilter.includes(t.model_label))) {
			if (!perTokenByReq.has(t.request_id)) perTokenByReq.set(t.request_id, []);
			perTokenByReq.get(t.request_id)!.push(t);
		}
		for (const [rid, tokens] of perTokenByReq) {
			const pid = reqIdToPid.get(rid);
			if (!pid) continue;
			const req = reqs.find(r => r.request_id === rid);
			if (!req) continue;
			for (const tok of tokens.sort((a, b) => a.token_idx - b.token_idx)) {
				const tokTs = us(tok.timestamp);
				const tokDur = Math.max(tok.itl_ms * 1000, 100);
				const args: Record<string, unknown> = { token_idx: tok.token_idx, itl_ms: tok.itl_ms };
				if (tok.kv_cache_size != null) args.kv_cache_size = tok.kv_cache_size;
				events.push({
					ph: 'X', pid, tid: 3,
					name: `tok_${tok.token_idx}`,
					ts: tokTs, dur: tokDur,
					cat: 'token',
					args,
				});
			}
		}

		return JSON.stringify({ traceEvents: events });
	}

	function downloadPerfetto() {
		const json = buildPerfettoJson();
		if (!json) return;
		const blob = new Blob([json], { type: 'application/json' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `vllm_traces_${$currentMode}.json`;
		a.click();
		URL.revokeObjectURL(url);
	}

	async function downloadAllTracesBundle() {
		const bundle = await loadAllTraceDataBundle();
		const payload = {
			exported_at: new Date().toISOString(),
			version: 'dashboard-v2',
			...bundle,
		};
		const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = 'vllm_traces_all_modes_bundle.json';
		a.click();
		URL.revokeObjectURL(url);
	}

	function handlePerfettoUpload(slot: 'A' | 'B', event: Event) {
		const input = event.target as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) return;
		const reader = new FileReader();
		reader.onload = () => {
			try {
				const data = JSON.parse(reader.result as string);
				const events = data.traceEvents || data;
				if (slot === 'A') uploadedPerfettoA = { name: file.name, events };
				else uploadedPerfettoB = { name: file.name, events };
				if ((slot === 'A' ? uploadedPerfettoB : uploadedPerfettoA)) runComparison();
			} catch { /* invalid json */ }
		};
		reader.readAsText(file);
	}

	function extractFwdDurations(events: Record<string, unknown>[]): Map<number, number> {
		const map = new Map<number, number>();
		for (const e of events) {
			if (e.cat === 'forward_pass' && e.ph === 'X') {
				const name = String(e.name || '');
				const match = name.match(/fwd_(\d+)/);
				if (match) {
					const fid = parseInt(match[1]);
					const dur = (e.dur as number) / 1000;
					if (!map.has(fid) || dur > (map.get(fid) || 0)) {
						map.set(fid, dur);
					}
				}
			}
		}
		return map;
	}

	function runComparison() {
		if (!uploadedPerfettoA || !uploadedPerfettoB) return;
		const durA = extractFwdDurations(uploadedPerfettoA.events);
		const durB = extractFwdDurations(uploadedPerfettoB.events);
		const allIds = new Set([...durA.keys(), ...durB.keys()]);
		const rows: typeof comparisonResult = [];
		for (const fid of [...allIds].sort((a, b) => a - b)) {
			const dA = durA.get(fid) ?? 0;
			const dB = durB.get(fid) ?? 0;
			const diff = Math.abs(dA - dB);
			rows.push({ fwdId: fid, durA: dA, durB: dB, diff, highlighted: diff > compThreshold });
		}
		comparisonResult = rows;
	}

	function buildComparisonChart() {
		if (!comparisonResult.length) return null;
		const fwdIds = comparisonResult.map(r => r.fwdId);
		return {
			backgroundColor: 'transparent',
			title: { text: 'Forward Pass Duration Comparison', textStyle: { color: '#E8EAF0', fontSize: 14, fontWeight: 500 }, left: 16, top: 12 },
			tooltip: {
				trigger: 'axis',
				backgroundColor: '#1C2030', borderColor: 'rgba(255,255,255,0.09)', textStyle: { color: '#E8EAF0' },
				formatter: (params: { seriesName: string; value: [number, number] }[]) => {
					const fid = params[0]?.value[0];
					const row = comparisonResult.find(r => r.fwdId === fid);
					if (!row) return '';
					return `<b>fwd_${fid}</b><br>A: ${row.durA.toFixed(1)} ms<br>B: ${row.durB.toFixed(1)} ms<br>Diff: ${row.diff.toFixed(1)} ms${row.highlighted ? ' ⚠️' : ''}`;
				}
			},
			legend: { bottom: 8, textStyle: { color: '#9BA1B0' } },
			grid: { left: 70, right: 30, top: 55, bottom: 50 },
			xAxis: { type: 'value', name: 'Forward Pass ID', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			yAxis: { type: 'value', name: 'Duration (ms)', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			visualMap: {
				show: false,
				dimension: 1,
				pieces: [{ min: 0, max: compThreshold, color: 'rgba(143,184,255,0.14)' }, { min: compThreshold, color: 'rgba(255,155,122,0.88)' }],
			},
			series: [
				{
					name: uploadedPerfettoA?.name || 'Trace A',
					type: 'line',
					data: comparisonResult.map(r => [r.fwdId, r.durA]),
					lineStyle: { width: 2.75, color: '#8FB8FF' },
					symbol: 'circle',
					showSymbol: false,
					itemStyle: { color: '#8FB8FF' },
				},
				{
					name: uploadedPerfettoB?.name || 'Trace B',
					type: 'line',
					data: comparisonResult.map(r => [r.fwdId, r.durB]),
					lineStyle: { width: 2.75, color: '#FF9B7A' },
					symbol: 'circle',
					showSymbol: false,
					itemStyle: { color: '#FF9B7A' },
				},
				{
					name: 'Difference',
					type: 'bar',
					data: comparisonResult.map(r => ({
						value: [r.fwdId, r.diff],
						itemStyle: { color: r.highlighted ? 'rgba(255,155,122,0.88)' : 'rgba(143,184,255,0.2)' },
					})),
					barWidth: 5,
				},
				{
					name: 'Threshold',
					type: 'line',
					data: [[fwdIds[0], compThreshold], [fwdIds[fwdIds.length - 1], compThreshold]],
					lineStyle: { type: 'dashed', color: '#F4CD78', width: 2 },
					symbol: 'none',
					itemStyle: { color: '#E2B866' },
				},
			],
		} as echarts.EChartsOption;
	}

	$effect(() => {
		compThreshold;
		if (uploadedPerfettoA && uploadedPerfettoB) runComparison();
	});
</script>

<div class="space-y-8" style="padding: 1.75rem 2rem;">
	<div>
		<h2 class="text-[1.25rem] font-bold" style="color: var(--text-primary)">Data & Export</h2>
		<p class="text-[0.8rem] mt-1" style="color: var(--text-muted)">Raw data tables, CSV download, Perfetto export, and trace comparison</p>
	</div>

	<!-- Model Filter -->
	<div class="flex items-center gap-3">
		<span class="text-sm" style="color: var(--text-secondary)">Filter:</span>
		{#each $meta.models as m}
			<button
				onclick={() => toggleModel(m)}
				class="px-3 py-1 rounded-full text-xs font-medium transition-all {modelFilter.includes(m) ? '' : 'hover:opacity-90'}"
				style={modelFilter.includes(m) ? 'background: var(--accent-subtle); color: var(--accent-primary); border: 1px solid rgba(124,147,219,0.18);' : 'color: var(--text-muted); border: 1px solid var(--border-default);'}
			>
				{m}
			</button>
		{/each}
	</div>

	<!-- Export Buttons -->
	<div class="flex gap-3 flex-wrap">
		<button onclick={() => downloadCsv(filteredReqs, 'request_traces.csv')} class="px-4 py-2 text-sm font-medium rounded-lg transition-colors hover:opacity-90" style="background: var(--accent-primary); color: white;">
			Download Requests CSV
		</button>
		<button onclick={() => downloadCsv(filteredFwd, 'fwd_traces.csv')} class="px-4 py-2 text-sm font-medium rounded-lg transition-colors hover:opacity-90" style="background: var(--accent-primary); color: white;">
			Download Fwd Passes CSV
		</button>
		<button onclick={downloadPerfetto} class="px-4 py-2 text-sm font-medium rounded-lg transition-colors hover:opacity-90" style="background: #9B7EC8; color: white;">
			Export Perfetto JSON (with tokens)
		</button>
		<button onclick={downloadAllTracesBundle} class="px-4 py-2 text-sm font-medium rounded-lg transition-colors hover:opacity-90" style="background: #2D9CC2; color: white;">
			Export ALL Traces Bundle
		</button>
	</div>

	<!-- Perfetto Comparison -->
	<div class="rounded-lg space-y-4 p-6" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
		<div>
			<h3 class="text-base font-semibold" style="color: var(--text-primary)">Perfetto Trace Comparison</h3>
			<p class="text-xs mt-1" style="color: var(--text-muted)">Upload two Perfetto JSON traces to compare forward pass durations. Highlighted bars indicate differences exceeding the threshold.</p>
		</div>

		<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
			<div>
				<label for="perfetto-a" class="block text-xs font-medium mb-1.5" style="color: var(--text-secondary)">Trace A {uploadedPerfettoA ? `(${uploadedPerfettoA.name})` : ''}</label>
				<input id="perfetto-a" type="file" accept=".json,.jsonl,application/json" onchange={(e) => handlePerfettoUpload('A', e)} class="block w-full text-sm" style="color: var(--text-secondary);" />
			</div>
			<div>
				<label for="perfetto-b" class="block text-xs font-medium mb-1.5" style="color: var(--text-secondary)">Trace B {uploadedPerfettoB ? `(${uploadedPerfettoB.name})` : ''}</label>
				<input id="perfetto-b" type="file" accept=".json,.jsonl,application/json" onchange={(e) => handlePerfettoUpload('B', e)} class="block w-full text-sm" style="color: var(--text-secondary);" />
			</div>
		</div>

		<div class="flex items-center gap-4">
			<label for="comp-threshold" class="text-sm" style="color: var(--text-secondary)">Highlight threshold:</label>
			<input id="comp-threshold" type="range" min="0.1" max="20" step="0.1" bind:value={compThreshold} class="w-48" />
			<span class="text-sm tabular-nums" style="color: var(--text-muted)">{compThreshold.toFixed(1)} ms</span>
		</div>

		{#if comparisonResult.length > 0}
			{@const chartOpts = buildComparisonChart()}
			{#if chartOpts}
				<Chart options={chartOpts} height="420px" />
			{/if}

			<div class="flex gap-4 text-sm">
				<span style="color: var(--text-secondary)">Total fwd passes: <span class="font-medium" style="color: var(--text-primary)">{comparisonResult.length}</span></span>
				<span style="color: var(--text-secondary)">Highlighted: <span class="font-medium" style="color: var(--color-danger)">{comparisonResult.filter(r => r.highlighted).length}</span></span>
				<span style="color: var(--text-secondary)">Within threshold: <span class="font-medium" style="color: var(--color-success)">{comparisonResult.filter(r => !r.highlighted).length}</span></span>
			</div>

			<div class="overflow-auto max-h-64">
				<table class="w-full text-sm">
					<thead class="sticky top-0" style="background: var(--surface-overlay);">
						<tr>
							<th class="px-3 py-2 text-left text-xs font-medium uppercase border-b" style="color: var(--text-secondary); border-color: var(--border-subtle);">Fwd ID</th>
							<th class="px-3 py-2 text-right text-xs font-medium uppercase border-b" style="color: var(--text-secondary); border-color: var(--border-subtle);">A (ms)</th>
							<th class="px-3 py-2 text-right text-xs font-medium uppercase border-b" style="color: var(--text-secondary); border-color: var(--border-subtle);">B (ms)</th>
							<th class="px-3 py-2 text-right text-xs font-medium uppercase border-b" style="color: var(--text-secondary); border-color: var(--border-subtle);">Diff (ms)</th>
							<th class="px-3 py-2 text-center text-xs font-medium uppercase border-b" style="color: var(--text-secondary); border-color: var(--border-subtle);">Status</th>
						</tr>
					</thead>
					<tbody>
						{#each comparisonResult as row, i}
							<tr class="data-export-table-row transition-colors" style="background: {row.highlighted ? 'rgba(212,129,107,0.08)' : i % 2 === 0 ? 'var(--surface-base)' : 'var(--surface-raised)'}">
								<td class="px-3 py-1.5 tabular-nums" style="color: var(--text-secondary)">{row.fwdId}</td>
								<td class="px-3 py-1.5 text-right tabular-nums" style="color: var(--text-secondary)">{row.durA.toFixed(1)}</td>
								<td class="px-3 py-1.5 text-right tabular-nums" style="color: var(--text-secondary)">{row.durB.toFixed(1)}</td>
								<td class="px-3 py-1.5 text-right tabular-nums {row.highlighted ? 'font-medium' : ''}" style="color: {row.highlighted ? 'var(--color-danger)' : 'var(--text-muted)'}">{row.diff.toFixed(1)}</td>
								<td class="px-3 py-1.5 text-center">{row.highlighted ? '⚠️' : '✓'}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	</div>

	<!-- Request Traces Table -->
	<div>
		<h3 class="text-sm font-semibold mb-3" style="color: var(--text-secondary)">Request Traces ({filteredReqs.length} rows)</h3>
		<DataTable columns={reqColumns} rows={filteredReqs} maxHeight="400px" />
	</div>

	<!-- Forward Pass Traces Table -->
	<div>
		<h3 class="text-sm font-semibold mb-3" style="color: var(--text-secondary)">Forward Pass Traces ({filteredFwd.length} rows)</h3>
		<DataTable columns={fwdColumns} rows={filteredFwd} maxHeight="400px" />
	</div>
</div>

<style>
	.data-export-table-row:hover {
		background: var(--surface-hover) !important;
	}
	input[type="file"]::file-selector-button {
		background: var(--accent-primary);
		color: white;
		margin-right: 0.75rem;
		padding: 0.375rem 0.75rem;
		border-radius: 0.5rem;
		border: 0;
		font-size: 0.875rem;
		font-weight: 500;
		cursor: pointer;
	}
	input[type="file"]::file-selector-button:hover {
		background: var(--accent-primary-hover);
	}
</style>
