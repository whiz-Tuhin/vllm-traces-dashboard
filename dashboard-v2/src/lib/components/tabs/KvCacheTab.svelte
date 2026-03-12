<script lang="ts">
	import { traceData, meta } from '$lib/stores';
	import Chart from '../Chart.svelte';
	import MetricCard from '../MetricCard.svelte';
	import type { KvCacheRow } from '$lib/types';

	let convIds = $derived(() => {
		const ids = new Set<string>();
		for (const r of $traceData.requests) {
			const match = r.prompt_id.match(/^(conv_\d+)/);
			if (match) ids.add(match[1]);
		}
		return [...ids].sort();
	});

	let selectedConv = $state('');

	$effect(() => {
		const ids = convIds();
		if (ids.length && !selectedConv) {
			selectedConv = ids[Math.min(2, ids.length - 1)];
		}
	});

	function getConvRequests(conv: string) {
		return $traceData.requests.filter(r => r.prompt_id.startsWith(conv + '_')).sort((a, b) => a.prompt_id.localeCompare(b.prompt_id));
	}

	function getConvKv(conv: string) {
		const reqIds = new Set(getConvRequests(conv).map(r => r.request_id));
		return $traceData.kvCache.filter(k => reqIds.has(k.request_id)).sort((a, b) => a.fwd_id - b.fwd_id);
	}

	function fmtDelta(v: number | null, decimals = 1): string {
		if (v == null || Number.isNaN(v)) return '—';
		if (Math.abs(v) < 0.0001) return '0';
		const sign = v > 0 ? '+' : '';
		return `${sign}${v.toFixed(decimals)}`;
	}

	function parseTurnIndex(promptId: string): number {
		const m = promptId.match(/turn(\d+)/);
		return m ? Number.parseInt(m[1], 10) : 0;
	}

	function getTurnReqRows() {
		const rows: {
			model: string;
			turnLabel: string;
			turnIndex: number;
			schedulingMs: number;
			prefillMs: number;
			decodeMs: number;
			otherMs: number;
			totalMs: number;
			ttftMs: number;
			outputTokens: number;
			decodeThroughput: number;
		}[] = [];
		for (const label of $meta.models) {
			const reqs = getConvRequests(selectedConv)
				.filter(r => r.model_label === label)
				.sort((a, b) => parseTurnIndex(a.prompt_id) - parseTurnIndex(b.prompt_id));
			for (const r of reqs) {
				const turnIndex = parseTurnIndex(r.prompt_id);
				const turnLabel = `T${String(turnIndex).padStart(2, '0')}`;
				const schedulingMs = r.scheduling_overhead_ms;
				const prefillMs = Math.max(r.ttft_ms - r.scheduling_overhead_ms, 0);
				const decodeMs = Math.max(r.decode_ms, 0);
				const totalMs = r.total_latency_ms;
				const otherMs = Math.max(totalMs - schedulingMs - prefillMs - decodeMs, 0);
				const decodeThroughput = decodeMs > 0 ? (r.output_tokens / (decodeMs / 1000)) : 0;
				rows.push({
					model: label,
					turnLabel,
					turnIndex,
					schedulingMs,
					prefillMs,
					decodeMs,
					otherMs,
					totalMs,
					ttftMs: r.ttft_ms,
					outputTokens: r.output_tokens,
					decodeThroughput,
				});
			}
		}
		return rows;
	}

	function buildTurnWaterfall() {
		const rows = getTurnReqRows();
		const turnLabels = [...new Set(rows.map(r => r.turnLabel))].sort((a, b) => Number.parseInt(a.slice(1)) - Number.parseInt(b.slice(1)));
		const series: echarts.EChartsOption['series'] = [];
		for (const label of $meta.models) {
			const byTurn = new Map(rows.filter(r => r.model === label).map(r => [r.turnLabel, r]));
			const buildSeries = (name: string, key: keyof typeof rows[number], colorKey: 'Scheduling' | 'Prefill' | 'Decode' | 'Queue/Other') => ({
				name: `${label} ${name}`,
				type: 'bar',
				stack: label,
				data: turnLabels.map(t => byTurn.get(t)?.[key] ?? 0),
				itemStyle: { color: $meta.phase_colors[colorKey] },
			});
			series.push(buildSeries('Scheduling', 'schedulingMs', 'Scheduling'));
			series.push(buildSeries('Prefill', 'prefillMs', 'Prefill'));
			series.push(buildSeries('Decode', 'decodeMs', 'Decode'));
			series.push(buildSeries('Queue/Other', 'otherMs', 'Queue/Other'));
		}
		return {
			backgroundColor: 'transparent',
			title: { text: 'Turn Latency Waterfall (Scheduling + Prefill + Decode + Other)', left: 16, top: 12 },
			tooltip: { trigger: 'axis' },
			legend: { bottom: 6, type: 'scroll' },
			grid: { left: 74, right: 26, top: 58, bottom: 62 },
			xAxis: { type: 'category', data: turnLabels, name: 'Turn' },
			yAxis: { type: 'value', name: 'Latency (ms)' },
			series,
		} as echarts.EChartsOption;
	}

	function buildTurnKvStep() {
		const series: echarts.EChartsOption['series'] = [];
		const turnLabels = [...new Set(getConvRequests(selectedConv).map(r => `T${String(parseTurnIndex(r.prompt_id)).padStart(2, '0')}`))]
			.sort((a, b) => Number.parseInt(a.slice(1)) - Number.parseInt(b.slice(1)));
		for (const label of $meta.models) {
			const reqs = getConvRequests(selectedConv)
				.filter(r => r.model_label === label)
				.sort((a, b) => parseTurnIndex(a.prompt_id) - parseTurnIndex(b.prompt_id));
			const starts: number[] = [];
			const ends: number[] = [];
			for (const req of reqs) {
				const kvRows = $traceData.kvCache.filter(k => k.model_label === label && k.request_id === req.request_id);
				const kvStart = kvRows.length ? Math.min(...kvRows.map(k => k.past_kv_cache_size)) : 0;
				const kvEnd = kvRows.length ? Math.max(...kvRows.map(k => k.past_kv_cache_size + k.num_scheduled_tokens)) : 0;
				starts.push(kvStart);
				ends.push(kvEnd);
			}
			series.push({
				name: `${label} KV Start`,
				type: 'line',
				step: 'end',
				data: turnLabels.map((_, i) => starts[i] ?? 0),
				lineStyle: { type: 'dashed', color: $meta.colors[label] },
				itemStyle: { color: $meta.colors[label] },
			});
			series.push({
				name: `${label} KV End`,
				type: 'line',
				step: 'end',
				data: turnLabels.map((_, i) => ends[i] ?? 0),
				lineStyle: { width: 2.8, color: $meta.colors[label] },
				itemStyle: { color: $meta.colors[label] },
				areaStyle: { color: $meta.colors[label], opacity: 0.1 },
			});
		}
		return {
			backgroundColor: 'transparent',
			title: { text: 'KV Cache Start/End per Turn', left: 16, top: 12 },
			tooltip: { trigger: 'axis' },
			legend: { bottom: 8, type: 'scroll' },
			grid: { left: 74, right: 26, top: 58, bottom: 52 },
			xAxis: { type: 'category', data: turnLabels, name: 'Turn' },
			yAxis: { type: 'value', name: 'KV Tokens' },
			series,
		} as echarts.EChartsOption;
	}

	function buildDecodeThroughputByTurn() {
		const rows = getTurnReqRows();
		const turnLabels = [...new Set(rows.map(r => r.turnLabel))].sort((a, b) => Number.parseInt(a.slice(1)) - Number.parseInt(b.slice(1)));
		const series: echarts.EChartsOption['series'] = [];
		for (const label of $meta.models) {
			const byTurn = new Map(rows.filter(r => r.model === label).map(r => [r.turnLabel, r]));
			series.push({
				name: `${label} decode tok/s`,
				type: 'line',
				data: turnLabels.map(t => Number((byTurn.get(t)?.decodeThroughput ?? 0).toFixed(1))),
				lineStyle: { width: 2.8, color: $meta.colors[label] },
				itemStyle: { color: $meta.colors[label] },
				areaStyle: { color: $meta.colors[label], opacity: 0.08 },
			});
		}
		return {
			backgroundColor: 'transparent',
			title: { text: 'Decode Throughput by Turn', left: 16, top: 12 },
			tooltip: { trigger: 'axis' },
			legend: { bottom: 8, type: 'scroll' },
			grid: { left: 74, right: 26, top: 58, bottom: 52 },
			xAxis: { type: 'category', data: turnLabels, name: 'Turn' },
			yAxis: { type: 'value', name: 'Decode Tokens / s' },
			series,
		} as echarts.EChartsOption;
	}

	function buildKvGrowth() {
		const series: echarts.EChartsOption['series'] = [];
		const markLines: { xAxis: number; label: { formatter: string } }[] = [];

		for (const label of $meta.models) {
			const kvRows = getConvKv(selectedConv).filter(k => k.model_label === label);
			if (!kvRows.length) continue;

			const byReq = new Map<string, KvCacheRow[]>();
			for (const k of kvRows) {
				if (!byReq.has(k.request_id)) byReq.set(k.request_id, []);
				byReq.get(k.request_id)!.push(k);
			}

			for (const [rid, rows] of byReq) {
				const turnMatch = rid.match(/turn(\d+)/);
				const turnLabel = turnMatch ? `T${turnMatch[1]}` : rid.slice(-8);
				series.push({
					name: `${label} ${turnLabel}`,
					type: 'line',
					data: rows.map(r => [r.fwd_id, r.past_kv_cache_size]),
					lineStyle: { width: 2, color: $meta.colors[label] },
					symbol: 'none',
					itemStyle: { color: $meta.colors[label] },
				});

				const prefillRows = rows.filter(r => r.is_prefill);
				for (const pr of prefillRows) {
					markLines.push({ xAxis: pr.fwd_id, label: { formatter: turnLabel } });
				}
			}
		}

		return {
			backgroundColor: 'transparent',
			title: { text: 'KV Cache Size Over Forward Passes', textStyle: { color: '#E8EAF0', fontSize: 14, fontWeight: 500 }, left: 16, top: 12 },
			tooltip: { trigger: 'axis', backgroundColor: '#1C2030', borderColor: 'rgba(255,255,255,0.09)', textStyle: { color: '#E8EAF0' } },
			legend: { bottom: 8, textStyle: { color: '#9BA1B0' }, type: 'scroll' },
			grid: { left: 80, right: 30, top: 55, bottom: 50 },
			xAxis: { type: 'value', name: 'Forward Pass ID', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			yAxis: { type: 'value', name: 'KV Cache Size (tokens)', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			series: [...series, {
				type: 'line',
				markLine: {
					silent: true,
					symbol: 'none',
					lineStyle: { type: 'dashed', color: '#E2B866', width: 1 },
					data: markLines,
					label: { color: '#E2B866', fontSize: 10 },
				},
				data: [],
			}],
		} as echarts.EChartsOption;
	}

	function buildPrefixDecode() {
		const series: echarts.EChartsOption['series'] = [];
		for (const label of $meta.models) {
			const kvRows = getConvKv(selectedConv).filter(k => k.model_label === label);
			const fwdIds = [...new Set(kvRows.map(k => k.fwd_id))].sort((a, b) => a - b);

			const prefillCountByFwd = new Map<number, number>();
			const decodeCountByFwd = new Map<number, number>();
			for (const k of kvRows) {
				if (k.is_prefill) {
					prefillCountByFwd.set(k.fwd_id, (prefillCountByFwd.get(k.fwd_id) || 0) + 1);
				} else {
					decodeCountByFwd.set(k.fwd_id, (decodeCountByFwd.get(k.fwd_id) || 0) + 1);
				}
			}

			series.push({
				name: `${label} Prefill Reqs`,
				type: 'bar',
				stack: label,
				data: fwdIds.map(id => [id, prefillCountByFwd.get(id) || 0]),
				itemStyle: { color: $meta.phase_colors['Prefill'] },
			});
			series.push({
				name: `${label} Decode Reqs`,
				type: 'bar',
				stack: label,
				data: fwdIds.map(id => [id, decodeCountByFwd.get(id) || 0]),
				itemStyle: { color: $meta.phase_colors['Decode'] },
			});
		}
		return {
			backgroundColor: 'transparent',
			title: { text: 'Prefill vs Decode Requests per Forward Pass', textStyle: { color: '#E8EAF0', fontSize: 14, fontWeight: 500 }, left: 16, top: 12 },
			tooltip: { trigger: 'axis', backgroundColor: '#1C2030', borderColor: 'rgba(255,255,255,0.09)', textStyle: { color: '#E8EAF0' } },
			legend: { bottom: 8, textStyle: { color: '#9BA1B0' } },
			grid: { left: 70, right: 30, top: 55, bottom: 50 },
			xAxis: { type: 'value', name: 'Forward Pass ID', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			yAxis: { type: 'value', name: 'Requests in Batch', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 5, height: 20, borderColor: 'rgba(255,255,255,0.06)' }],
			series,
		} as echarts.EChartsOption;
	}

	function buildTokenBreakdown() {
		const series: echarts.EChartsOption['series'] = [];
		for (const label of $meta.models) {
			const kvRows = getConvKv(selectedConv).filter(k => k.model_label === label);
			const byReq = new Map<string, KvCacheRow[]>();
			for (const k of kvRows) {
				if (!byReq.has(k.request_id)) byReq.set(k.request_id, []);
				byReq.get(k.request_id)!.push(k);
			}

			for (const [rid, rows] of byReq) {
				const turnMatch = rid.match(/turn(\d+)/);
				const turnLabel = turnMatch ? `T${turnMatch[1]}` : rid.slice(-8);
				const sorted = rows.sort((a, b) => a.fwd_id - b.fwd_id);
				const prefillRow = sorted.find(r => r.is_prefill);
				const decodeRows = sorted.filter(r => !r.is_prefill);

				series.push({
					name: `${label} ${turnLabel} Decode`,
					type: 'line',
					data: decodeRows.map(r => [r.fwd_id, r.decode_tokens]),
					lineStyle: { width: 2, color: $meta.colors[label] },
					symbol: 'none',
					itemStyle: { color: $meta.colors[label] },
					markPoint: prefillRow ? {
						data: [{ coord: [prefillRow.fwd_id, prefillRow.prefix_tokens], value: `${turnLabel} prefill: ${prefillRow.prefix_tokens} tokens`, symbol: 'diamond', symbolSize: 12, itemStyle: { color: $meta.phase_colors['Prefill'] } }],
						label: { show: false },
					} : undefined,
				});
			}
		}
		return {
			backgroundColor: 'transparent',
			title: { text: 'Tokens Scheduled per Forward Pass (by Turn)', textStyle: { color: '#E8EAF0', fontSize: 14, fontWeight: 500 }, left: 16, top: 12 },
			tooltip: { trigger: 'axis', backgroundColor: '#1C2030', borderColor: 'rgba(255,255,255,0.09)', textStyle: { color: '#E8EAF0' } },
			legend: { bottom: 8, textStyle: { color: '#9BA1B0' }, type: 'scroll' },
			grid: { left: 70, right: 30, top: 55, bottom: 50 },
			xAxis: { type: 'value', name: 'Forward Pass ID', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			yAxis: { type: 'value', name: 'Tokens Scheduled', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 5, height: 20, borderColor: 'rgba(255,255,255,0.06)' }],
			series,
		} as echarts.EChartsOption;
	}

	function buildTokensGenerated() {
		const series: echarts.EChartsOption['series'] = [];
		for (const label of $meta.models) {
			const kvRows = getConvKv(selectedConv).filter(k => k.model_label === label);
			const byReq = new Map<string, KvCacheRow[]>();
			for (const k of kvRows) {
				if (!byReq.has(k.request_id)) byReq.set(k.request_id, []);
				byReq.get(k.request_id)!.push(k);
			}
			for (const [rid, rows] of byReq) {
				const turnMatch = rid.match(/turn(\d+)/);
				const turnLabel = turnMatch ? `T${turnMatch[1]}` : rid.slice(-8);
				series.push({
					name: `${label} ${turnLabel}`,
					type: 'line',
					data: rows.map(r => [r.fwd_id, r.tokens_generated_so_far]),
					lineStyle: { width: 2, color: $meta.colors[label] },
					symbol: 'none',
					itemStyle: { color: $meta.colors[label] },
				});
			}
		}
		return {
			backgroundColor: 'transparent',
			title: { text: 'Tokens Generated So Far', textStyle: { color: '#E8EAF0', fontSize: 14, fontWeight: 500 }, left: 16, top: 12 },
			tooltip: { trigger: 'axis', backgroundColor: '#1C2030', borderColor: 'rgba(255,255,255,0.09)', textStyle: { color: '#E8EAF0' } },
			legend: { bottom: 8, textStyle: { color: '#9BA1B0' }, type: 'scroll' },
			grid: { left: 70, right: 30, top: 55, bottom: 50 },
			xAxis: { type: 'value', name: 'Forward Pass ID', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			yAxis: { type: 'value', name: 'Tokens Generated', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			series,
		} as echarts.EChartsOption;
	}

	let turnSummary = $derived(() => {
		const rows: {
			label: string;
			turn: string;
			turnIndex: number;
			promptTokens: number;
			outputTokens: number;
			kvStart: number;
			kvEnd: number;
			prefillFwds: number;
			decodeFwds: number;
			ttftMs: number;
			e2eMs: number;
			dTtftMs: number | null;
			dE2eMs: number | null;
			dKvEnd: number | null;
		}[] = [];
		for (const label of $meta.models) {
			const reqs = getConvRequests(selectedConv).filter(r => r.model_label === label);
			let prev: { ttftMs: number; e2eMs: number; kvEnd: number } | null = null;
			for (const req of reqs) {
				const kvRows = $traceData.kvCache.filter(k => k.request_id === req.request_id && k.model_label === label);
				const kvStart = kvRows.length ? Math.min(...kvRows.map(k => k.past_kv_cache_size)) : 0;
				const kvEnd = kvRows.length ? Math.max(...kvRows.map(k => k.past_kv_cache_size + k.num_scheduled_tokens)) : 0;
				const prefillFwds = kvRows.filter(k => k.is_prefill).length;
				const decodeFwds = kvRows.filter(k => !k.is_prefill).length;
				const turnIndex = parseTurnIndex(req.prompt_id);
				const ttftMs = req.ttft_ms;
				const e2eMs = req.total_latency_ms;
				const dTtftMs = prev ? ttftMs - prev.ttftMs : null;
				const dE2eMs = prev ? e2eMs - prev.e2eMs : null;
				const dKvEnd = prev ? kvEnd - prev.kvEnd : null;
				rows.push({
					label,
					turn: req.prompt_id,
					turnIndex,
					promptTokens: req.prompt_tokens,
					outputTokens: req.output_tokens,
					kvStart,
					kvEnd,
					prefillFwds,
					decodeFwds,
					ttftMs,
					e2eMs,
					dTtftMs,
					dE2eMs,
					dKvEnd,
				});
				prev = { ttftMs, e2eMs, kvEnd };
			}
		}
		return rows.sort((a, b) => {
			if (a.label !== b.label) return a.label.localeCompare(b.label);
			return a.turnIndex - b.turnIndex;
		});
	});
</script>

<div class="space-y-8" style="padding: 1.75rem 2rem;">
	<div class="flex items-center justify-between">
		<div>
			<h2 class="text-[1.25rem] font-bold" style="color: var(--text-primary)">KV Cache & Turns</h2>
			<p class="text-[0.8rem] mt-1" style="color: var(--text-muted)">Multi-turn conversation analysis with KV cache growth</p>
		</div>
		<select bind:value={selectedConv} class="rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500" style="background: var(--surface-raised); border: 1px solid var(--border-default); color: var(--text-primary);">
			{#each convIds() as cid}
				<option value={cid}>{cid}</option>
			{/each}
		</select>
	</div>

	<!-- Summary -->
	<div class="grid grid-cols-2 md:grid-cols-4 gap-3">
		<MetricCard label="Conversation" value={selectedConv} />
		<MetricCard label="Turns" value={String(getConvRequests(selectedConv).length / $meta.models.length)} />
		<MetricCard label="KV Entries" value={String(getConvKv(selectedConv).length)} />
		<MetricCard label="Models" value={String($meta.models.length)} />
	</div>

	<div class="rounded-lg p-3" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
		<Chart options={buildKvGrowth()} height="420px" />
	</div>

	<div class="rounded-lg p-3" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
		<Chart options={buildTurnWaterfall()} height="430px" />
	</div>

	<div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
		<div class="rounded-lg p-3" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
			<Chart options={buildTurnKvStep()} height="380px" />
		</div>
		<div class="rounded-lg p-3" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
			<Chart options={buildDecodeThroughputByTurn()} height="380px" />
		</div>
	</div>

	<div class="rounded-lg p-3" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
		<Chart options={buildPrefixDecode()} height="400px" />
	</div>

	<div class="rounded-lg p-3" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
		<Chart options={buildTokenBreakdown()} height="400px" />
	</div>

	<div class="rounded-lg p-3" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
		<Chart options={buildTokensGenerated()} height="400px" />
	</div>

	<!-- Per-Turn Summary Table -->
	<div class="rounded-xl overflow-hidden" style="border: 1px solid var(--border-subtle);">
		<div class="px-4 py-3 border-b" style="border-color: var(--border-subtle);">
			<h3 class="text-sm font-semibold" style="color: var(--text-secondary)">Per-Turn Summary</h3>
		</div>
		<div class="overflow-auto max-h-96">
			<table class="w-full text-sm">
				<thead class="sticky top-0" style="background: var(--surface-overlay);">
					<tr>
						<th class="px-3 py-2 text-left text-xs font-medium uppercase border-b" style="color: var(--text-muted); border-color: var(--border-subtle);">Model</th>
						<th class="px-3 py-2 text-left text-xs font-medium uppercase border-b" style="color: var(--text-muted); border-color: var(--border-subtle);">Turn</th>
						<th class="px-3 py-2 text-right text-xs font-medium uppercase border-b" style="color: var(--text-muted); border-color: var(--border-subtle);">Prompt Tok</th>
						<th class="px-3 py-2 text-right text-xs font-medium uppercase border-b" style="color: var(--text-muted); border-color: var(--border-subtle);">Output Tok</th>
						<th class="px-3 py-2 text-right text-xs font-medium uppercase border-b" style="color: var(--text-muted); border-color: var(--border-subtle);">KV Start</th>
						<th class="px-3 py-2 text-right text-xs font-medium uppercase border-b" style="color: var(--text-muted); border-color: var(--border-subtle);">KV End</th>
						<th class="px-3 py-2 text-right text-xs font-medium uppercase border-b" style="color: {$meta.phase_colors['Prefill']}; border-color: var(--border-subtle);">Prefill Fwds</th>
						<th class="px-3 py-2 text-right text-xs font-medium uppercase border-b" style="color: {$meta.phase_colors['Decode']}; border-color: var(--border-subtle);">Decode Fwds</th>
						<th class="px-3 py-2 text-right text-xs font-medium uppercase border-b" style="color: var(--text-muted); border-color: var(--border-subtle);">TTFT (ms)</th>
						<th class="px-3 py-2 text-right text-xs font-medium uppercase border-b" style="color: var(--text-muted); border-color: var(--border-subtle);">E2E (ms)</th>
						<th class="px-3 py-2 text-right text-xs font-medium uppercase border-b" style="color: var(--text-muted); border-color: var(--border-subtle);">ΔTTFT</th>
						<th class="px-3 py-2 text-right text-xs font-medium uppercase border-b" style="color: var(--text-muted); border-color: var(--border-subtle);">ΔE2E</th>
						<th class="px-3 py-2 text-right text-xs font-medium uppercase border-b" style="color: var(--text-muted); border-color: var(--border-subtle);">ΔKV End</th>
					</tr>
				</thead>
				<tbody>
					{#each turnSummary() as row, i}
						<tr class="data-row transition-colors" style="background: {i % 2 === 0 ? 'var(--surface-base)' : 'var(--surface-raised)'};">
							<td class="px-3 py-1.5 font-medium" style="color: {$meta.colors[row.label]}">{row.label}</td>
							<td class="px-3 py-1.5 tabular-nums" style="color: var(--text-secondary);">{row.turn}</td>
							<td class="px-3 py-1.5 text-right tabular-nums" style="color: var(--text-secondary);">{row.promptTokens}</td>
							<td class="px-3 py-1.5 text-right tabular-nums" style="color: var(--text-secondary);">{row.outputTokens}</td>
							<td class="px-3 py-1.5 text-right tabular-nums" style="color: var(--text-secondary);">{row.kvStart}</td>
							<td class="px-3 py-1.5 text-right tabular-nums" style="color: var(--text-secondary);">{row.kvEnd}</td>
							<td class="px-3 py-1.5 text-right tabular-nums" style="color: {$meta.phase_colors['Prefill']}">{row.prefillFwds}</td>
							<td class="px-3 py-1.5 text-right tabular-nums" style="color: {$meta.phase_colors['Decode']}">{row.decodeFwds}</td>
							<td class="px-3 py-1.5 text-right tabular-nums" style="color: var(--text-secondary);">{row.ttftMs.toFixed(0)}</td>
							<td class="px-3 py-1.5 text-right tabular-nums" style="color: var(--text-secondary);">{row.e2eMs.toFixed(0)}</td>
							<td class="px-3 py-1.5 text-right tabular-nums font-medium" style="color: {row.dTtftMs == null ? 'var(--text-muted)' : row.dTtftMs > 0 ? 'var(--color-danger)' : row.dTtftMs < 0 ? 'var(--color-success)' : 'var(--text-secondary)'}">{fmtDelta(row.dTtftMs, 1)}</td>
							<td class="px-3 py-1.5 text-right tabular-nums font-medium" style="color: {row.dE2eMs == null ? 'var(--text-muted)' : row.dE2eMs > 0 ? 'var(--color-danger)' : row.dE2eMs < 0 ? 'var(--color-success)' : 'var(--text-secondary)'}">{fmtDelta(row.dE2eMs, 1)}</td>
							<td class="px-3 py-1.5 text-right tabular-nums font-medium" style="color: {row.dKvEnd == null ? 'var(--text-muted)' : row.dKvEnd > 0 ? 'var(--color-danger)' : row.dKvEnd < 0 ? 'var(--color-success)' : 'var(--text-secondary)'}">{fmtDelta(row.dKvEnd, 0)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	</div>
</div>

<style>
	tr.data-row:hover {
		background: var(--surface-hover) !important;
	}
</style>
