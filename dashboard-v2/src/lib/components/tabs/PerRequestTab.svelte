<script lang="ts">
	import { traceData, meta } from '$lib/stores';
	import { median, percentile } from '$lib/stats';
	import Chart from '../Chart.svelte';
	import MetricCard from '../MetricCard.svelte';

	let promptIds = $derived([...new Set($traceData.requests.map(r => r.prompt_id))].sort());
	let selectedPrompt = $state('');

	$effect(() => {
		if (promptIds.length && !selectedPrompt) {
			selectedPrompt = promptIds[Math.min(10, promptIds.length - 1)];
		}
	});

	function buildGantt() {
		const series: unknown[] = [];
		const yLabels: string[] = [];
		let yIdx = 0;

		for (const label of $meta.models) {
			const req = $traceData.requests.find(r => r.model_label === label && r.prompt_id === selectedPrompt);
			if (!req) continue;

			const phases = [
				{ name: 'Scheduling', start: req.rel_api_receive_ts, end: req.rel_engine_add_request_ts, color: $meta.phase_colors['Scheduling'] },
				{ name: 'Prefill', start: req.rel_engine_add_request_ts, end: req.rel_first_token_ts, color: $meta.phase_colors['Prefill'] },
				{ name: 'Decode', start: req.rel_first_token_ts, end: req.rel_completion_ts, color: $meta.phase_colors['Decode'] },
			];

			yLabels.push(label);
			for (const ph of phases) {
				series.push({
					type: 'custom',
					name: ph.name,
					renderItem: (_params: unknown, api: { value: (i: number) => number; coord: (v: number[]) => number[]; size: (v: number[]) => number[] }) => {
						const start = api.coord([api.value(0), yIdx]);
						const end = api.coord([api.value(1), yIdx]);
						const h = api.size([0, 1])[1] * 0.6;
						return {
							type: 'rect',
							shape: { x: start[0], y: start[1] - h / 2, width: end[0] - start[0], height: h },
							style: { fill: ph.color },
						};
					},
					encode: { x: [0, 1], y: 2 },
					data: [[ph.start, ph.end, yIdx]],
					tooltip: { formatter: () => `<b>${ph.name}</b><br>${((ph.end - ph.start) * 1000).toFixed(0)} ms` },
				});
			}

			const fwds = $traceData.join.filter(j => j.model_label === label && j.prompt_id === selectedPrompt).sort((a, b) => a.rel_start_s - b.rel_start_s);
			if (fwds.length) {
				series.push({
					type: 'scatter',
					data: fwds.map(f => [f.rel_start_s, yIdx]),
					symbol: 'rect',
					symbolSize: [2, 20],
					itemStyle: { color: 'rgba(255,255,255,0.6)' },
					tooltip: {
						formatter: (p: { dataIndex: number }) => {
							const f = fwds[p.dataIndex];
							return `<b>Fwd pass</b><br>Duration: ${f.duration_ms.toFixed(1)} ms<br>Batch: ${f.batch_size}<br>fwd_id: ${f.fwd_id}`;
						}
					},
				});
			}
			yIdx++;
		}

		return {
			backgroundColor: 'transparent',
			title: { text: `Request Lifecycle — ${selectedPrompt}`, textStyle: { color: '#E8EAF0', fontSize: 14, fontWeight: 500 }, left: 16, top: 12 },
			tooltip: { backgroundColor: '#1C2030', borderColor: 'rgba(255,255,255,0.09)', textStyle: { color: '#E8EAF0' } },
			grid: { left: 100, right: 30, top: 55, bottom: 35 },
			xAxis: { type: 'value', name: 'Relative Time (s)', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			yAxis: { type: 'category', data: yLabels, axisLabel: { color: '#9BA1B0' } },
			series,
		} as echarts.EChartsOption;
	}

	function buildFwdScatter() {
		const series: echarts.EChartsOption['series'] = [];
		for (const label of $meta.models) {
			const fwds = $traceData.join.filter(j => j.model_label === label && j.prompt_id === selectedPrompt).sort((a, b) => a.rel_start_s - b.rel_start_s);
			if (!fwds.length) continue;
			const firstFwd = fwds[0].fwd_id;
			series.push({
				name: label,
				type: 'scatter',
				data: fwds.map(f => [f.rel_start_s, f.duration_ms, f.fwd_id === firstFwd ? 1 : 0, f.batch_size, f.fwd_id]),
				symbolSize: (val: number[]) => val[2] ? 14 : 7,
				symbol: (val: number[]) => val[2] ? 'diamond' : 'circle',
				itemStyle: { color: $meta.colors[label] },
				tooltip: {
					formatter: (p: { value: number[] }) => {
						const [t, dur, isPf, bs, fid] = p.value;
						return `<b>${label}</b><br>Time: ${t.toFixed(3)}s<br>Duration: ${dur.toFixed(1)} ms<br>Batch: ${bs}<br>fwd_id: ${fid}<br>${isPf ? '★ Prefill' : '● Decode'}`;
					}
				},
			});
		}
		return {
			backgroundColor: 'transparent',
			title: { text: `Forward Passes — ◆ prefill, ● decode`, textStyle: { color: '#E8EAF0', fontSize: 14, fontWeight: 500 }, left: 16, top: 12 },
			tooltip: { backgroundColor: '#1C2030', borderColor: 'rgba(255,255,255,0.09)', textStyle: { color: '#E8EAF0' } },
			legend: { bottom: 8, textStyle: { color: '#9BA1B0' } },
			grid: { left: 70, right: 30, top: 55, bottom: 45 },
			xAxis: { type: 'value', name: 'Relative Time (s)', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			yAxis: { type: 'log', name: 'Duration (ms)', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			series,
		} as echarts.EChartsOption;
	}

	function buildItlChart() {
		const series: echarts.EChartsOption['series'] = [];
		let hasKvData = false;
		for (const label of $meta.models) {
			const req = $traceData.requests.find(r => r.model_label === label && r.prompt_id === selectedPrompt);
			if (!req) continue;
			const tokens = $traceData.perToken.filter(t => t.model_label === label && t.request_id === req.request_id).sort((a, b) => a.token_idx - b.token_idx);
			if (!tokens.length) continue;

			series.push({
				name: `${label} ITL`,
				type: 'line',
				data: tokens.map(t => [t.token_idx, t.itl_ms]),
				lineStyle: { width: 1.5, color: $meta.colors[label] },
				symbol: 'none',
				itemStyle: { color: $meta.colors[label] },
				areaStyle: { color: $meta.colors[label], opacity: 0.08 },
				yAxisIndex: 0,
			});

			if (tokens.some(t => t.kv_cache_size != null)) {
				hasKvData = true;
				series.push({
					name: `${label} KV Cache`,
					type: 'line',
					data: tokens.filter(t => t.kv_cache_size != null).map(t => [t.token_idx, t.kv_cache_size]),
					lineStyle: { width: 2, color: $meta.colors[label], type: 'dashed' },
					symbol: 'none',
					itemStyle: { color: $meta.colors[label] },
					yAxisIndex: 1,
				});
			}
		}

		const yAxes: echarts.EChartsOption['yAxis'] = [
			{ type: 'value', name: 'ITL (ms)', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
		];
		if (hasKvData) {
			(yAxes as unknown[]).push({
				type: 'value', name: 'KV Cache Size (tokens)', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { show: false }, position: 'right',
			});
		}

		return {
			backgroundColor: 'transparent',
			title: { text: hasKvData ? 'Inter-Token Latency & KV Cache Size' : 'Inter-Token Latency per Token', textStyle: { color: '#E8EAF0', fontSize: 14, fontWeight: 500 }, left: 16, top: 12 },
			tooltip: { trigger: 'axis', backgroundColor: '#1C2030', borderColor: 'rgba(255,255,255,0.09)', textStyle: { color: '#E8EAF0' } },
			legend: { bottom: 8, textStyle: { color: '#9BA1B0' } },
			grid: { left: 60, right: hasKvData ? 80 : 30, top: 55, bottom: 45 },
			xAxis: { type: 'value', name: 'Token Index', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			yAxis: yAxes,
			dataZoom: [{ type: 'inside' }],
			series,
		} as echarts.EChartsOption;
	}

	let reqMetrics = $derived($meta.models.map(label => {
		const req = $traceData.requests.find(r => r.model_label === label && r.prompt_id === selectedPrompt);
		if (!req) return null;
		const tokens = $traceData.perToken.filter(t => t.model_label === label && t.request_id === req.request_id && t.token_idx > 0);
		const itls = tokens.map(t => t.itl_ms);
		return {
			label,
			promptTokens: req.prompt_tokens,
			outputTokens: req.output_tokens,
			ttft: req.ttft_ms.toFixed(1),
			tpot: req.tpot_ms.toFixed(2),
			e2e: req.total_latency_ms.toFixed(0),
			medianItl: itls.length ? median(itls).toFixed(1) : '–',
			p95Itl: itls.length ? percentile(itls, 95).toFixed(1) : '–',
		};
	}).filter(Boolean));
</script>

<div class="px-8 py-7 space-y-8">
	<div class="flex items-center justify-between">
		<div>
			<h2 class="text-[1.25rem] font-bold" style="color: var(--text-primary)">Per-Request Deep Dive</h2>
			<p class="text-[0.8rem] mt-1" style="color: var(--text-muted)">Inspect individual request lifecycle, forward passes, and per-token latency</p>
		</div>
		<select bind:value={selectedPrompt} class="rounded-lg px-3 py-2 text-sm focus:outline-none" style="background: var(--surface-raised); border: 1px solid var(--border-default); color: var(--text-primary);">
			{#each promptIds as pid}
				<option value={pid}>{pid}</option>
			{/each}
		</select>
	</div>

	<!-- Metrics -->
	{#each reqMetrics as m}
		{#if m}
			<div>
				<h3 class="text-sm font-semibold mb-2" style="color: {$meta.colors[m.label]}">{m.label}</h3>
				<div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
					<MetricCard label="Prompt Tokens" value={String(m.promptTokens)} />
					<MetricCard label="Output Tokens" value={String(m.outputTokens)} />
					<MetricCard label="TTFT" value="{m.ttft} ms" />
					<MetricCard label="TPOT" value="{m.tpot} ms" />
					<MetricCard label="E2E" value="{m.e2e} ms" />
					<MetricCard label="Median ITL" value="{m.medianItl} ms" />
					<MetricCard label="p95 ITL" value="{m.p95Itl} ms" />
				</div>
			</div>
		{/if}
	{/each}

	<!-- Gantt -->
	<div class="rounded-lg p-3" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
		<Chart options={buildGantt()} height="280px" />
	</div>

	<!-- Fwd Pass Scatter -->
	<div class="rounded-lg p-3" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
		<Chart options={buildFwdScatter()} height="400px" />
	</div>

	<!-- ITL Chart -->
	{#if $traceData.perToken.length > 0}
		<div class="rounded-lg p-3" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
			<Chart options={buildItlChart()} height="400px" />
		</div>
	{/if}
</div>
