<script lang="ts">
	import { traceData, meta } from '$lib/stores';
	import { rollingMean, histogram } from '$lib/stats';
	import Chart from '../Chart.svelte';

	let rollWindow = $state(20);

	function buildTimelineScatter() {
		const series: echarts.EChartsOption['series'] = [];
		for (const label of $meta.models) {
			const fwds = $traceData.forwardPasses.filter(f => f.model_label === label);
			series.push({
				name: label,
				type: 'scatter',
				data: fwds.map(f => [f.rel_start_s, f.duration_ms, f.batch_size, f.fwd_id, f.total_tokens]),
				symbolSize: (val: number[]) => Math.max(val[2] * 1.5 + 3, 4),
				itemStyle: { color: $meta.colors[label], opacity: 0.55 },
				tooltip: {
					formatter: (p: { value: number[] }) => {
						const [t, dur, bs, fid, tok] = p.value;
						return `<b>${label}</b><br>Time: ${t.toFixed(2)}s<br>Duration: ${dur.toFixed(1)} ms<br>Batch: ${bs}<br>Tokens: ${tok}<br>fwd_id: ${fid}`;
					}
				},
			});
		}
		return {
			backgroundColor: 'transparent',
			title: { text: 'GPU Time per Forward Pass (log scale)', textStyle: { color: '#E8EAF0', fontSize: 14, fontWeight: 500 }, left: 16, top: 12 },
			tooltip: { backgroundColor: '#1C2030', borderColor: 'rgba(255,255,255,0.09)', textStyle: { color: '#E8EAF0' } },
			legend: { bottom: 8, textStyle: { color: '#9BA1B0' } },
			grid: { left: 70, right: 30, top: 55, bottom: 45 },
			xAxis: { type: 'value', name: 'Relative Time (s)', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			yAxis: { type: 'log', name: 'Duration (ms)', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 5, height: 20, borderColor: 'rgba(255,255,255,0.06)', fillerColor: 'rgba(124,147,219,0.08)' }],
			series,
		} as echarts.EChartsOption;
	}

	function buildBatchSizeOverTime() {
		const series: echarts.EChartsOption['series'] = [];
		for (const label of $meta.models) {
			const fwds = $traceData.forwardPasses.filter(f => f.model_label === label).sort((a, b) => a.rel_start_s - b.rel_start_s);
			const times = fwds.map(f => f.rel_start_s);
			const sizes = fwds.map(f => f.batch_size);
			const avg = rollingMean(sizes, rollWindow);
			series.push({
				name: `${label} raw`,
				type: 'line',
				data: times.map((t, i) => [t, sizes[i]]),
				lineStyle: { width: 1, type: 'dotted', color: $meta.colors[label], opacity: 0.4 },
				symbol: 'none',
				itemStyle: { color: $meta.colors[label] },
			});
			series.push({
				name: `${label} avg`,
				type: 'line',
				data: times.map((t, i) => [t, avg[i]]),
				lineStyle: { width: 2.5, color: $meta.colors[label] },
				symbol: 'none',
				itemStyle: { color: $meta.colors[label] },
			});
		}
		return {
			backgroundColor: 'transparent',
			title: { text: 'Continuous Batching Occupancy', textStyle: { color: '#E8EAF0', fontSize: 14, fontWeight: 500 }, left: 16, top: 12 },
			tooltip: { trigger: 'axis', backgroundColor: '#1C2030', borderColor: 'rgba(255,255,255,0.09)', textStyle: { color: '#E8EAF0' } },
			legend: { bottom: 8, textStyle: { color: '#9BA1B0' } },
			grid: { left: 60, right: 30, top: 55, bottom: 45 },
			xAxis: { type: 'value', name: 'Relative Time (s)', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			yAxis: { type: 'value', name: 'Requests in Batch', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			series,
		} as echarts.EChartsOption;
	}

	function buildBatchHistogram() {
		const series: echarts.EChartsOption['series'] = [];
		for (const label of $meta.models) {
			const sizes = $traceData.forwardPasses.filter(f => f.model_label === label).map(f => f.batch_size);
			const { edges, counts } = histogram(sizes, 17);
			series.push({
				name: label,
				type: 'bar',
				data: counts.map((c, i) => [edges[i], c]),
				itemStyle: { color: $meta.colors[label], opacity: 0.7 },
			});
		}
		return {
			backgroundColor: 'transparent',
			title: { text: 'Batch Size Distribution', textStyle: { color: '#E8EAF0', fontSize: 14, fontWeight: 500 }, left: 16, top: 12 },
			tooltip: { trigger: 'axis', backgroundColor: '#1C2030', borderColor: 'rgba(255,255,255,0.09)', textStyle: { color: '#E8EAF0' } },
			legend: { bottom: 8, textStyle: { color: '#9BA1B0' } },
			grid: { left: 60, right: 30, top: 55, bottom: 45 },
			xAxis: { type: 'value', name: 'Requests per Fwd Pass', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			yAxis: { type: 'value', name: 'Count', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			series,
		} as echarts.EChartsOption;
	}

	function buildThroughput() {
		const binSize = 1;
		const series: echarts.EChartsOption['series'] = [];
		for (const label of $meta.models) {
			const fwds = $traceData.forwardPasses.filter(f => f.model_label === label);
			const bins = new Map<number, number>();
			for (const f of fwds) {
				const tb = Math.floor(f.rel_start_s / binSize) * binSize;
				bins.set(tb, (bins.get(tb) || 0) + f.total_tokens);
			}
			const sorted = [...bins.entries()].sort((a, b) => a[0] - b[0]);
			series.push({
				name: label,
				type: 'line',
				data: sorted.map(([t, tok]) => [t, tok / binSize]),
				lineStyle: { width: 2, color: $meta.colors[label] },
				symbol: 'circle',
				symbolSize: 4,
				itemStyle: { color: $meta.colors[label] },
			});
		}
		return {
			backgroundColor: 'transparent',
			title: { text: 'Token Throughput Over Time', textStyle: { color: '#E8EAF0', fontSize: 14, fontWeight: 500 }, left: 16, top: 12 },
			tooltip: { trigger: 'axis', backgroundColor: '#1C2030', borderColor: 'rgba(255,255,255,0.09)', textStyle: { color: '#E8EAF0' } },
			legend: { bottom: 8, textStyle: { color: '#9BA1B0' } },
			grid: { left: 60, right: 30, top: 55, bottom: 45 },
			xAxis: { type: 'value', name: 'Relative Time (s)', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			yAxis: { type: 'value', name: 'Tokens / s', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			series,
		} as echarts.EChartsOption;
	}
</script>

<div class="px-8 py-7 space-y-8">
	<div>
		<h2 class="text-[1.25rem] font-bold" style="color: var(--text-primary);">Forward Pass</h2>
		<p class="text-[0.8rem] mt-1" style="color: var(--text-muted);">GPU forward pass timeline, batch size, and throughput</p>
	</div>

	<div class="rounded-lg p-3" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
		<Chart options={buildTimelineScatter()} height="450px" />
	</div>

	<div class="flex items-center gap-4 px-2">
		<label for="roll-window" class="text-sm" style="color: var(--text-secondary);">Rolling avg window:</label>
		<input id="roll-window" type="range" min="1" max="50" bind:value={rollWindow} class="w-48" />
		<span class="text-sm tabular-nums" style="color: var(--text-muted);">{rollWindow} passes</span>
	</div>

	<div class="rounded-lg p-3" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
		<Chart options={buildBatchSizeOverTime()} height="380px" />
	</div>

	<div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
		<div class="rounded-lg p-3" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
			<Chart options={buildBatchHistogram()} height="380px" />
		</div>
		<div class="rounded-lg p-3" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
			<Chart options={buildThroughput()} height="380px" />
		</div>
	</div>
</div>
