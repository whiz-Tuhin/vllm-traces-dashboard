<script lang="ts">
	import { traceData, meta, currentMode } from '$lib/stores';
	import { median, percentile, mean, cdf, histogram } from '$lib/stats';
	import Chart from '../Chart.svelte';
	import MetricCard from '../MetricCard.svelte';
	import type { RequestTrace, ForwardPass, JoinRow } from '$lib/types';

	function modelData(arr: { model_label: string }[], label: string) {
		return arr.filter(r => r.model_label === label);
	}

	function summaryStats(reqs: RequestTrace[], fwds: ForwardPass[]) {
		const lats = reqs.map(r => r.total_latency_ms);
		const ttfts = reqs.map(r => r.ttft_ms);
		const tpots = reqs.map(r => r.tpot_ms);
		const wall = Math.max(...reqs.map(r => r.completion_ts)) - Math.min(...reqs.map(r => r.api_receive_ts));
		return {
			requests: reqs.length,
			fwdPasses: fwds.length,
			medE2E: median(lats).toFixed(0),
			p95E2E: percentile(lats, 95).toFixed(0),
			medTTFT: median(ttfts).toFixed(1),
			p95TTFT: percentile(ttfts, 95).toFixed(1),
			medTPOT: median(tpots).toFixed(2),
			p95TPOT: percentile(tpots, 95).toFixed(2),
			wallTime: wall.toFixed(1),
		};
	}

	function buildCdfOption(field: string, title: string, xLabel: string) {
		const series: echarts.EChartsOption['series'] = [];
		const markPoints: echarts.EChartsOption['series'] = [];
		for (const label of $meta.models) {
			const vals = modelData($traceData.requests, label).map(r => (r as Record<string, number>)[field]).sort((a, b) => a - b);
			const { x, y } = cdf(vals);
			series.push({
				name: label,
				type: 'line',
				data: x.map((v, i) => [v, y[i]]),
				smooth: true,
				symbol: 'none',
				lineStyle: { width: 2.5, color: $meta.colors[label] },
				itemStyle: { color: $meta.colors[label] },
			});
			for (const [p, tag] of [[50, 'p50'], [95, 'p95']] as [number, string][]) {
				const idx = Math.floor((p / 100) * (vals.length - 1));
				markPoints.push({
					type: 'scatter',
					name: `${label} ${tag}`,
					data: [[vals[idx], p]],
					symbol: 'circle',
					symbolSize: 10,
					itemStyle: { color: 'transparent', borderColor: $meta.colors[label], borderWidth: 2 },
					label: { show: true, formatter: `${tag}: ${vals[idx]?.toFixed(1)}`, position: 'right', fontSize: 10, color: $meta.colors[label] },
				});
			}
		}
		return {
			backgroundColor: 'transparent',
			title: { text: title, textStyle: { color: '#E8EAF0', fontSize: 14, fontWeight: 500 }, left: 16, top: 12 },
			tooltip: { trigger: 'axis', backgroundColor: '#1C2030', borderColor: 'rgba(255,255,255,0.09)', textStyle: { color: '#E8EAF0' } },
			legend: { show: true, bottom: 8, textStyle: { color: '#9BA1B0' } },
			grid: { left: 60, right: 30, top: 55, bottom: 45 },
			xAxis: { type: 'value', name: xLabel, nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			yAxis: { type: 'value', name: 'Percentile (%)', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			series: [...series, ...markPoints],
		} as echarts.EChartsOption;
	}

	function buildPhaseBreakdown() {
		const models = $meta.models;
		const phases = ['Scheduling', 'Queue/Other', 'Prefill', 'Decode'];
		const phaseData: Record<string, number[]> = {};
		for (const ph of phases) phaseData[ph] = [];

		for (const label of models) {
			const reqs = modelData($traceData.requests, label);
			const isStr = $currentMode === 'streaming' || $currentMode === 'multiturn';
			const sched = median(reqs.map(r => r.scheduling_overhead_ms));
			let prefill: number, decode: number;
			if (isStr) {
				const ttft = median(reqs.map(r => r.ttft_ms));
				prefill = ttft - sched;
				decode = median(reqs.map(r => r.decode_ms));
			} else {
				const joins = modelData($traceData.join, label);
				const byReq = new Map<string, JoinRow[]>();
				for (const j of joins) {
					if (!byReq.has(j.request_id)) byReq.set(j.request_id, []);
					byReq.get(j.request_id)!.push(j);
				}
				const prefills: number[] = [];
				const decodes: number[] = [];
				for (const [, rows] of byReq) {
					const sorted = rows.sort((a, b) => a.rel_start_s - b.rel_start_s);
					if (sorted.length > 0) prefills.push(sorted[0].duration_ms);
					if (sorted.length > 1) decodes.push(sorted.slice(1).reduce((s, r) => s + r.duration_ms, 0));
				}
				prefill = median(prefills);
				decode = median(decodes);
			}
			const e2e = median(reqs.map(r => r.total_latency_ms));
			const queue = Math.max(e2e - sched - prefill - decode, 0);
			phaseData['Scheduling'].push(Math.round(sched));
			phaseData['Queue/Other'].push(Math.round(queue));
			phaseData['Prefill'].push(Math.round(prefill));
			phaseData['Decode'].push(Math.round(decode));
		}

		return {
			backgroundColor: 'transparent',
			title: { text: 'Where Time Goes (Median per Request)', textStyle: { color: '#E8EAF0', fontSize: 14, fontWeight: 500 }, left: 16, top: 12 },
			tooltip: { trigger: 'axis', backgroundColor: '#1C2030', borderColor: 'rgba(255,255,255,0.09)', textStyle: { color: '#E8EAF0' } },
			legend: { bottom: 8, textStyle: { color: '#9BA1B0' } },
			grid: { left: 60, right: 30, top: 55, bottom: 45 },
			xAxis: { type: 'category', data: models, axisLabel: { color: '#636B7E' } },
			yAxis: { type: 'value', name: 'Time (ms)', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			series: phases.map(ph => ({
				name: ph,
				type: 'bar',
				stack: 'total',
				data: phaseData[ph],
				itemStyle: { color: $meta.phase_colors[ph] },
				label: { show: true, position: 'inside', formatter: (p: { value: number }) => p.value > 0 ? `${p.value}` : '', color: '#fff', fontSize: 11 },
			})),
		} as echarts.EChartsOption;
	}

	function buildHistogram() {
		const series: echarts.EChartsOption['series'] = [];
		for (const label of $meta.models) {
			const vals = modelData($traceData.requests, label).map(r => r.total_latency_ms);
			const { edges, counts } = histogram(vals, 50);
			series.push({
				name: label,
				type: 'bar',
				data: counts.map((c, i) => [edges[i], c]),
				barWidth: '90%',
				itemStyle: { color: $meta.colors[label], opacity: 0.6 },
			});
		}
		return {
			backgroundColor: 'transparent',
			title: { text: 'E2E Latency Distribution', textStyle: { color: '#E8EAF0', fontSize: 14, fontWeight: 500 }, left: 16, top: 12 },
			tooltip: { trigger: 'axis', backgroundColor: '#1C2030', borderColor: 'rgba(255,255,255,0.09)', textStyle: { color: '#E8EAF0' } },
			legend: { bottom: 8, textStyle: { color: '#9BA1B0' } },
			grid: { left: 60, right: 30, top: 55, bottom: 45 },
			xAxis: { type: 'value', name: 'E2E Latency (ms)', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			yAxis: { type: 'value', name: 'Count', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			series,
		} as echarts.EChartsOption;
	}

	function buildBoxPlot() {
		const boxData: number[][] = [];
		const categories: string[] = [];
		for (const label of $meta.models) {
			const vals = modelData($traceData.forwardPasses, label).map(f => f.duration_ms).sort((a, b) => a - b);
			if (!vals.length) continue;
			categories.push(label);
			boxData.push([
				percentile(vals, 0),
				percentile(vals, 25),
				median(vals),
				percentile(vals, 75),
				percentile(vals, 100),
			]);
		}
		return {
			backgroundColor: 'transparent',
			title: { text: 'Forward Pass Duration', textStyle: { color: '#E8EAF0', fontSize: 14, fontWeight: 500 }, left: 16, top: 12 },
			tooltip: { trigger: 'item', backgroundColor: '#1C2030', borderColor: 'rgba(255,255,255,0.09)', textStyle: { color: '#E8EAF0' } },
			grid: { left: 60, right: 30, top: 55, bottom: 35 },
			xAxis: { type: 'category', data: categories, axisLabel: { color: '#636B7E' } },
			yAxis: { type: 'log', name: 'Duration (ms)', nameTextStyle: { color: '#636B7E' }, axisLabel: { color: '#636B7E' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } } },
			series: [{
				type: 'boxplot',
				data: boxData,
				itemStyle: { color: '#3f3f46', borderColor: '#a1a1aa' },
			}],
		} as echarts.EChartsOption;
	}

	let stats = $derived($meta.models.map(label => {
		const reqs = modelData($traceData.requests, label);
		const fwds = modelData($traceData.forwardPasses, label);
		return { label, ...summaryStats(reqs as RequestTrace[], fwds as ForwardPass[]) };
	}));
</script>

<div class="space-y-8 px-8 py-7">
	<div>
		<h2 class="text-[1.25rem] font-bold" style="color: var(--text-primary)">Overview</h2>
		<p class="text-[0.8rem] mt-1" style="color: var(--text-muted)">Summary statistics and latency distributions</p>
	</div>

	<!-- Summary Stats Cards -->
	{#each stats as s}
		<div>
			<h3 class="text-sm font-semibold mb-3" style="color: var(--text-secondary)">{s.label}</h3>
			<div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-3">
				<MetricCard label="Requests" value={String(s.requests)} />
				<MetricCard label="Fwd Passes" value={String(s.fwdPasses)} />
				<MetricCard label="Median E2E" value="{s.medE2E} ms" />
				<MetricCard label="p95 E2E" value="{s.p95E2E} ms" />
				<MetricCard label="Median TTFT" value="{s.medTTFT} ms" />
				<MetricCard label="p95 TTFT" value="{s.p95TTFT} ms" />
				<MetricCard label="Median TPOT" value="{s.medTPOT} ms" />
				<MetricCard label="p95 TPOT" value="{s.p95TPOT} ms" />
				<MetricCard label="Wall Time" value="{s.wallTime} s" />
			</div>
		</div>
	{/each}

	<!-- CDF Charts -->
	<div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
		<div class="rounded-lg p-3" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
			<Chart options={buildCdfOption('total_latency_ms', 'End-to-End Latency CDF', 'Latency (ms)')} height="380px" />
		</div>
		<div class="rounded-lg p-3" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
			<Chart options={buildCdfOption('ttft_ms', 'Time-to-First-Token CDF', 'TTFT (ms)')} height="380px" />
		</div>
	</div>

	<!-- Phase Breakdown + Box Plot -->
	<div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
		<div class="rounded-lg p-3" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
			<Chart options={buildPhaseBreakdown()} height="400px" />
		</div>
		<div class="rounded-lg p-3" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
			<Chart options={buildBoxPlot()} height="400px" />
		</div>
	</div>

	<!-- Histogram -->
	<div class="rounded-lg p-3" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
		<Chart options={buildHistogram()} height="350px" />
	</div>
</div>
