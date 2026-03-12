<script lang="ts">
	import { onMount } from 'svelte';
	import * as echarts from 'echarts';

	let { options, height = '400px', className = '' }: {
		options: echarts.EChartsOption;
		height?: string;
		className?: string;
	} = $props();

	let container: HTMLDivElement;
	let chart: echarts.ECharts | null = null;
	let observer: MutationObserver | null = null;

	const DARK_THEME = 'vllm-enterprise-dark';
	const LIGHT_THEME = 'vllm-enterprise-light';

	function registerThemes() {
		echarts.registerTheme(DARK_THEME, {
			backgroundColor: 'transparent',
			textStyle: { fontFamily: 'Inter, system-ui, sans-serif', color: '#9BA1B0' },
			title: { textStyle: { color: '#E8EAF0', fontSize: 15, fontWeight: 600 }, subtextStyle: { color: '#636B7E' } },
			legend: { textStyle: { color: '#9BA1B0', fontSize: 12 } },
			tooltip: {
				backgroundColor: '#1C2030',
				borderColor: 'rgba(255,255,255,0.09)',
				textStyle: { color: '#E8EAF0', fontSize: 12 },
				extraCssText: 'border-radius: 8px; box-shadow: 0 8px 32px rgba(0,0,0,0.4); backdrop-filter: blur(8px);',
			},
			categoryAxis: {
				axisLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
				axisTick: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
				axisLabel: { color: '#636B7E', fontSize: 12 },
				splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } },
			},
			valueAxis: {
				axisLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
				axisTick: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
				axisLabel: { color: '#636B7E', fontSize: 12 },
				splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } },
				nameTextStyle: { color: '#636B7E', fontSize: 12 },
			},
			logAxis: {
				axisLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
				axisLabel: { color: '#636B7E', fontSize: 12 },
				splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } },
				nameTextStyle: { color: '#636B7E', fontSize: 12 },
			},
			dataZoom: [
				{ type: 'inside' },
				{
					type: 'slider',
					borderColor: 'rgba(255,255,255,0.06)',
					fillerColor: 'rgba(143,184,255,0.12)',
					handleStyle: { color: '#8FB8FF', borderColor: '#8FB8FF' },
					textStyle: { color: '#636B7E' },
					dataBackground: { lineStyle: { color: 'rgba(255,255,255,0.06)' }, areaStyle: { color: 'rgba(255,255,255,0.02)' } },
				},
			],
			color: ['#8FB8FF', '#FF9B7A', '#7ED7AB', '#B495F5', '#F4CD78', '#7CD8F0', '#F0A6CA', '#8DD3C7'],
		});

		echarts.registerTheme(LIGHT_THEME, {
			backgroundColor: 'transparent',
			textStyle: { fontFamily: 'Inter, system-ui, sans-serif', color: '#344054' },
			title: { textStyle: { color: '#111827', fontSize: 15, fontWeight: 600 }, subtextStyle: { color: '#667085' } },
			legend: { textStyle: { color: '#344054', fontSize: 12 } },
			tooltip: {
				backgroundColor: '#FFFFFF',
				borderColor: 'rgba(17,24,39,0.12)',
				textStyle: { color: '#111827', fontSize: 12 },
				extraCssText: 'border-radius: 8px; box-shadow: 0 10px 24px rgba(0,0,0,0.12);',
			},
			categoryAxis: {
				axisLine: { lineStyle: { color: 'rgba(17,24,39,0.16)' } },
				axisTick: { lineStyle: { color: 'rgba(17,24,39,0.16)' } },
				axisLabel: { color: '#667085', fontSize: 12 },
				splitLine: { lineStyle: { color: 'rgba(17,24,39,0.06)' } },
			},
			valueAxis: {
				axisLine: { lineStyle: { color: 'rgba(17,24,39,0.16)' } },
				axisTick: { lineStyle: { color: 'rgba(17,24,39,0.16)' } },
				axisLabel: { color: '#667085', fontSize: 12 },
				splitLine: { lineStyle: { color: 'rgba(17,24,39,0.06)' } },
				nameTextStyle: { color: '#667085', fontSize: 12 },
			},
			logAxis: {
				axisLine: { lineStyle: { color: 'rgba(17,24,39,0.16)' } },
				axisLabel: { color: '#667085', fontSize: 12 },
				splitLine: { lineStyle: { color: 'rgba(17,24,39,0.06)' } },
				nameTextStyle: { color: '#667085', fontSize: 12 },
			},
			dataZoom: [
				{ type: 'inside' },
				{
					type: 'slider',
					borderColor: 'rgba(17,24,39,0.14)',
					fillerColor: 'rgba(71,119,230,0.14)',
					handleStyle: { color: '#4777E6', borderColor: '#4777E6' },
					textStyle: { color: '#667085' },
					dataBackground: { lineStyle: { color: 'rgba(17,24,39,0.16)' }, areaStyle: { color: 'rgba(17,24,39,0.04)' } },
				},
			],
			color: ['#4777E6', '#E06A42', '#1F8A5B', '#8A63D2', '#C98914', '#2D9CC2', '#D1669C', '#3E9A90'],
		});
	}

	function currentThemeName(): string {
		return document.documentElement.dataset.theme === 'light' ? LIGHT_THEME : DARK_THEME;
	}

	function compactGrid(grid: unknown, hasTitle: boolean): unknown {
		const compactOne = (g: Record<string, unknown>) => {
			const next = { ...g } as Record<string, unknown>;
			next.containLabel = true;
			if (typeof next.top === 'number') next.top = hasTitle ? Math.max(56, next.top - 2) : Math.max(44, next.top - 2);
			if (typeof next.bottom === 'number' && next.bottom >= 46) next.bottom = Math.max(38, next.bottom - 4);
			if (typeof next.left === 'number' && next.left >= 64) next.left = Math.max(58, next.left - 4);
			if (typeof next.right === 'number' && next.right >= 28) next.right = Math.max(22, next.right - 4);
			return next;
		};
		if (Array.isArray(grid)) return grid.map((g) => (g && typeof g === 'object' ? compactOne(g as Record<string, unknown>) : g));
		if (grid && typeof grid === 'object') return compactOne(grid as Record<string, unknown>);
		return { left: 58, right: 22, top: hasTitle ? 56 : 44, bottom: 38, containLabel: true };
	}

	function enhanceSeries(series: unknown): unknown {
		if (!Array.isArray(series)) return series;
		return series.map((s) => {
			if (!s || typeof s !== 'object') return s;
			const row = { ...(s as Record<string, unknown>) };
			const type = row.type as string | undefined;
			if (type === 'line') {
				const existing = (row.lineStyle as Record<string, unknown> | undefined) ?? {};
				const width = typeof existing.width === 'number' ? Math.max(existing.width, 2.4) : 2.4;
				row.lineStyle = { ...existing, width };
				if (row.symbol === undefined) row.symbol = 'circle';
				if (row.showSymbol === undefined) row.showSymbol = false;
			}
			if (type === 'scatter') {
				if (typeof row.symbolSize === 'number') row.symbolSize = Math.max(row.symbolSize, 7);
			}
			if (type === 'bar') {
				if (row.barMaxWidth === undefined) row.barMaxWidth = 14;
			}
			return row;
		});
	}

	function normalizeOptions(opts: echarts.EChartsOption): echarts.EChartsOption {
		const next = { ...opts } as echarts.EChartsOption;
		const light = document.documentElement.dataset.theme === 'light';
		const textPrimary = light ? '#111827' : '#E8EAF0';
		const textSecondary = light ? '#667085' : '#636B7E';
		const legendText = light ? '#344054' : '#9BA1B0';
		const gridLine = light ? 'rgba(17,24,39,0.06)' : 'rgba(255,255,255,0.04)';
		const axisLine = light ? 'rgba(17,24,39,0.16)' : 'rgba(255,255,255,0.06)';
		const tooltipBg = light ? '#FFFFFF' : '#1C2030';
		const tooltipBorder = light ? 'rgba(17,24,39,0.12)' : 'rgba(255,255,255,0.09)';

		const normalizeAxis = (axis: unknown) => {
			const one = (a: Record<string, unknown>) => ({
				...a,
				nameTextStyle: { ...(a.nameTextStyle as Record<string, unknown> ?? {}), color: textSecondary, fontSize: 12 },
				axisLabel: { ...(a.axisLabel as Record<string, unknown> ?? {}), color: textSecondary, fontSize: 12 },
				axisLine: { ...(a.axisLine as Record<string, unknown> ?? {}), lineStyle: { ...((a.axisLine as Record<string, unknown> | undefined)?.lineStyle as Record<string, unknown> ?? {}), color: axisLine } },
				splitLine: { ...(a.splitLine as Record<string, unknown> ?? {}), lineStyle: { ...((a.splitLine as Record<string, unknown> | undefined)?.lineStyle as Record<string, unknown> ?? {}), color: gridLine } },
			});
			if (Array.isArray(axis)) return axis.map((a) => (a && typeof a === 'object' ? one(a as Record<string, unknown>) : a));
			if (axis && typeof axis === 'object') return one(axis as Record<string, unknown>);
			return axis;
		};

		const normalizeYAxis = (axis: unknown) => {
			const one = (a: Record<string, unknown>) => ({
				...a,
				nameLocation: 'middle',
				nameGap: typeof a.nameGap === 'number' ? Math.max(a.nameGap, 42) : 42,
			});
			if (Array.isArray(axis)) return axis.map((a) => (a && typeof a === 'object' ? one(a as Record<string, unknown>) : a));
			if (axis && typeof axis === 'object') return one(axis as Record<string, unknown>);
			return axis;
		};

		const normalizeTitle = (title: unknown) => {
			const one = (t: Record<string, unknown>) => ({
				...t,
				textStyle: { ...(t.textStyle as Record<string, unknown> ?? {}), color: textPrimary, fontSize: 15, fontWeight: 600 },
				subtextStyle: { ...(t.subtextStyle as Record<string, unknown> ?? {}), color: textSecondary },
			});
			if (Array.isArray(title)) return title.map((t) => (t && typeof t === 'object' ? one(t as Record<string, unknown>) : t));
			if (title && typeof title === 'object') return one(title as Record<string, unknown>);
			return title;
		};

		const normalizeLegend = (legend: unknown) => {
			const one = (l: Record<string, unknown>) => ({ ...l, textStyle: { ...(l.textStyle as Record<string, unknown> ?? {}), color: legendText, fontSize: 12 } });
			if (Array.isArray(legend)) return legend.map((l) => (l && typeof l === 'object' ? one(l as Record<string, unknown>) : l));
			if (legend && typeof legend === 'object') return one(legend as Record<string, unknown>);
			return legend;
		};

		next.title = normalizeTitle(next.title) as echarts.EChartsOption['title'];
		next.legend = normalizeLegend(next.legend) as echarts.EChartsOption['legend'];
		next.tooltip = {
			...(next.tooltip as Record<string, unknown> ?? {}),
			backgroundColor: tooltipBg,
			borderColor: tooltipBorder,
			textStyle: { ...((next.tooltip as Record<string, unknown> | undefined)?.textStyle as Record<string, unknown> ?? {}), color: textPrimary, fontSize: 12 },
		};
		next.xAxis = normalizeAxis(next.xAxis) as echarts.EChartsOption['xAxis'];
		next.yAxis = normalizeYAxis(normalizeAxis(next.yAxis)) as echarts.EChartsOption['yAxis'];
		next.grid = compactGrid(next.grid, Boolean(next.title)) as echarts.EChartsOption['grid'];
		next.series = enhanceSeries(next.series) as echarts.EChartsOption['series'];
		return next;
	}

	function initChart() {
		const theme = currentThemeName();
		chart?.dispose();
		chart = echarts.init(container, theme, { renderer: 'canvas' });
		chart.setOption(normalizeOptions(options), { notMerge: true });
	}

	onMount(() => {
		registerThemes();
		initChart();

		const ro = new ResizeObserver(() => chart?.resize());
		ro.observe(container);
		observer = new MutationObserver(() => {
			initChart();
		});
		observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

		return () => {
			ro.disconnect();
			observer?.disconnect();
			chart?.dispose();
		};
	});

	$effect(() => {
		if (chart && options) {
			chart.setOption(normalizeOptions(options), { notMerge: true });
		}
	});
</script>

<div bind:this={container} class="w-full {className}" style="height: {height}"></div>
