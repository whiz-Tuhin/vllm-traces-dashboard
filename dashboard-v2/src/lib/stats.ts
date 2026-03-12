export function median(arr: number[]): number {
	if (!arr.length) return 0;
	const s = [...arr].sort((a, b) => a - b);
	const mid = Math.floor(s.length / 2);
	return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

export function percentile(arr: number[], p: number): number {
	if (!arr.length) return 0;
	const s = [...arr].sort((a, b) => a - b);
	const idx = (p / 100) * (s.length - 1);
	const lo = Math.floor(idx);
	const hi = Math.ceil(idx);
	if (lo === hi) return s[lo];
	return s[lo] + (s[hi] - s[lo]) * (idx - lo);
}

export function mean(arr: number[]): number {
	if (!arr.length) return 0;
	return arr.reduce((a, b) => a + b, 0) / arr.length;
}

export function std(arr: number[]): number {
	if (arr.length < 2) return 0;
	const m = mean(arr);
	return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / (arr.length - 1));
}

export function cdf(arr: number[]): { x: number[]; y: number[] } {
	const s = [...arr].sort((a, b) => a - b);
	const x = s;
	const y = s.map((_, i) => (i / s.length) * 100);
	return { x, y };
}

export function ols(xs: number[], ys: number[]): { slope: number; intercept: number; xFit: number[]; yFit: number[] } {
	const n = xs.length;
	if (n < 2) return { slope: 0, intercept: 0, xFit: [], yFit: [] };
	const mx = mean(xs);
	const my = mean(ys);
	let num = 0, den = 0;
	for (let i = 0; i < n; i++) {
		num += (xs[i] - mx) * (ys[i] - my);
		den += (xs[i] - mx) ** 2;
	}
	const slope = den ? num / den : 0;
	const intercept = my - slope * mx;
	const xMin = Math.min(...xs);
	const xMax = Math.max(...xs);
	const step = (xMax - xMin) / 100;
	const xFit: number[] = [];
	const yFit: number[] = [];
	for (let x = xMin; x <= xMax; x += step) {
		xFit.push(x);
		yFit.push(slope * x + intercept);
	}
	return { slope, intercept, xFit, yFit };
}

export function rollingMean(arr: number[], window: number): number[] {
	const result: number[] = [];
	for (let i = 0; i < arr.length; i++) {
		const start = Math.max(0, i - window + 1);
		const slice = arr.slice(start, i + 1);
		result.push(mean(slice));
	}
	return result;
}

export function histogram(arr: number[], bins: number): { edges: number[]; counts: number[] } {
	if (!arr.length) return { edges: [], counts: [] };
	const min = Math.min(...arr);
	const max = Math.max(...arr);
	const binWidth = (max - min) / bins || 1;
	const edges: number[] = [];
	const counts: number[] = new Array(bins).fill(0);
	for (let i = 0; i <= bins; i++) edges.push(min + i * binWidth);
	for (const v of arr) {
		let idx = Math.floor((v - min) / binWidth);
		if (idx >= bins) idx = bins - 1;
		if (idx < 0) idx = 0;
		counts[idx]++;
	}
	return { edges, counts };
}
