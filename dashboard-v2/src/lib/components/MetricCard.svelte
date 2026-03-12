<script lang="ts">
	let { label, value, sub = '', color = '' }: {
		label: string;
		value: string;
		sub?: string;
		color?: string;
	} = $props();
</script>

<div class="rounded-lg px-4 py-3.5 flex flex-col gap-1.5" style="background: var(--surface-raised); border: 1px solid var(--border-subtle);">
	<span class="text-[0.7rem] font-medium tracking-wide uppercase" style="color: var(--text-muted)">{label}</span>
	<span class="text-[1.35rem] font-semibold tabular-nums font-[JetBrains_Mono,monospace] tracking-tight" style={color ? `color: ${color}` : 'color: var(--text-primary)'}>{value}</span>
	{#if sub}
		<span class="text-[0.7rem]" style="color: var(--text-muted)">{sub}</span>
	{/if}
</div>
