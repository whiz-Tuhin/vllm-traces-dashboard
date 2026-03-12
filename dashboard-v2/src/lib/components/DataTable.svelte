<script lang="ts">
	let { columns, rows, maxHeight = '400px' }: {
		columns: { key: string; label: string; format?: (v: unknown) => string }[];
		rows: Record<string, unknown>[];
		maxHeight?: string;
	} = $props();
</script>

<div class="rounded-lg overflow-hidden" style="border: 1px solid var(--border-subtle);">
	<div class="overflow-auto" style="max-height: {maxHeight}">
		<table class="w-full text-[0.8rem]">
			<thead class="sticky top-0 z-10" style="background: var(--surface-overlay);">
				<tr>
					{#each columns as col}
						<th class="px-4 py-2.5 text-left text-[0.65rem] font-semibold uppercase tracking-wider" style="color: var(--text-muted); border-bottom: 1px solid var(--border-default);">{col.label}</th>
					{/each}
				</tr>
			</thead>
			<tbody>
				{#each rows as row, i}
					<tr class="transition-colors" style="background: {i % 2 === 0 ? 'var(--surface-base)' : 'var(--surface-raised)'};" onmouseenter={(e) => (e.currentTarget as HTMLElement).style.background = 'var(--surface-hover)'} onmouseleave={(e) => (e.currentTarget as HTMLElement).style.background = i % 2 === 0 ? 'var(--surface-base)' : 'var(--surface-raised)'}>
						{#each columns as col}
							<td class="px-4 py-2 tabular-nums whitespace-nowrap font-[JetBrains_Mono,monospace] text-[0.75rem]" style="color: var(--text-secondary);">
								{col.format ? col.format(row[col.key]) : row[col.key]}
							</td>
						{/each}
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
</div>
