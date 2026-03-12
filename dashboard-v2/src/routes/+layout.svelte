<script lang="ts">
	import '../app.css';
	import { currentMode, currentTab, meta, traceData, loading, availableTabs } from '$lib/stores';
	import { loadMeta, loadTraceData } from '$lib/data';
	import { onMount } from 'svelte';
	import type { TraceMode } from '$lib/types';

	let { children } = $props();

	const modeLabels: Record<TraceMode, { label: string; desc: string }> = {
		'non-streaming': { label: 'Single-Turn', desc: 'Non-Streaming' },
		'streaming': { label: 'Single-Turn', desc: 'Streaming' },
		'multiturn': { label: 'Multi-Turn', desc: 'KV Cache' },
	};

	const tabIcons: Record<string, string> = {
		'overview': '◈',
		'forward-pass': '⟐',
		'per-request': '◉',
		'kv-cache': '⬡',
		'fwd-compare': '⇋',
		'data-export': '⊞',
	};

	let uiTheme = $state<'dark' | 'light'>('dark');
	let loadError = $state<string | null>(null);

	function applyTheme(theme: 'dark' | 'light') {
		uiTheme = theme;
		document.documentElement.dataset.theme = theme;
		try {
			localStorage.setItem('vllm-dashboard-theme', theme);
		} catch {
			/* ignore storage errors */
		}
	}

	async function switchMode(mode: TraceMode) {
		$loading = true;
		$loadError = null;
		$currentMode = mode;
		try {
			const data = await loadTraceData(mode);
			$traceData = data;
			const tabs = $availableTabs;
			if (!tabs.find(t => t.id === $currentTab)) {
				$currentTab = tabs[0].id;
			}
		} catch (e) {
			$loadError = e instanceof Error ? e.message : String(e);
		} finally {
			$loading = false;
		}
	}

	onMount(async () => {
		try {
			const savedTheme = localStorage.getItem('vllm-dashboard-theme');
			if (savedTheme === 'light' || savedTheme === 'dark') {
				applyTheme(savedTheme);
			} else {
				applyTheme('dark');
			}
		} catch {
			applyTheme('dark');
		}

		$loading = true;
		$loadError = null;
		try {
			const m = await loadMeta();
			$meta = m;
			await switchMode('streaming');
		} catch (e) {
			$loadError = e instanceof Error ? e.message : String(e);
			$loading = false;
		}
	});
</script>

<div class="flex h-screen overflow-hidden" style="background: var(--surface-base);">
	<!-- Sidebar - z-index ensures it stays clickable above any overlay -->
	<aside class="w-[260px] flex flex-col shrink-0 relative z-10" style="background: var(--surface-raised); border-right: 1px solid var(--border-subtle);">
		<!-- Logo -->
		<div class="px-6 py-5" style="border-bottom: 1px solid var(--border-subtle);">
			<div class="flex items-center gap-3">
				<div class="w-8 h-8 rounded-lg flex items-center justify-center" style="background: var(--accent-subtle);">
					<span class="text-sm font-bold" style="color: var(--accent-primary);">V</span>
				</div>
				<div>
					<h1 class="text-[0.95rem] font-semibold tracking-tight" style="color: var(--text-primary);">vLLM Traces</h1>
					<p class="text-[0.65rem] font-medium" style="color: var(--text-muted);">Performance Dashboard</p>
				</div>
			</div>
		</div>

		<!-- Mode Selector -->
		<div class="px-5 py-4" style="border-bottom: 1px solid var(--border-subtle);">
			<span class="text-[0.6rem] font-semibold uppercase tracking-[0.12em]" style="color: var(--text-muted);">Trace Category</span>
			<div class="mt-3 flex flex-col gap-1">
				{#each Object.entries(modeLabels) as [key, info]}
					<button
						type="button"
						onclick={() => switchMode(key as TraceMode)}
						class="flex items-center gap-3 px-3 py-2.5 rounded-lg text-[0.8rem] transition-all cursor-pointer w-full text-left"
						style={$currentMode === key
							? `background: var(--accent-subtle); color: var(--accent-primary); border: 1px solid rgba(124,147,219,0.18);`
							: `color: var(--text-secondary); border: 1px solid transparent;`}
						onmouseenter={(e) => { if ($currentMode !== key) (e.currentTarget as HTMLElement).style.background = 'var(--surface-hover)' }}
						onmouseleave={(e) => { if ($currentMode !== key) (e.currentTarget as HTMLElement).style.background = 'transparent' }}
					>
						<div class="text-left">
							<div class="font-medium">{info.label}</div>
							<div class="text-[0.65rem] mt-0.5" style="color: {$currentMode === key ? 'var(--accent-primary)' : 'var(--text-muted)'}">{info.desc}</div>
						</div>
					</button>
				{/each}
			</div>
		</div>

		<!-- Tab Navigation -->
		<nav class="flex-1 px-5 py-4 overflow-y-auto">
			<span class="text-[0.6rem] font-semibold uppercase tracking-[0.12em]" style="color: var(--text-muted);">Views</span>
			<div class="mt-3 flex flex-col gap-0.5">
				{#each $availableTabs as tab}
					<button
						type="button"
						onclick={() => $currentTab = tab.id}
						class="flex items-center gap-3 px-3 py-2.5 rounded-lg text-[0.8rem] transition-all cursor-pointer w-full text-left"
						style={$currentTab === tab.id
							? `background: var(--surface-overlay); color: var(--text-primary); font-weight: 500;`
							: `color: var(--text-secondary);`}
						onmouseenter={(e) => { if ($currentTab !== tab.id) (e.currentTarget as HTMLElement).style.background = 'var(--surface-hover)' }}
						onmouseleave={(e) => { if ($currentTab !== tab.id) (e.currentTarget as HTMLElement).style.background = 'transparent' }}
					>
						<span class="text-[0.85rem] w-5 text-center" style="color: {$currentTab === tab.id ? 'var(--accent-primary)' : 'var(--text-muted)'}">{tabIcons[tab.id] || '◇'}</span>
						{tab.label}
					</button>
				{/each}
			</div>
		</nav>

		<!-- Footer -->
		<div class="px-6 py-4" style="border-top: 1px solid var(--border-subtle);">
			<div class="flex items-center justify-between mb-3">
				<span class="text-[0.6rem] font-semibold uppercase tracking-[0.12em]" style="color: var(--text-muted);">Theme</span>
				<div class="inline-flex rounded-md p-0.5" style="background: var(--surface-overlay); border: 1px solid var(--border-subtle);">
					<button
						class="px-2.5 py-1 rounded text-[0.65rem] font-medium transition-colors"
						style={uiTheme === 'dark' ? 'background: var(--accent-subtle); color: var(--accent-primary);' : 'color: var(--text-muted);'}
						onclick={() => applyTheme('dark')}
					>
						Dark
					</button>
					<button
						class="px-2.5 py-1 rounded text-[0.65rem] font-medium transition-colors"
						style={uiTheme === 'light' ? 'background: var(--accent-subtle); color: var(--accent-primary);' : 'color: var(--text-muted);'}
						onclick={() => applyTheme('light')}
					>
						Light
					</button>
				</div>
			</div>
			<div class="flex items-center gap-2">
				<div class="w-1.5 h-1.5 rounded-full" style="background: var(--color-success);"></div>
				<span class="text-[0.6rem] font-medium" style="color: var(--text-muted);">L40s · vLLM 0.7.3 · 200 prompts</span>
			</div>
		</div>
	</aside>

	<!-- Main Content -->
	<main class="flex-1 overflow-y-auto" style="background: var(--surface-base);">
		{#if $loadError}
			<div class="flex flex-col items-center justify-center h-full gap-4 p-8">
				<div class="text-[0.9rem] font-medium" style="color: var(--color-error, #e5534b);">Failed to load traces</div>
				<p class="text-[0.8rem] text-center max-w-md" style="color: var(--text-muted);">{$loadError}</p>
				<p class="text-[0.7rem] text-center max-w-md" style="color: var(--text-muted);">
					If using Blob: ensure BLOB_READ_WRITE_TOKEN is set, the Blob store is linked to this project, and files are at data/meta.json, data/streaming/requests.json, etc.
				</p>
				<button
					class="px-4 py-2 rounded-lg text-[0.8rem] font-medium"
					style="background: var(--accent-subtle); color: var(--accent-primary); border: 1px solid rgba(124,147,219,0.3);"
					onclick={async () => {
						$loadError = null;
						$loading = true;
						try {
							const m = await loadMeta();
							$meta = m;
							await switchMode('streaming');
						} catch (e) {
							$loadError = e instanceof Error ? e.message : String(e);
						} finally {
							$loading = false;
						}
					}}
				>
					Retry
				</button>
			</div>
		{:else if $loading}
			<div class="flex flex-col items-center justify-center h-full gap-4">
				<div class="w-9 h-9 rounded-full animate-spin" style="border: 2px solid var(--border-default); border-top-color: var(--accent-primary);"></div>
				<span class="text-[0.8rem] font-medium" style="color: var(--text-muted);">Loading traces...</span>
			</div>
		{:else}
			{@render children()}
		{/if}
	</main>
</div>
