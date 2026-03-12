<script lang="ts">
	import { currentTab } from '$lib/stores';
	import OverviewTab from '$lib/components/tabs/OverviewTab.svelte';
	import ForwardPassTab from '$lib/components/tabs/ForwardPassTab.svelte';
	import PerRequestTab from '$lib/components/tabs/PerRequestTab.svelte';
	import KvCacheTab from '$lib/components/tabs/KvCacheTab.svelte';
	import FwdPassCompareTab from '$lib/components/tabs/FwdPassCompareTab.svelte';
	import DataExportTab from '$lib/components/tabs/DataExportTab.svelte';
</script>

<svelte:head>
	<title>vLLM Trace Dashboard</title>
</svelte:head>

{#if $currentTab === 'overview'}
	<OverviewTab />
{:else if $currentTab === 'forward-pass'}
	<ForwardPassTab />
{:else if $currentTab === 'per-request'}
	<PerRequestTab />
{:else if $currentTab === 'kv-cache'}
	<KvCacheTab />
{:else if $currentTab === 'fwd-compare'}
	<FwdPassCompareTab />
{:else if $currentTab === 'data-export'}
	<DataExportTab />
{/if}
