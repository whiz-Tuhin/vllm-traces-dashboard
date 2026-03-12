export interface RequestTrace {
	request_id: string;
	model: string;
	prompt_id: string;
	max_tokens: number;
	prompt_tokens: number;
	output_tokens: number;
	api_receive_ts: number;
	engine_add_request_ts: number;
	first_token_ts: number;
	completion_ts: number;
	ttft_ms: number;
	tpot_ms: number;
	total_latency_ms: number;
	scheduling_overhead_ms: number;
	model_label: string;
	rel_api_receive_ts: number;
	rel_engine_add_request_ts: number;
	rel_first_token_ts: number;
	rel_completion_ts: number;
	decode_ms: number;
}

export interface ForwardPass {
	fwd_id: number;
	start_ts: number;
	end_ts: number;
	duration_ms: number;
	batch_size: number;
	total_tokens: number;
	model_label: string;
	rel_start_s: number;
	rel_end_s: number;
}

export interface JoinRow {
	fwd_id: number;
	request_id: string;
	prompt_id: string;
	prompt_tokens: number;
	output_tokens: number;
	tokens_in_pass: number;
	rel_start_s: number;
	rel_end_s: number;
	duration_ms: number;
	batch_size: number;
	total_tokens: number;
	model_label: string;
}

export interface KvCacheRow {
	fwd_id: number;
	request_id: string;
	rel_start_s: number;
	duration_ms: number;
	is_prefill: boolean;
	past_kv_cache_size: number;
	prefix_tokens: number;
	decode_tokens: number;
	tokens_generated_so_far: number;
	num_prompt_tokens: number;
	num_scheduled_tokens: number;
	model_label: string;
}

export interface PerTokenRow {
	request_id: string;
	prompt_id: string;
	token_idx: number;
	timestamp: number;
	rel_ts_ms: number;
	itl_ms: number;
	model_label: string;
	kv_cache_size?: number;
}

export interface Meta {
	models: string[];
	modes: string[];
	colors: Record<string, string>;
	phase_colors: Record<string, string>;
}

export type TraceMode = 'non-streaming' | 'streaming' | 'multiturn';

export interface TraceData {
	requests: RequestTrace[];
	forwardPasses: ForwardPass[];
	join: JoinRow[];
	kvCache: KvCacheRow[];
	perToken: PerTokenRow[];
}
