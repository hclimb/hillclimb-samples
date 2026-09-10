"""Summarize the inference-speed benchmark from bench_msa.json + bench_membed.json.

Drops batch 0 of each run (JIT-compilation warmup) and reports steady-state
prefill latency (single forward over doc memory), decode throughput, and
end-to-end per-query latency, alongside the doc-token budget.

    python scripts/embed/analyze_speed.py <bench_dir>
"""
import sys, os, json, statistics

root = sys.argv[1] if len(sys.argv) > 1 else 'results/membench'

def load(name):
    p = os.path.join(root, name)
    return json.load(open(p)) if os.path.exists(p) else None

msa = load('bench_msa.json')
mb = load('bench_membed.json')

def mean(xs): return statistics.mean(xs) if xs else float('nan')

print(f"# bench dir: {root}\n")

if msa:
    b = msa.get('batches', [])[1:] or msa.get('batches', [])  # drop warmup
    pf = [x['prefill_s'] / x['B'] for x in b]
    dec_tps = [x['decode_steps'] * x['B'] / x['decode_s'] for x in b if x['decode_s'] > 0]
    e2e = [(x['prefill_s'] + x['decode_s']) / x['B'] for x in b]
    msa['_steady'] = {
        'prefill_s_per_query': round(mean(pf), 4),
        'decode_tokens_per_s': round(mean(dec_tps), 1),
        'e2e_s_per_query': round(mean(e2e), 4),
        'n_batches_used': len(b),
    }
    print("## MSA-4B")
    print(json.dumps({**{k: v for k, v in msa.items() if k != 'batches'},
                      'batches_used': len(b)}, indent=2))

if mb:
    b = mb.get('batches', [])[1:] or mb.get('batches', [])
    pf = [x['prefill_s'] / x['B'] for x in b]
    e2e = [x['gen_e2e_s'] / x['B'] for x in b]
    # decode throughput is a LOWER bound: gen may early-stop before max_new_tokens
    dec_tps = [x['max_new_tokens'] * x['B'] / (x['gen_e2e_s'] - x['prefill_s'])
               for x in b if x['gen_e2e_s'] > x['prefill_s']]
    mb['_steady'] = {
        'prefill_s_per_query': round(mean(pf), 4),
        'decode_tokens_per_s_lowerbound': round(mean(dec_tps), 1),
        'e2e_s_per_query': round(mean(e2e), 4),
        'n_batches_used': len(b),
    }
    print("\n## qwen3_mem_embed")
    print(json.dumps({**{k: v for k, v in mb.items() if k != 'batches'},
                      'batches_used': len(b)}, indent=2))

if msa and mb:
    s_msa, s_mb = msa['_steady'], mb['_steady']
    print("\n## Steady-state (warmup dropped)")
    print(f"{'metric':32s} {'MSA-4B':>14s} {'membed':>14s}")
    print(f"{'prefill s/query':32s} {s_msa['prefill_s_per_query']:>14.4f} {s_mb['prefill_s_per_query']:>14.4f}")
    print(f"{'e2e s/query':32s} {s_msa['e2e_s_per_query']:>14.4f} {s_mb['e2e_s_per_query']:>14.4f}")
    print(f"{'encode corpus (s)':32s} {msa['encode_corpus_s']:>14.3f} {mb['encode_corpus_s']:>14.3f}")
