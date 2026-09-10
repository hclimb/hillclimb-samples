# Two-Pass Top-K Memory Retrieval

The two-pass method achieves gradient-efficient training over large memory banks. Instead of backpropagating through a full `[B, T, N, M]` score matrix (expensive when M is large), it identifies the top-K indices without gradients (Pass 1), then recomputes scores only for those K entries with gradients enabled (Pass 2).

```mermaid
flowchart TD
    Q["Query q\n[B, T, N, H]"]
    MK["Memory Keys mem_k\n[M, H]"]
    MV["Memory Values mem_v\n[M, Dv]"]
    POS["pos_slot_indices\n[B, P]\n(ground-truth doc slots)"]

    subgraph P1["PASS 1 — stop_gradient (no gradients)"]
        direction TB
        SG["stop_gradient applied to q, mem_k"]
        CHUNK["Chunked scan over all M entries\nfor each chunk:\n  scores_chunk = q_sg @ mem_k_sg^T / sqrt(H)\n  update running top-K buffer"]
        IDX["top_k_indices [B, N, T, K]\n(stop_gradient)"]
        FULL["full_scores [B, T, N, M]\n(stop_gradient — saved for aux losses)"]
        SG --> CHUNK
        CHUNK --> IDX
        CHUNK --> FULL
    end

    subgraph P2["PASS 2 — sparse gradients flow"]
        direction TB
        NORM["mem_k_normed = rms_norm(mem_k)\n← gradient enabled"]
        GATHER["Sparse gather — K rows only\nmem_k_k = mem_k_normed[top_k_indices]  [B, N, T, K, H]\nmem_v_k = mem_v[top_k_indices]          [B, N, T, K, Dv]"]
        SCORES["logits_k = einsum(q, mem_k_k) / sqrt(H)  [B, N, T, K]"]
        SOFT["weights = softmax(logits_k, axis=-1)  [B, N, T, K]"]
        AGG["output = einsum(weights, mem_v_k)  [B, N, T, Dv]"]
        POSGATHER["Gather correct slots\nmem_k_pos = mem_k_normed[pos_slot_indices]  [B, P, H]"]
        POSLOGITS["pos_logits = einsum(q, mem_k_pos) / sqrt(H)  [B, N, T, P]\n(logits for ground-truth slots)"]
        NORM --> GATHER
        NORM --> POSGATHER
        GATHER --> SCORES
        SCORES --> SOFT
        SOFT --> AGG
        POSGATHER --> POSLOGITS
    end

    OUT["Output → W_o projection → LLM residual stream"]
    LOSS["doc_access_top_k_loss\n(retrieved logits vs correct slot logits)"]

    Q --> P1
    MK --> P1
    Q --> P2
    MK --> P2
    MV --> P2
    POS --> P2
    IDX -- "top_k_indices\n(which K slots)" --> P2
    P2 --> OUT
    SCORES -. "top-K logits" .-> LOSS
    POSLOGITS -. "correct slot logits" .-> LOSS
    FULL -. "full_scores\n(mem_uniform_kl, doc_access_loss)" .-> LOSS

    style P1 fill:#fff3cd,stroke:#d4a017,color:#000
    style P2 fill:#d4edda,stroke:#28a745,color:#000
    style LOSS fill:#f8d7da,stroke:#dc3545,color:#000
```

## Gradient flow

| | Target |
|---|---|
| **Flows** | `q`, `mem_k` (K selected rows only), `mem_v` (K selected rows only) |
| **Blocked** | Index selection in Pass 1 (full `stop_gradient` boundary) |
| **Zero grad** | The M−K unselected memory entries |

## Why two passes?

**Naive single-pass:** `scores = q @ mem_k^T` produces a `[B, T, N, M]` matrix. Storing and differentiating through it is O(M) per example — prohibitive at M=16,384+.

**Two-pass:** Pass 1 identifies the K relevant entries cheaply (chunked, no gradient storage). Pass 2 differentiates through a `[B, T, N, K]` matrix. With K=128 and M=16,384, gradient memory is ~128× smaller.

## Approximation

The softmax in Pass 2 normalizes over K entries only, not all M. Weights differ slightly from `softmax(full_scores)[K_indices]`. The error is small when K is large (128 here) because the top-K entries dominate the probability mass.
