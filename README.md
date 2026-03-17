# Terra Dourada — LLM Dataset Sanitizer
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a8d87a03-5696-4a9f-9203-54f9ca71ec54" />


Detect and remove web scraping garbage from LLM training datasets.
No GPU. No neural model. No human labels. Pure Rust.

## The Problem

Web scraping contaminates LLM training datasets with:

- Navigation menus mixed with article content
- Copyright footers and spam inside real documents
- Multiple unrelated topics in one document
- Code from different languages mixed together

Existing filters (CCNet, FineWeb) need GPUs and human-labeled data.
Terra Dourada needs neither.

## How It Differs from FineWeb / CCNet

|  | FineWeb | Terra Dourada |
|--|---------|---------------|
| Human labels needed | Yes | No |
| GPU to filter | Yes | No |
| Intra-doc rupture detection | No | Yes |
| Self-supervised | No | Yes |
| Model size | 500MB+ | 267KB |
| Language | Python | Rust |

## Benchmark Results ✅

Experiment: GPT-2 trained on **Wikipedia (WikiText-2) with injected garbage** vs **sanitized version**.

Validation set: 800 clean Wikipedia paragraphs (same for both models — fair comparison).

```
Dataset: WikiText-2 (real Wikipedia) + 40% garbage injected
Garbage types: nav menus, random topics, SEO farms, boilerplate HTML

Terra Dourada rejected: 252 docs (5.6% of training set)

Model                          Docs    PPL Final    Learning curve
─────────────────────────────────────────────────────────────────
GPT-2 (raw, with garbage)      4480    2057.81      +95.5%
GPT-2 (Terra Dourada, clean)   4228    1721.54      +96.3%
─────────────────────────────────────────────────────────────────
Delta perplexity : -336.27 points  ← lower is better
Delta curve      : +0.8%
```

**Result: removing only 5.6% of the worst documents reduced perplexity by 336 points.**
The clean model also learned faster (+0.8% learning curve improvement).

## The Algorithm - FXL Turbo

Detects abrupt topic changes within documents using a temporal context window:

```
ctx(t) = (1/N) x sum[ 1 - |sim(t) - sim(t-i)| ]
```

When ctx(t) < threshold for 2+ consecutive sentences, a rupture is detected.
Combined with lexical features: comp_med, pct_lixo, pct_nav, pct_cod.

## Quick Start

```bash
git clone https://github.com/armanfm/llm-dataset-sanitizer
cd llm-dataset-sanitizer/sanitizer
cargo build --release

./target/release/tds --demo
./target/release/tds -i dataset.jsonl -o clean.jsonl
./target/release/tds -i dataset.jsonl -o clean.jsonl --threads 8
```

## Options

```
-i, --input       Input dataset (.txt or .jsonl)
-o, --output      Clean output file
-r, --rejeitados  Save rejected docs (optional)
    --relatorio   Save JSON report (optional)
-s, --sensibilidade  Sensitivity 0.0-1.0 (default: 0.5)
    --threads     Parallel threads (default: all CPUs)
    --max-docs    Limit documents (0 = all)
    --codigo      Source code mode
    --demo        Built-in demo
```

Sensitivity presets:
- `-s 0.2`  permissive  ~10% rejected
- `-s 0.5`  balanced    ~25% rejected (default)
- `-s 0.8`  strict      ~40% rejected

## Reproduce the Benchmark

```bash
pip install transformers accelerate datasets torch
python benchmark/benchmark_gpt2.py
```

For larger scale (needs internet, ~30min):
```bash
python benchmark/benchmark_gpt2.py --docs 10000 --epochs 5
```

## Architecture

```
Raw Dataset
    |
    v  Layer 1: Lexical Features
       comp_med, pct_lixo, pct_nav, pct_cod
    |
    v  Layer 2: FXL Turbo
       ctx(t) = mean(1 - |sim(t) - sim(t-i)|)
       Detects abrupt topic changes
    |
    ---------
    |       |
  Clean   Rejected + reason
```

## Throughput (8 threads, modern CPU)

```
10k docs   →  2s
100k docs  →  20s
1M docs    →  3min
100M docs  →  5h
```

## Open Benchmark — GPU Help Wanted

Want to run at full scale on real Common Crawl data?

```bash
pip install transformers datasets torch
python benchmark/benchmark_gpt2.py --real --docs 50000
```

This requires a GPU and downloads TinyStories (~1GB).
Expected result: ~5-15% perplexity improvement at scale.

## Citation

```
@software{terra_dourada_2026,
  title  = {Terra Dourada: Self-Supervised LLM Dataset Sanitizer},
  author = {Armando, Recife, Brazil},
  year   = {2026},
  url    = {https://github.com/armanfm/llm-dataset-sanitizer}
}
```

## License

MIT - use freely in research and commercial projects.

Built in Recife, Brazil.
"You do not mine the gold. You filter the soil to find where the gold is."
