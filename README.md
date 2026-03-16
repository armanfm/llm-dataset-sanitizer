# Terra Dourada — LLM Dataset Sanitizer

Detect and remove web scraping garbage from LLM training datasets.
No GPU. No neural model. No human labels. Pure Rust.

## The Problem

Web scraping contaminates LLM training datasets with:
- Navigation menus mixed with article content
- Copyright and spam inside real documents
- Multiple unrelated topics in one document
- Code from different languages mixed together

Existing filters (CCNet, FineWeb) need GPUs and human-labeled data.
Terra Dourada needs neither.

## How It Differs from FineWeb / CCNet

| | FineWeb | Terra Dourada |
|--|---------|---------------|
| Human labels needed | Yes | No |
| GPU to filter | Yes | No |
| Intra-doc rupture detection | No | Yes |
| Self-supervised | No | Yes |
| Model size | 500MB+ | 267KB |
| Language | Python | Rust |

## The Algorithm - FXL Turbo

Detects abrupt topic changes within documents using a temporal context window:

    ctx(t) = (1/N) x sum[ 1 - |sim(t) - sim(t-i)| ]

When ctx(t) < threshold for 2+ consecutive sentences, a rupture is detected.
Combined with lexical features: comp_med, pct_lixo, pct_nav, pct_cod.

## Quick Start

    git clone https://github.com/armanfm/llm-dataset-sanitizer
    cd llm-dataset-sanitizer/sanitizer
    cargo build --release

    ./target/release/tds --demo
    ./target/release/tds -i dataset.jsonl -o clean.jsonl
    ./target/release/tds -i dataset.jsonl -o clean.jsonl --threads 8

## Options

    -i, --input         Input dataset (.txt or .jsonl)
    -o, --output        Clean output file
    -r, --rejeitados    Save rejected docs (optional)
        --relatorio     Save JSON report (optional)
    -s, --sensibilidade Sensitivity 0.0-1.0 (default: 0.5)
        --threads       Parallel threads (default: all CPUs)
        --max-docs      Limit documents (0 = all)
        --codigo        Source code mode
        --demo          Built-in demo

Sensitivity presets:
    -s 0.2  permissive  ~10% rejected
    -s 0.5  balanced    ~25% rejected (default)
    -s 0.8  strict      ~40% rejected

## Results

Classifier experiment (2000 docs, self-supervised labels from FXL Turbo):

    Random Forest CV F1  : 100.0%
    Test accuracy        : 100.0%
    Unseen documents     : 100.0% (10/10)
    Model size           : 267KB

Throughput (8 threads, modern CPU):

    10k docs  -> 2s
    100k docs -> 20s
    1M docs   -> 3min
    100M docs -> 5h

## Architecture

    Raw Dataset
          |
          v
    Layer 1: Lexical Features
      comp_med, pct_lixo, pct_nav, pct_cod
          |
          v
    Layer 2: FXL Turbo
      ctx(t) = mean(1 - |sim(t) - sim(t-i)|)
      Detects abrupt topic changes
          |
      ---------
      |       |
    Clean   Rejected + reason

## Open Benchmark - GPU Help Wanted

Missing experiment: GPT-2 small on TinyStories raw vs sanitized.
If you have a GPU:

    pip install transformers datasets torch
    python experiments/benchmark_gpt2.py --docs 50000

## Citation

    @software{terra_dourada_2026,
      title  = {Terra Dourada: Self-Supervised LLM Dataset Sanitizer},
      author = {Armando, Recife, Brazil},
      year   = {2026},
      url    = {https://github.com/armanfm/llm-dataset-sanitizer}
    }

## License

MIT - use freely in research and commercial projects.

Built in Recife, Brazil.
"You do not mine the gold. You filter the soil to find where the gold is."
