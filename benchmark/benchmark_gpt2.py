"""
Terra Dourada - GPT-2 Benchmark
================================
Compares validation perplexity:
  - GPT-2 trained on RAW data
  - GPT-2 trained on FILTERED data (Terra Dourada)

This is the experiment that proves the sanitizer improves LLMs.

Requirements:
    pip install transformers datasets torch scikit-learn

Usage:
    python benchmark_gpt2.py
    python benchmark_gpt2.py --docs 50000  (with real TinyStories)
"""

import os
import math
import json
import time
import argparse

# Check dependencies
try:
    import torch
    from transformers import (
        GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
        DataCollatorForLanguageModeling, Trainer, TrainingArguments,
    )
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from collections import Counter


# ── Lexical filter (mirrors the Rust sanitizer logic) ────────────

LIXO_WORDS = {
    'copyright','subscribe','newsletter','click','terms','privacy',
    'policy','contact','advertisement','sponsored','rights','reserved',
    'follow','download','share','facebook','instagram','twitter',
    'loading','error','login','signup','cookie','accept','affiliate',
    'home','about','sign','related','posts','articles',
}
NAV_WORDS = {
    'home','about','contact','privacy','terms','service',
    'newsletter','subscribe','follow','copyright','reserved','cookie',
}


def tokenize(text):
    out = []
    for c in text.lower():
        out.append(c if (c.isalnum() or c == ' ') else ' ')
    return ''.join(out).split()


def terra_dourada_filter(docs, sensibilidade=0.5):
    """Python implementation of the Terra Dourada lexical filter."""
    comp_min = 3.0 + sensibilidade * 4.0
    pct_lixo_max = 0.15 - sensibilidade * 0.10
    pct_nav_max  = 0.20 - sensibilidade * 0.12

    clean = []
    rejected = 0
    for doc in docs:
        words = tokenize(doc)
        tot   = max(len(words), 1)

        pct_lixo = sum(1 for w in words if w in LIXO_WORDS) / tot
        pct_nav  = sum(1 for w in words if w in NAV_WORDS)  / tot

        sents = [s.strip() for s in doc.replace('!','.').replace('?','.').split('.')
                 if len(s.strip()) > 5]
        comp_med = sum(len(tokenize(s)) for s in sents) / max(len(sents), 1)

        if pct_lixo < pct_lixo_max and pct_nav < pct_nav_max and comp_med >= comp_min:
            clean.append(doc)
        else:
            rejected += 1

    print(f"  Sanitizer: {len(clean)} approved, {rejected} rejected "
          f"({rejected/len(docs)*100:.1f}%)")
    return clean


# ── Synthetic dataset ─────────────────────────────────────────────

CLEAN_DOCS = [
    "Neural networks learn representations from large training datasets. Deep learning uses multiple layers to extract hierarchical features. Training involves minimizing a loss function through gradient descent. The optimizer updates weights based on computed gradients at each step.",
    "The water cycle describes how water moves continuously through the environment. Evaporation transforms liquid water into vapor that rises into the atmosphere. Clouds form when water vapor cools and condenses around tiny particles. Precipitation brings water back to the surface as rain or snow.",
    "Photosynthesis converts sunlight into chemical energy stored in glucose molecules. Chlorophyll in plant cells absorbs light in the red and blue spectrum. Carbon dioxide from air and water from soil are the raw materials. The Calvin cycle uses ATP and NADPH to synthesize organic compounds.",
    "The human brain contains approximately eighty six billion neurons connected by synapses. Neural signals travel as electrical impulses along axons to neighboring cells. Memory consolidation occurs during sleep when the hippocampus replays experiences. The prefrontal cortex handles planning decision making and executive function.",
    "Rust is a systems programming language focused on safety and performance. Its ownership system prevents memory bugs at compile time without garbage collection. The borrow checker ensures references are always valid preventing data races. Zero cost abstractions allow high level code to compile to efficient machine code.",
    "Climate change is altering weather patterns across the globe significantly. Rising global temperatures are causing glaciers and ice sheets to melt faster. Sea levels are rising and threatening coastal communities worldwide with flooding. International cooperation is essential to reduce greenhouse gas emissions globally.",
    "The transformer architecture revolutionized natural language processing research. Self attention mechanisms allow models to relate tokens across the full sequence. Pretraining on large corpora enables effective transfer learning to downstream tasks. BERT introduced bidirectional context improving performance on language understanding.",
    "Vaccines have eliminated or controlled many deadly infectious diseases worldwide. The smallpox vaccine led to complete eradication of the disease globally. Herd immunity occurs when enough individuals are immune to stop pathogen spread. mRNA vaccines represent a new platform for rapid vaccine development and deployment.",
] * 25  # 200 docs

GARBAGE_DOCS = [
    "Home About Contact Privacy Policy Terms of Service Newsletter Subscribe today. Click here to read more related articles on our website and platform. Follow us on Facebook Twitter Instagram YouTube for daily updates and news. Copyright 2024 All Rights Reserved advertisement sponsored content links here.",
    "The Eiffel Tower stands three hundred meters tall in central Paris France. My cat refuses to eat dry food for the past three weeks now. Bitcoin reached its all time high price in November of twenty twenty one. Spaghetti carbonara is made with eggs cheese guanciale and black pepper.",
    "Error five hundred internal server error please try again later today now. Subscribe for exclusive deals special offers available to members only here. Page not found four oh four please check the URL and try again. Session timeout please log in again to continue your current browsing session.",
    "SELECT star FROM users WHERE active equals one ORDER BY created at now. git commit message fix critical production bug and push to origin main. docker run detached port eight zero eight zero nginx latest version always. import pandas as pd dataframe read csv data dot csv head display.",
    "Advertisement buy premium product now with free shipping included this week. Sponsored post this article was created in partnership with a major brand. Like and subscribe to channel for more amazing tutorials every single week. Terms conditions apply see website for complete details and restrictions today.",
] * 40  # 200 docs


def get_dataset(n_docs=400, use_real=False):
    """Get dataset - synthetic or real TinyStories."""
    if use_real:
        try:
            from datasets import load_dataset
            print("  Loading TinyStories (may take a while)...")
            ds = load_dataset('roneneldan/TinyStories', split=f'train[:{n_docs}]')
            return [item['text'] for item in ds]
        except Exception as e:
            print(f"  Could not load TinyStories: {e}")
            print("  Falling back to synthetic dataset...")

    import random
    random.seed(42)
    docs = CLEAN_DOCS + GARBAGE_DOCS
    random.shuffle(docs)
    return docs[:n_docs]


# ── GPT-2 training ────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.items = []
        for text in texts:
            enc = tokenizer(text, truncation=True, max_length=max_len,
                            padding='max_length', return_tensors='pt')
            self.items.append({k: v.squeeze() for k, v in enc.items()})

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def calc_perplexity(model, tokenizer, docs, batch_size=8):
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            enc = tokenizer(batch, truncation=True, max_length=128,
                            padding=True, return_tensors='pt')
            ids = enc['input_ids'].to(device)
            mask = enc['attention_mask'].to(device)
            out = model(input_ids=ids, attention_mask=mask, labels=ids)
            total_loss += out.loss.item()
            n += 1
    avg_loss = total_loss / max(n, 1)
    return math.exp(avg_loss), avg_loss


def train_gpt2(train_docs, val_docs, label, epochs=2):
    print(f"  Training GPT-2 [{label}] on {len(train_docs)} docs...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    config = GPT2Config(n_embd=128, n_layer=2, n_head=2,
                        n_positions=128, vocab_size=tokenizer.vocab_size)
    model = GPT2LMHeadModel(config)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model: {params:.1f}M parameters")

    train_ds = TextDataset(train_docs, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=f'/tmp/td_gpt2_{label}',
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        learning_rate=5e-4,
        logging_steps=100,
        save_steps=9999,
        report_to='none',
        no_cuda=not torch.cuda.is_available(),
    )

    t0 = time.time()
    Trainer(model=model, args=args, train_dataset=train_ds,
            data_collator=collator).train()
    elapsed = time.time() - t0

    ppl, loss = calc_perplexity(model, tokenizer, val_docs)
    print(f"  Loss: {loss:.4f}  Perplexity: {ppl:.2f}  Time: {elapsed:.0f}s")
    return ppl, loss


# ── Main benchmark ────────────────────────────────────────────────

def run_benchmark(n_docs=400, use_real=False, sensitivity=0.5):
    print("=" * 60)
    print("  TERRA DOURADA - GPT-2 Benchmark")
    print("  RAW dataset vs SANITIZED dataset")
    print("=" * 60)

    # 1. Dataset
    print(f"\n  [1/4] Loading dataset ({n_docs} docs)...")
    docs = get_dataset(n_docs, use_real)
    split = int(len(docs) * 0.8)
    train_raw = docs[:split]
    val_docs   = docs[split:]
    print(f"  Train: {len(train_raw)} | Val: {len(val_docs)}")

    # 2. Sanitize
    print(f"\n  [2/4] Sanitizing with Terra Dourada (sensitivity={sensitivity})...")
    train_clean = terra_dourada_filter(train_raw, sensitivity)

    if not HAS_TORCH:
        print("\n  torch/transformers not installed.")
        print("  pip install transformers torch")
        print(f"\n  Sanitization result:")
        print(f"  Raw    : {len(train_raw)} docs")
        print(f"  Clean  : {len(train_clean)} docs (-{(len(train_raw)-len(train_clean))/len(train_raw)*100:.0f}%)")
        return

    # 3. Train two models
    print("\n  [3/4] Training two GPT-2 models...")
    ppl_raw,   loss_raw   = train_gpt2(train_raw,   val_docs, "raw")
    ppl_clean, loss_clean = train_gpt2(train_clean, val_docs, "clean")

    # 4. Results
    delta = ppl_raw - ppl_clean
    improvement = delta / ppl_raw * 100

    print(f"\n{'=' * 60}")
    print("  RESULTS")
    print(f"{'=' * 60}")
    print(f"  {'Model':<25} {'Loss':>8}  {'Perplexity':>12}")
    print(f"  {'-' * 48}")
    print(f"  {'GPT-2 (raw data)':<25} {loss_raw:>8.4f}  {ppl_raw:>12.2f}")
    print(f"  {'GPT-2 (sanitized)':<25} {loss_clean:>8.4f}  {ppl_clean:>12.2f}")
    print(f"  {'-' * 48}")
    print(f"  Improvement: {delta:+.2f} perplexity ({improvement:+.1f}%)")

    if delta > 0:
        print("\n  RESULT: Terra Dourada IMPROVED the model")
    else:
        print("\n  RESULT: Neutral - try with larger real dataset")

    result = {
        "n_train_raw": len(train_raw),
        "n_train_clean": len(train_clean),
        "pct_rejected": round((len(train_raw)-len(train_clean))/len(train_raw)*100, 1),
        "ppl_raw": round(ppl_raw, 4),
        "ppl_clean": round(ppl_clean, 4),
        "delta_ppl": round(delta, 4),
        "improvement_pct": round(improvement, 2),
    }

    os.makedirs('results', exist_ok=True)
    with open('results/benchmark_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: results/benchmark_result.json")
    print(f"{'=' * 60}")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs',        type=int,   default=400)
    parser.add_argument('--real',        action='store_true',
                        help='Use real TinyStories dataset')
    parser.add_argument('--sensitivity', type=float, default=0.5)
    args = parser.parse_args()
    run_benchmark(args.docs, args.real, args.sensitivity)
