"""
Terra Dourada - GPT-2 Benchmark v2
====================================
Compares:
  - Initial perplexity (before training) vs Final perplexity
  - GPT-2 trained on RAW data vs SANITIZED data

New in v2:
  - Shows learning curve (initial vs final perplexity)
  - Z-Score word rarity filter (detects scraping artifacts)
  - Safe for technical text (whitelist for domain vocabulary)

Requirements:
    pip install transformers datasets torch scikit-learn

Usage:
    python benchmark_gpt2.py
    python benchmark_gpt2.py --docs 50000 --real
    python benchmark_gpt2.py --zscore        (enable Z-Score filter)
    python benchmark_gpt2.py --zscore-only   (Z-Score only, no lexical)
"""

import os
import math
import json
import time
import argparse
from collections import Counter

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


# ================================================================
# LEXICAL FILTER (mirrors the Rust sanitizer)
# ================================================================

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


# ================================================================
# Z-SCORE WORD RARITY FILTER
# ================================================================

# Whitelist: technical terms that are rare but legitimate.
# Without this, words like 'mitochondrial', 'backpropagation', 'qubit'
# would be penalized as "suspicious rare tokens".
TECHNICAL_WHITELIST = {
    # Biology / Medicine
    'mitochondrial','transcription','methylation','genomic','cortical',
    'synaptic','hippocampal','neuronal','myelination','axonal',
    # ML / AI
    'backpropagation','autoregressive','tokenization','embedding',
    'softmax','transformer','perplexity','hyperparameter','gradient',
    'regularization','overfitting','convolutional','recurrent',
    # Physics / Chemistry
    'qubit','entanglement','photon','neutron','proton','thermodynamic',
    'electromagnetic','magnetization','crystalline','nucleotide',
    # Computer Science
    'recursion','parallelism','concurrency','serialization','polymorphism',
    'abstraction','encapsulation','deterministic','heuristic',
}


def build_corpus_stats(docs):
    """Build word frequency stats across the entire corpus."""
    total_counter = Counter()
    for doc in docs:
        total_counter.update(tokenize(doc))
    
    freqs = list(total_counter.values())
    n = len(freqs)
    if n == 0:
        return total_counter, 0.0, 1.0
    
    mean = sum(freqs) / n
    std  = math.sqrt(sum((f - mean) ** 2 for f in freqs) / n)
    return total_counter, mean, max(std, 1e-9)


def zscore_filter(docs, threshold=2.5, max_rare_pct=0.30):
    """
    Rejects documents where too many words have abnormal Z-Score.
    
    A word is "suspicious" if:
      z = (freq - mean) / std < -threshold   (extremely rare = scraping artifact)
      OR
      z > threshold * 3                       (spam word repeated excessively)
    
    BUT: words in TECHNICAL_WHITELIST are never penalized,
    so legitimate technical text is preserved.
    
    Args:
        threshold:    Z-Score cutoff for suspicion (default: 2.5)
        max_rare_pct: Max fraction of suspicious words before rejection (default: 30%)
    """
    word_freq, mean, std = build_corpus_stats(docs)
    
    clean = []
    rejected = 0
    
    for doc in docs:
        words = tokenize(doc)
        if len(words) < 5:
            clean.append(doc)
            continue
        
        suspicious = 0
        for w in words:
            if w in TECHNICAL_WHITELIST:
                continue  # Never penalize legitimate technical terms
            freq = word_freq.get(w, 0)
            z = (freq - mean) / std
            # Extremely rare (possible scraping artifact) OR excessively repeated (spam)
            if z < -threshold or z > threshold * 3:
                suspicious += 1
        
        pct_suspicious = suspicious / len(words)
        if pct_suspicious <= max_rare_pct:
            clean.append(doc)
        else:
            rejected += 1
    
    print(f"  Z-Score filter: {len(clean)} approved, {rejected} rejected "
          f"({rejected/len(docs)*100:.1f}%) | threshold={threshold}")
    return clean


# ================================================================
# COMBINED TERRA DOURADA FILTER
# ================================================================

def terra_dourada_filter(docs, sensitivity=0.5, use_zscore=False,
                         zscore_only=False):
    """
    Combined Terra Dourada filter.
    
    Layer 1 (Lexical): comp_med, pct_lixo, pct_nav
    Layer 2 (Z-Score): word rarity anomaly detection (optional)
    """
    result = list(docs)
    n_original = len(result)
    
    if not zscore_only:
        # Lexical filter
        comp_min     = 3.0 + sensitivity * 4.0
        pct_lixo_max = 0.15 - sensitivity * 0.10
        pct_nav_max  = 0.20 - sensitivity * 0.12
        
        clean_lex = []
        for doc in result:
            words = tokenize(doc)
            tot   = max(len(words), 1)
            pct_lixo = sum(1 for w in words if w in LIXO_WORDS) / tot
            pct_nav  = sum(1 for w in words if w in NAV_WORDS)  / tot
            sents    = [s.strip() for s in doc.replace('!','.').replace('?','.').split('.')
                        if len(s.strip()) > 5]
            comp_med = sum(len(tokenize(s)) for s in sents) / max(len(sents), 1)
            
            if (pct_lixo < pct_lixo_max and pct_nav < pct_nav_max
                    and comp_med >= comp_min):
                clean_lex.append(doc)
        
        rej_lex = len(result) - len(clean_lex)
        print(f"  Lexical filter : {len(clean_lex)} approved, {rej_lex} rejected "
              f"({rej_lex/len(result)*100:.1f}%)")
        result = clean_lex
    
    if use_zscore and len(result) > 10:
        result = zscore_filter(result)
    
    total_rejected = n_original - len(result)
    print(f"  Total          : {len(result)} approved, {total_rejected} rejected "
          f"({total_rejected/n_original*100:.1f}%) from {n_original} docs")
    return result


# ================================================================
# SYNTHETIC DATASET
# ================================================================

CLEAN_DOCS = [
    "Neural networks learn representations from large training datasets. Deep learning uses multiple layers to extract hierarchical features. Training involves minimizing a loss function through gradient descent. The optimizer updates weights based on computed gradients at each step.",
    "The water cycle describes how water moves continuously through the environment. Evaporation transforms liquid water into vapor that rises into the atmosphere. Clouds form when water vapor cools and condenses around tiny particles. Precipitation brings water back to the surface as rain or snow.",
    "Photosynthesis converts sunlight into chemical energy stored in glucose molecules. Chlorophyll in plant cells absorbs light in the red and blue spectrum. Carbon dioxide from air and water from soil are the raw materials used. The Calvin cycle uses ATP and NADPH to synthesize organic compounds.",
    "The human brain contains approximately eighty six billion neurons connected by synapses. Neural signals travel as electrical impulses along axons to neighboring cells. Memory consolidation occurs during sleep when the hippocampus replays experiences. The prefrontal cortex handles planning decision making and executive function.",
    "Rust is a systems programming language focused on safety and performance. Its ownership system prevents memory bugs at compile time without garbage collection. The borrow checker ensures references are always valid preventing data races. Zero cost abstractions allow high level code to compile to efficient machine code.",
    "Climate change is altering weather patterns across the globe significantly. Rising temperatures are causing glaciers and ice sheets to melt faster. Sea levels are rising and threatening coastal communities worldwide. International cooperation is essential to reduce greenhouse gas emissions globally.",
    "Vaccines have eliminated or controlled many deadly infectious diseases worldwide. The smallpox vaccine led to complete eradication of the disease globally. Herd immunity occurs when enough individuals are immune to stop spread. mRNA vaccines represent a new platform for rapid vaccine development.",
] * 30  # 210 docs

GARBAGE_DOCS = [
    "Home About Contact Privacy Policy Terms of Service Newsletter Subscribe. Click here to read more related articles on our website and platform. Follow us on Facebook Twitter Instagram YouTube for daily updates and news. Copyright 2024 All Rights Reserved advertisement sponsored content links.",
    "The Eiffel Tower stands three hundred meters tall in central Paris France. My cat refuses to eat dry food for the past three weeks now. Bitcoin reached its all time high price in November of twenty twenty one. Spaghetti carbonara is made with eggs cheese guanciale and black pepper.",
    "Error five hundred internal server error please try again later today. Subscribe for exclusive deals special offers available to members only. Page not found four oh four please check the URL and try again. Session timeout please log in again to continue your browsing session.",
    "SELECT star FROM users WHERE active equals one ORDER BY created at now. git commit message fix critical production bug and push to origin main. docker run detached port eight zero eight zero nginx latest version. import pandas as pd dataframe read csv data head display rows.",
    "Advertisement buy premium product now with free shipping included today. Sponsored post this article was created in partnership with a major brand. Like and subscribe for more amazing tutorials every single week daily. Terms conditions apply see website for complete details and restrictions.",
] * 40  # 200 docs


def get_dataset(n_docs=400, use_real=False):
    if use_real:
        try:
            from datasets import load_dataset
            print("  Loading TinyStories...")
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


# ================================================================
# GPT-2 TRAINING — with initial perplexity measurement
# ================================================================

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.items = []
        for text in texts:
            enc = tokenizer(text, truncation=True, max_length=max_len,
                            padding='max_length', return_tensors='pt')
            self.items.append({k: v.squeeze() for k, v in enc.items()})
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def calc_perplexity(model, tokenizer, docs, batch_size=8, label=""):
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for i in range(0, min(len(docs), 200), batch_size):
            batch = docs[i:i+batch_size]
            enc = tokenizer(batch, truncation=True, max_length=128,
                            padding=True, return_tensors='pt')
            ids  = enc['input_ids'].to(device)
            mask = enc['attention_mask'].to(device)
            out  = model(input_ids=ids, attention_mask=mask, labels=ids)
            total_loss += out.loss.item()
            n += 1
    avg_loss = total_loss / max(n, 1)
    ppl = math.exp(avg_loss)
    if label:
        print(f"  {label:<30} loss={avg_loss:.4f}  ppl={ppl:.2f}")
    return ppl, avg_loss


def train_gpt2(train_docs, val_docs, label, epochs=3):
    print(f"\n  === GPT-2 [{label}] | {len(train_docs)} train docs ===")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    config = GPT2Config(n_embd=128, n_layer=2, n_head=2,
                        n_positions=128, vocab_size=tokenizer.vocab_size)
    model = GPT2LMHeadModel(config)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model: {params:.1f}M parameters")

    # === INITIAL perplexity (before any training) ===
    ppl_initial, loss_initial = calc_perplexity(
        model, tokenizer, val_docs, label="Initial (untrained)")

    train_ds = TextDataset(train_docs, tokenizer)
    collator  = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=f'/tmp/td_gpt2_{label.replace(" ","_")}',
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        learning_rate=5e-4,
        logging_steps=50,
        save_steps=9999,
        report_to='none',
        no_cuda=not torch.cuda.is_available(),
    )

    t0 = time.time()
    Trainer(model=model, args=args, train_dataset=train_ds,
            data_collator=collator).train()
    elapsed = time.time() - t0

    # === FINAL perplexity (after training) ===
    ppl_final, loss_final = calc_perplexity(
        model, tokenizer, val_docs, label="Final (trained)")

    # Learning curve: how much did training improve?
    improvement = (ppl_initial - ppl_final) / ppl_initial * 100
    print(f"  Learning curve: {ppl_initial:.2f} -> {ppl_final:.2f} "
          f"({improvement:+.1f}% improvement in {elapsed:.0f}s)")

    return {
        "ppl_initial": round(ppl_initial, 4),
        "ppl_final":   round(ppl_final, 4),
        "loss_final":  round(loss_final, 4),
        "improvement": round(improvement, 2),
        "n_train":     len(train_docs),
        "time_s":      round(elapsed, 1),
    }


# ================================================================
# MAIN BENCHMARK
# ================================================================

def run_benchmark(n_docs=400, use_real=False, sensitivity=0.5,
                  use_zscore=False, zscore_only=False):
    print("=" * 65)
    print("  TERRA DOURADA - GPT-2 Benchmark v2")
    print("  Initial perplexity | Final perplexity | Learning curve")
    print("=" * 65)

    # 1. Dataset
    print(f"\n  [1/4] Loading dataset ({n_docs} docs, real={use_real})...")
    docs  = get_dataset(n_docs, use_real)
    split = int(len(docs) * 0.8)
    train_raw = docs[:split]
    val_docs  = docs[split:]
    print(f"  Train: {len(train_raw)} | Val: {len(val_docs)}")

    # 2. Sanitize
    print(f"\n  [2/4] Sanitizing (sensitivity={sensitivity}, "
          f"zscore={use_zscore})...")
    train_clean = terra_dourada_filter(
        train_raw, sensitivity, use_zscore, zscore_only)

    if not HAS_TORCH:
        print("\n  torch/transformers not installed.")
        print("  Run: pip install transformers torch")
        return

    # 3. Train two models (RAW and CLEAN)
    print("\n  [3/4] Training models...")
    stats_raw   = train_gpt2(train_raw,   val_docs, "RAW")
    stats_clean = train_gpt2(train_clean, val_docs, "CLEAN")

    # 4. Results
    delta_initial = stats_raw["ppl_initial"] - stats_clean["ppl_initial"]
    delta_final   = stats_raw["ppl_final"]   - stats_clean["ppl_final"]
    delta_curve   = stats_clean["improvement"] - stats_raw["improvement"]

    print(f"\n{'=' * 65}")
    print("  RESULTS")
    print(f"{'=' * 65}")
    print(f"  {'Metric':<35} {'RAW':>10}  {'CLEAN':>10}  {'Delta':>8}")
    print(f"  {'-' * 65}")
    print(f"  {'Training docs':<35} {stats_raw['n_train']:>10}  "
          f"{stats_clean['n_train']:>10}  "
          f"{stats_clean['n_train']-stats_raw['n_train']:>+8}")
    print(f"  {'Initial perplexity (untrained)':<35} "
          f"{stats_raw['ppl_initial']:>10.2f}  "
          f"{stats_clean['ppl_initial']:>10.2f}  "
          f"{delta_initial:>+8.2f}")
    print(f"  {'Final perplexity (after training)':<35} "
          f"{stats_raw['ppl_final']:>10.2f}  "
          f"{stats_clean['ppl_final']:>10.2f}  "
          f"{delta_final:>+8.2f}")
    print(f"  {'Learning improvement %':<35} "
          f"{stats_raw['improvement']:>9.1f}%  "
          f"{stats_clean['improvement']:>9.1f}%  "
          f"{delta_curve:>+7.1f}%")
    print(f"  {'Training time':<35} "
          f"{stats_raw['time_s']:>9.1f}s  "
          f"{stats_clean['time_s']:>9.1f}s")
    print(f"  {'-' * 65}")

    # Verdict
    print()
    if delta_final > 0:
        print(f"  PERPLEXITY:  CLEAN model is BETTER by {delta_final:.2f} points")
    else:
        print(f"  PERPLEXITY:  Neutral ({delta_final:.2f}) - try larger real dataset")

    if delta_curve > 0:
        print(f"  LEARNING:    CLEAN model learned FASTER (+{delta_curve:.1f}% curve)")
    else:
        print(f"  LEARNING:    Similar learning speed ({delta_curve:.1f}%)")

    # Save
    result = {
        "raw":   stats_raw,
        "clean": stats_clean,
        "delta_final_ppl":   round(delta_final, 4),
        "delta_learning_pct": round(delta_curve, 2),
        "config": {
            "n_docs": n_docs,
            "use_real": use_real,
            "sensitivity": sensitivity,
            "use_zscore": use_zscore,
        }
    }
    os.makedirs('results', exist_ok=True)
    with open('results/benchmark_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: results/benchmark_result.json")
    print(f"{'=' * 65}")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Terra Dourada GPT-2 Benchmark v2')
    parser.add_argument('--docs',         type=int,   default=400,
                        help='Number of documents')
    parser.add_argument('--real',         action='store_true',
                        help='Use real TinyStories dataset')
    parser.add_argument('--sensitivity',  type=float, default=0.5,
                        help='Lexical filter sensitivity 0.0-1.0')
    parser.add_argument('--zscore',       action='store_true',
                        help='Enable Z-Score rarity filter')
    parser.add_argument('--zscore-only',  action='store_true',
                        help='Use only Z-Score filter (skip lexical)')
    parser.add_argument('--epochs',       type=int,   default=3,
                        help='Training epochs per model')
    args = parser.parse_args()

    run_benchmark(
        n_docs=args.docs,
        use_real=args.real,
        sensitivity=args.sensitivity,
        use_zscore=args.zscore,
        zscore_only=args.zscore_only,
    )
