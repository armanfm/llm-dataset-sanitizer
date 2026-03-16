"""
Terra Dourada — Benchmark GPT-2 (versão universal)
=====================================================
Funciona com QUALQUER versão do transformers.
Detecta automaticamente GPU/CPU e argumentos corretos.

RODA:
    python terra_dourada_benchmark_local.py

Tempo estimado: 5-15 minutos dependendo do PC.
"""

import math, random, time, json, sys
from collections import Counter

random.seed(42)

# ── Verifica dependências ──────────────────────────────────────────
try:
    import torch
    from torch.utils.data import Dataset
    import transformers
    from transformers import (
        GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
        DataCollatorForLanguageModeling, Trainer, TrainingArguments,
    )
    print("✓ Dependências OK")
    print(f"  torch        : {torch.__version__}")
    print(f"  transformers : {transformers.__version__}")
    print(f"  GPU          : {torch.cuda.is_available()}")
except ImportError as e:
    print(f"ERRO: {e}")
    print("\nInstale com:")
    print("  pip install transformers accelerate torch")
    sys.exit(1)


# ================================================================
# DETECTA argumento correto do TrainingArguments
# (no_cuda foi deprecated, use_cpu é o novo)
# ================================================================

def criar_training_args(output_dir, epochs, batch_size, lr):
    """Cria TrainingArguments compatível com qualquer versão."""
    usar_gpu = torch.cuda.is_available()
    base = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=999,  # silencia logs de treino
        save_steps=9999,
        report_to='none',
    )
    # Tenta use_cpu primeiro (versões novas), depois no_cuda (versões antigas)
    for arg, val in [('use_cpu', not usar_gpu), ('no_cuda', not usar_gpu)]:
        try:
            args = TrainingArguments(**base, **{arg: val})
            return args
        except TypeError:
            continue
    # Fallback sem argumento de CPU/GPU
    return TrainingArguments(**base)


# ================================================================
# FILTRO TERRA DOURADA
# ================================================================

SPAM = {
    'copyright','subscribe','newsletter','click','terms','privacy',
    'policy','contact','rights','reserved','follow','download',
    'share','facebook','instagram','login','signup','cookie',
    'home','about','advertisement','sponsored','free','offer',
    'today','now','limited','deal','promo','buy','sale',
}

def tokens(t):
    o = []
    for c in t.lower():
        o.append(c if (c.isalnum() or c == ' ') else ' ')
    return ''.join(o).split()

def sanitizar(docs):
    clean, rej = [], 0
    for doc in docs:
        ws  = tokens(doc); tot = max(len(ws), 1)
        pct = sum(1 for w in ws if w in SPAM) / tot
        ss  = [s.strip() for s in doc.split('.') if len(s.strip()) > 5]
        comp = sum(len(tokens(s)) for s in ss) / max(len(ss), 1)
        if pct < 0.10 and comp >= 5.0:
            clean.append(doc)
        else:
            rej += 1
    n = len(docs)
    print(f"  Terra Dourada: {len(clean)} aprovados, {rej} rejeitados ({rej/n*100:.1f}%)")
    return clean


# ================================================================
# DATASET SINTÉTICO
# ================================================================

LIMPOS = [
    "Neural networks learn representations from large training datasets. Deep learning uses multiple layers to extract hierarchical features. Training involves minimizing a loss function through gradient descent. The optimizer updates weights based on computed gradients at each step.",
    "The water cycle describes how water moves continuously through the environment. Evaporation transforms liquid water into vapor that rises into the atmosphere. Clouds form when water vapor cools and condenses around tiny particles. Precipitation brings water back to the surface as rain or snow.",
    "Photosynthesis converts sunlight into chemical energy stored in glucose molecules. Chlorophyll in plant cells absorbs light in the red and blue spectrum. Carbon dioxide from air and water from soil are the raw materials needed. The Calvin cycle uses ATP and NADPH to synthesize organic compounds.",
    "The human brain contains approximately eighty six billion neurons connected by synapses. Neural signals travel as electrical impulses along axons to neighboring cells. Memory consolidation occurs during sleep when the hippocampus replays experiences. The prefrontal cortex handles planning, decision making, and executive function.",
    "Rust is a systems programming language focused on safety and performance. Its ownership system prevents memory bugs at compile time without garbage collection. The borrow checker ensures references are always valid, preventing data races. Zero cost abstractions allow high level code to compile to efficient machine code.",
    "Climate change is altering weather patterns across the globe significantly. Rising temperatures are causing glaciers and ice sheets to melt faster each year. Sea levels are rising and threatening coastal communities worldwide with flooding. International cooperation is essential to reduce greenhouse gas emissions effectively.",
    "Vaccines have eliminated or controlled many deadly infectious diseases worldwide. The smallpox vaccine led to the complete eradication of the disease globally. Herd immunity occurs when enough individuals are immune to stop pathogen spread. mRNA vaccines represent a new platform for rapid and effective vaccine development.",
    "DNA encodes genetic information in sequences of four nucleotide bases precisely. Genes are transcribed into RNA which is then translated into proteins inside cells. Mutations can alter protein function and affect cellular processes significantly over time. Evolution occurs when beneficial mutations spread through a population across generations.",
]

LIXO_DOCS = [
    "Home About Contact Privacy Policy Terms of Service Newsletter Subscribe today. Click here to read more related articles on our website and platform today. Follow us on Facebook Twitter Instagram YouTube for daily updates and promotions now. Copyright 2024 All Rights Reserved advertisement sponsored content affiliate links here.",
    "The Eiffel Tower stands three hundred meters tall in central Paris today. My cat refuses to eat dry food for the past three weeks now always here. Bitcoin reached its all time high price in November of twenty twenty one. Spaghetti carbonara is made with eggs cheese guanciale and black pepper traditionally.",
    "Subscribe for exclusive deals and special offers available to members only today. Page not found please check the URL and try again later today please. Session timeout please log in again to continue your current browsing session now. Advertisement buy premium product now with free shipping included this week.",
    "Error five hundred internal server error please try again later today now here. Like and subscribe to our channel for more amazing content every single week. Terms and conditions apply see website for complete details and restrictions today. Sponsored post created in partnership with a major brand for our readers now.",
]

def get_dataset(n=800):
    limpos = (LIMPOS * 200)[:n//2]
    lixo   = (LIXO_DOCS * 200)[:n//2]
    docs   = limpos + lixo
    random.shuffle(docs)
    return docs


# ================================================================
# GPT-2 TRAINING
# ================================================================

class TextDS(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.items = []
        for text in texts:
            enc = tokenizer(text, truncation=True, max_length=max_len,
                            padding='max_length', return_tensors='pt')
            self.items.append({k: v.squeeze() for k, v in enc.items()})
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def calc_ppl(model, tokenizer, docs, bs=8):
    model.eval()
    device = next(model.parameters()).device
    total, n = 0., 0
    with torch.no_grad():
        for i in range(0, min(len(docs), 200), bs):
            enc  = tokenizer(docs[i:i+bs], truncation=True, max_length=128,
                             padding=True, return_tensors='pt')
            ids  = enc['input_ids'].to(device)
            mask = enc['attention_mask'].to(device)
            out  = model(input_ids=ids, attention_mask=mask, labels=ids)
            total += out.loss.item(); n += 1
    avg = total / max(n, 1)
    return math.exp(avg), avg


def treinar(train_docs, val_docs, label, epochs=3):
    print(f"\n  ── GPT-2 [{label}] | {len(train_docs)} docs de treino ──")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Modelo pequeno: roda em CPU em minutos
    config = GPT2Config(n_embd=128, n_layer=2, n_head=2,
                        n_positions=128, vocab_size=tokenizer.vocab_size)
    model  = GPT2LMHeadModel(config)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Modelo: {n_params:.1f}M parâmetros")

    # Perplexidade ANTES do treino
    ppl_i, loss_i = calc_ppl(model, tokenizer, val_docs)
    print(f"  Inicial (sem treino): ppl={ppl_i:.1f}")

    # Treino
    train_ds = TextDS(train_docs, tokenizer)
    collator  = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    args      = criar_training_args(f'./tmp_{label}', epochs, 8, 5e-4)

    t0 = time.time()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silencia FutureWarning do no_cuda
        Trainer(model=model, args=args,
                train_dataset=train_ds, data_collator=collator).train()
    elapsed = time.time() - t0

    # Perplexidade DEPOIS do treino
    ppl_f, loss_f = calc_ppl(model, tokenizer, val_docs)
    curva = (ppl_i - ppl_f) / ppl_i * 100
    print(f"  Final   (treinado):   ppl={ppl_f:.1f}  "
          f"(curva {curva:+.1f}%, {elapsed:.0f}s)")

    return {
        'ppl_i': round(ppl_i, 2), 'ppl_f': round(ppl_f, 2),
        'loss':  round(loss_f, 4), 'curva': round(curva, 2),
        'n':     len(train_docs),  'tempo': round(elapsed, 1),
    }


# ================================================================
# BENCHMARK PRINCIPAL
# ================================================================

def benchmark(n_docs=800, epochs=3):
    print()
    print("=" * 58)
    print("  TERRA DOURADA — Benchmark GPT-2")
    print("  Dataset BRUTO vs Dataset SANITIZADO")
    print("=" * 58)

    # 1. Dataset
    print(f"\n  [1/4] Dataset: {n_docs} documentos...")
    docs  = get_dataset(n_docs)
    split = int(len(docs) * 0.8)
    train_raw = docs[:split]; val = docs[split:]
    print(f"  Treino: {len(train_raw)} | Validação: {len(val)}")

    # 2. Sanitiza
    print(f"\n  [2/4] Sanitizando com Terra Dourada...")
    train_clean = sanitizar(train_raw)
    pct = (len(train_raw)-len(train_clean)) / len(train_raw) * 100
    print(f"  Redução: {len(train_raw)} → {len(train_clean)} docs (-{pct:.1f}%)")

    # 3. Treina dois modelos
    print(f"\n  [3/4] Treinando dois GPT-2 ({epochs} épocas cada)...")
    r_raw   = treinar(train_raw,   val, "RAW",   epochs)
    r_clean = treinar(train_clean, val, "CLEAN", epochs)

    # 4. Resultado
    delta_ppl   = r_raw['ppl_f'] - r_clean['ppl_f']
    delta_curva = r_clean['curva'] - r_raw['curva']

    print(f"\n{'=' * 58}")
    print("  RESULTADO FINAL")
    print(f"{'=' * 58}")
    print(f"\n  {'Modelo':<30} {'Docs':>5}  {'PPL Final':>10}  {'Curva':>8}")
    print(f"  {'─' * 56}")
    print(f"  {'GPT-2 (bruto)':<30} {r_raw['n']:>5}  "
          f"{r_raw['ppl_f']:>10.2f}  {r_raw['curva']:>+7.1f}%")
    print(f"  {'GPT-2 (Terra Dourada)':<30} {r_clean['n']:>5}  "
          f"{r_clean['ppl_f']:>10.2f}  {r_clean['curva']:>+7.1f}%")
    print(f"  {'─' * 56}")
    print(f"\n  Delta perplexidade : {delta_ppl:+.2f} pontos")
    print(f"  Delta curva        : {delta_curva:+.1f} pontos %")

    print()
    if delta_ppl > 0.5:
        print("  ✅ TERRA DOURADA MELHOROU O MODELO!")
        print(f"  Modelo treinado em dados limpos teve perplexidade")
        print(f"  {delta_ppl:.2f} pontos menor — aprende melhor sem lixo.")
    elif delta_ppl > -0.5:
        print("  ➡️  Resultado neutro com dataset sintético.")
        print("  Use --real --docs 5000 para resultado com dados reais.")
    else:
        print("  ⚠️  Tente: python benchmark.py --real --docs 5000")

    result = {'raw': r_raw, 'clean': r_clean,
              'delta_ppl': round(delta_ppl, 2),
              'delta_curva': round(delta_curva, 2)}
    with open('benchmark_resultado.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Salvo: benchmark_resultado.json")
    print(f"{'=' * 58}")
    return result


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--docs',   type=int, default=800)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--real',   action='store_true',
                   help='TinyStories real (precisa de internet)')
    args = p.parse_args()

    if args.real:
        try:
            from datasets import load_dataset
            print(f"  Baixando TinyStories ({args.docs} docs)...")
            ds   = load_dataset('roneneldan/TinyStories',
                                split=f'train[:{args.docs}]')
            docs = [item['text'] for item in ds]
            print(f"  ✓ {len(docs)} docs carregados")
            split       = int(len(docs)*0.8)
            train_raw   = docs[:split]; val = docs[split:]
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_clean = sanitizar(train_raw)
                r_raw   = treinar(train_raw,   val, "RAW",   args.epochs)
                r_clean = treinar(train_clean, val, "CLEAN", args.epochs)
            delta = r_raw['ppl_f'] - r_clean['ppl_f']
            print(f"\n  Delta perplexidade: {delta:+.2f}")
            if delta > 0:
                print(f"  ✅ Terra Dourada MELHOROU {delta:.2f} pontos!")
            else:
                print(f"  ➡️  Resultado neutro — tente mais épocas ou docs")
        except Exception as e:
            print(f"  Erro TinyStories: {e}")
            print("  Usando dataset sintético...")
            benchmark(args.docs, args.epochs)
    else:
        benchmark(args.docs, args.epochs)
