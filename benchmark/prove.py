"""
Terra Dourada — Benchmark com Wikipedia Real
=============================================
Usa WikiText-103 — texto real da Wikipedia.
Baixa em ~30 segundos (não horas como o C4).

Estratégia de teste justa:
  - Pega artigos da Wikipedia
  - Injeta lixo real misturado (simula scraping)
  - Terra Dourada filtra
  - Compara perplexidade: RAW vs CLEAN

RODA:
    python terra_dourada_benchmark_local.py

Tempo total: ~15-20 minutos
"""

import math, random, time, json, sys, warnings
from collections import Counter

random.seed(42)

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
    print("Instale: pip install transformers accelerate datasets torch")
    sys.exit(1)


# ================================================================
# FILTRO TERRA DOURADA
# ================================================================

SPAM = {
    'copyright','subscribe','newsletter','click','terms','privacy',
    'policy','contact','rights','reserved','follow','download',
    'share','facebook','instagram','login','signup','cookie',
    'home','about','advertisement','sponsored','free','offer',
    'today','now','limited','deal','promo','buy','sale','here',
}

def tokenize(t):
    o = []
    for c in t.lower():
        o.append(c if (c.isalnum() or c == ' ') else ' ')
    return ''.join(o).split()

def sanitizar(docs):
    clean, rej = [], 0
    motivos = Counter()
    for doc in docs:
        ws   = tokenize(doc); tot = max(len(ws), 1)
        pct  = sum(1 for w in ws if w in SPAM) / tot
        ss   = [s.strip() for s in doc.split('.') if len(s.strip()) > 5]
        comp = sum(len(tokenize(s)) for s in ss) / max(len(ss), 1)
        if pct >= 0.10:
            motivos['spam_lexico'] += 1; rej += 1
        elif comp < 5.0:
            motivos['frases_curtas'] += 1; rej += 1
        else:
            clean.append(doc)
    print(f"  Terra Dourada: {len(clean)} aprovados, {rej} rejeitados "
          f"({rej/len(docs)*100:.1f}%)")
    for m, c in motivos.most_common():
        print(f"    {m}: {c} docs")
    return clean


# ================================================================
# LIXO REAL PARA INJETAR
# ================================================================

LIXO_INJETADO = [
    "Home About Contact Privacy Policy Terms Newsletter Subscribe today. Click here for more articles follow us on social media platforms now. Copyright 2024 All Rights Reserved advertisement sponsored content here. Follow Facebook Twitter Instagram LinkedIn for daily updates and promotions.",
    "Subscribe for exclusive deals and special offers available to members. Page not found please check the URL and try again later today. Session expired please log in again to continue your browsing session. Cookie notice this website uses cookies to improve your experience here.",
    "The Eiffel Tower stands in Paris France. My cat refuses to eat. Bitcoin reached all time high in November. Spaghetti carbonara uses eggs and cheese.",
    "Dogs are wonderful animals loved by people worldwide always everywhere. Dogs are amazing companions that families enjoy having in their homes. Dogs are great pets that children love to play with every day. Dogs are loyal animals that make excellent friends for everyone always.",
    "Buy now with free shipping limited time offer ends soon today here. Advertisement sponsored by our partners terms and conditions apply now. Like share subscribe follow for more amazing content every single week. All rights reserved copyright trademark registered follow us online today.",
    "Error loading page please refresh your browser and try again later. Javascript must be enabled to view this website please check settings. Loading please wait your request is being processed thank you patience. Internal server error please contact support if this problem continues.",
    "Learning Python is very useful for programmers and developers today. Learning Python is very useful for beginners and students everywhere. Learning Python is very useful for data scientists working daily now. Learning Python is very useful for many people in various fields always.",
    "Share this article on Facebook Twitter LinkedIn Pinterest Email today. Print save bookmark for later reading and sharing with others now. Was this article helpful yes no leave a comment below subscribe here. Related articles you might also like popular posts trending now more.",
]


# ================================================================
# CARREGA DATASET REAL (rápido)
# ================================================================

def carregar_dataset_real(n=4000):
    """
    Tenta carregar datasets reais do menor para o maior.
    WikiText-2: ~2MB, baixa em segundos.
    """
    from datasets import load_dataset

    # Opção 1: WikiText-2 (menor, mais rápido)
    try:
        print("  Tentando WikiText-2 (~2MB, rápido)...")
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        # Filtra parágrafos com pelo menos 50 palavras
        textos = []
        for item in ds:
            t = item['text'].strip()
            if len(t.split()) >= 50:
                textos.append(t)
            if len(textos) >= n:
                break
        if len(textos) >= 100:
            print(f"  ✓ {len(textos)} parágrafos carregados do WikiText-2")
            return textos
    except Exception as e:
        print(f"  WikiText-2 falhou: {e}")

    # Opção 2: WikiText-103 (maior mas ainda rápido)
    try:
        print("  Tentando WikiText-103 (~180MB)...")
        ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
        textos = []
        for item in ds:
            t = item['text'].strip()
            if len(t.split()) >= 50:
                textos.append(t)
            if len(textos) >= n:
                break
        if len(textos) >= 100:
            print(f"  ✓ {len(textos)} parágrafos carregados do WikiText-103")
            return textos
    except Exception as e:
        print(f"  WikiText-103 falhou: {e}")

    # Opção 3: TinyStories (já conhecido, baixa rápido depois do cache)
    try:
        print("  Tentando TinyStories (já em cache se rodou antes)...")
        ds = load_dataset('roneneldan/TinyStories', split=f'train[:{n}]')
        textos = [item['text'] for item in ds if len(item['text'].split()) >= 30]
        if textos:
            print(f"  ✓ {len(textos)} histórias carregadas do TinyStories")
            return textos
    except Exception as e:
        print(f"  TinyStories falhou: {e}")

    return None


def criar_dataset_sujo(textos_limpos, proporcao_lixo=0.40):
    """
    Pega textos limpos reais e injeta lixo em 40% dos docs.
    Simula o que acontece num dataset de web scraping real.
    """
    n_lixo = int(len(textos_limpos) * proporcao_lixo)

    # Docs com lixo injetado no meio
    docs_sujos = []
    for i in range(n_lixo):
        texto_base = textos_limpos[i % len(textos_limpos)]
        lixo = LIXO_INJETADO[i % len(LIXO_INJETADO)]
        # Injeta o lixo no meio do documento
        meio = len(texto_base) // 2
        doc_sujo = texto_base[:meio] + " " + lixo + " " + texto_base[meio:]
        docs_sujos.append(doc_sujo)

    # Dataset completo: limpos + sujos misturados
    dataset = textos_limpos + docs_sujos
    random.shuffle(dataset)

    print(f"  Dataset criado:")
    print(f"    {len(textos_limpos)} docs limpos reais (Wikipedia)")
    print(f"    {len(docs_sujos)} docs com lixo injetado ({proporcao_lixo*100:.0f}%)")
    print(f"    Total: {len(dataset)} docs")

    return dataset, textos_limpos  # textos_limpos serve como val set limpo


# ================================================================
# GPT-2
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
        for i in range(0, min(len(docs), 300), bs):
            enc  = tokenizer(docs[i:i+bs], truncation=True, max_length=128,
                             padding=True, return_tensors='pt')
            ids  = enc['input_ids'].to(device)
            mask = enc['attention_mask'].to(device)
            out  = model(input_ids=ids, attention_mask=mask, labels=ids)
            total += out.loss.item(); n += 1
    return math.exp(total/max(n,1)), total/max(n,1)


def criar_args(label, epochs):
    base = dict(
        output_dir=f'./tmp_{label}',
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        learning_rate=5e-4,
        logging_steps=999,
        save_steps=9999,
        report_to='none',
    )
    use_cpu = not torch.cuda.is_available()
    for arg in ['use_cpu', 'no_cuda']:
        try:
            return TrainingArguments(**base, **{arg: use_cpu})
        except TypeError:
            continue
    return TrainingArguments(**base)


def treinar(train_docs, val_docs, label, epochs=3):
    print(f"\n  ── GPT-2 [{label}] | {len(train_docs)} docs ──")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    config = GPT2Config(n_embd=128, n_layer=2, n_head=2,
                        n_positions=128, vocab_size=tokenizer.vocab_size)
    model  = GPT2LMHeadModel(config)

    ppl_i, _ = calc_ppl(model, tokenizer, val_docs)
    print(f"  Inicial (sem treino): ppl={ppl_i:.1f}")

    train_ds = TextDS(train_docs, tokenizer)
    collator  = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Trainer(model=model, args=criar_args(label, epochs),
                train_dataset=train_ds, data_collator=collator).train()
    elapsed = time.time() - t0

    ppl_f, loss_f = calc_ppl(model, tokenizer, val_docs)
    curva = (ppl_i - ppl_f) / ppl_i * 100
    print(f"  Final   (treinado):   ppl={ppl_f:.1f}  "
          f"curva={curva:+.1f}%  tempo={elapsed:.0f}s")

    return {'ppl_i': round(ppl_i,2), 'ppl_f': round(ppl_f,2),
            'loss': round(loss_f,4), 'curva': round(curva,2),
            'n': len(train_docs), 'tempo': round(elapsed,1)}


# ================================================================
# BENCHMARK
# ================================================================

def benchmark(n_docs=4000, epochs=3):
    print()
    print("=" * 65)
    print("  TERRA DOURADA — Benchmark com Wikipedia Real")
    print("  Wikipedia limpa + lixo injetado = simula web scraping")
    print("=" * 65)

    # 1. Carrega dados reais
    print(f"\n  [1/5] Carregando dados reais...")
    textos = carregar_dataset_real(n_docs)

    if textos is None:
        print("\n  ERRO: Não conseguiu carregar nenhum dataset.")
        print("  Verifique sua conexão com a internet.")
        sys.exit(1)

    # Usa subset para ser mais rápido
    textos = textos[:n_docs]
    random.shuffle(textos)

    # 2. Cria dataset sujo
    print(f"\n  [2/5] Criando dataset sujo (40% lixo injetado)...")
    split_val = int(len(textos) * 0.2)
    val_limpo  = textos[:split_val]   # validação: só texto limpo
    base_treino = textos[split_val:]   # base para treino

    dataset_sujo, _ = criar_dataset_sujo(base_treino, proporcao_lixo=0.40)
    train_raw = dataset_sujo
    print(f"\n  Validação: {len(val_limpo)} docs LIMPOS (Wikipedia pura)")
    print(f"  Treino RAW: {len(train_raw)} docs (60% limpo + 40% com lixo)")

    # 3. Sanitiza
    print(f"\n  [3/5] Sanitizando treino com Terra Dourada...")
    train_clean = sanitizar(train_raw)
    pct = (len(train_raw)-len(train_clean)) / len(train_raw) * 100
    print(f"  Redução: {len(train_raw)} → {len(train_clean)} docs (-{pct:.1f}%)")

    # 4. Treina
    print(f"\n  [4/5] Treinando dois GPT-2 ({epochs} épocas)...")
    print(f"  Ambos validam nos mesmos {len(val_limpo)} docs LIMPOS")
    print(f"  (comparação justa — mesmo set de validação)\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_raw   = treinar(train_raw,   val_limpo, "RAW",   epochs)
        r_clean = treinar(train_clean, val_limpo, "CLEAN", epochs)

    # 5. Resultado
    delta_ppl   = r_raw['ppl_f']   - r_clean['ppl_f']
    delta_curva = r_clean['curva'] - r_raw['curva']

    print(f"\n{'=' * 65}")
    print("  RESULTADO FINAL")
    print(f"{'=' * 65}")
    print(f"\n  {'Modelo':<32} {'Docs':>6}  {'PPL':>8}  {'Curva':>8}")
    print(f"  {'─' * 60}")
    print(f"  {'GPT-2 (RAW, com lixo)':<32} {r_raw['n']:>6}  "
          f"{r_raw['ppl_f']:>8.2f}  {r_raw['curva']:>+7.1f}%")
    print(f"  {'GPT-2 (Terra Dourada, limpo)':<32} {r_clean['n']:>6}  "
          f"{r_clean['ppl_f']:>8.2f}  {r_clean['curva']:>+7.1f}%")
    print(f"  {'─' * 60}")
    print(f"\n  Docs rejeitados  : {len(train_raw)-len(train_clean)} ({pct:.1f}%)")
    print(f"  Delta PPL        : {delta_ppl:+.2f} pontos")
    print(f"  Delta curva      : {delta_curva:+.1f}%")
    print()

    if delta_ppl > 1:
        print(f"  ✅ TERRA DOURADA MELHOROU O MODELO!")
        print(f"  PPL caiu {delta_ppl:.2f} pontos treinando em dados limpos.")
        print(f"  Prova: sanitizar melhora o aprendizado da LLM.")
    elif delta_ppl > -1:
        print(f"  ➡️  Resultado neutro — tente --epochs 5")
    else:
        print(f"  ⚠️  Tente com mais épocas: --epochs 5")

    result = {
        'dataset': 'Wikipedia (WikiText) + lixo injetado',
        'raw': r_raw, 'clean': r_clean,
        'delta_ppl': round(delta_ppl, 2),
        'delta_curva': round(delta_curva, 2),
        'pct_rejeitado': round(pct, 1),
    }
    with open('benchmark_resultado.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Salvo: benchmark_resultado.json")
    print(f"{'=' * 65}")
    return result


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--docs',   type=int, default=4000)
    p.add_argument('--epochs', type=int, default=3)
    args = p.parse_args()
    benchmark(args.docs, args.epochs)
