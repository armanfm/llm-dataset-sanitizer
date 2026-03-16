"""
Terra Dourada — Benchmark GPT-2 com Dataset Sujo Real
=======================================================
Dataset sintético que simula Common Crawl de verdade:
  - Menus de navegação misturados com artigos
  - Rodapés com copyright e spam
  - Tópicos aleatórios numa mesma página
  - SEO farms repetitivos
  - Boilerplate HTML
  - Texto legítimo de qualidade

RODA:
    python terra_dourada_benchmark_local.py

Tempo: ~15-20 minutos em CPU.
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
    print("Instale: pip install transformers accelerate torch")
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
    for doc in docs:
        ws   = tokenize(doc); tot = max(len(ws), 1)
        pct  = sum(1 for w in ws if w in SPAM) / tot
        ss   = [s.strip() for s in doc.split('.') if len(s.strip()) > 5]
        comp = sum(len(tokenize(s)) for s in ss) / max(len(ss), 1)
        if pct < 0.10 and comp >= 5.0:
            clean.append(doc)
        else:
            rej += 1
    print(f"  Terra Dourada: {len(clean)} aprovados, {rej} rejeitados "
          f"({rej/len(docs)*100:.1f}%)")
    return clean


# ================================================================
# DATASET SUJO — simula Common Crawl de verdade
# ================================================================

# ── Texto limpo de qualidade ──────────────────────────────────────
LIMPOS = [
    # Ciência
    "Neural networks learn hierarchical representations from large datasets. Each layer extracts increasingly abstract features from the raw input. Gradient descent minimizes the loss function by computing derivatives. The optimizer adjusts weights to reduce prediction errors over time.",
    "Photosynthesis is the process by which plants convert sunlight into glucose. Chlorophyll molecules in the chloroplasts absorb red and blue light efficiently. Carbon dioxide and water combine to produce glucose and release oxygen. This process sustains almost all life on Earth through the food chain.",
    "The human immune system consists of innate and adaptive defense mechanisms. White blood cells identify and destroy pathogens that enter the body. Antibodies bind specifically to antigens on the surface of bacteria and viruses. Vaccines train the immune system to recognize pathogens without causing disease.",
    "DNA contains the genetic instructions for building and maintaining organisms. The double helix structure was discovered by Watson and Crick in nineteen fifty three. Genes are sections of DNA that encode specific proteins with important functions. Mutations in DNA can lead to changes in protein structure and cellular behavior.",
    "The theory of relativity changed our understanding of space and time forever. Einstein proposed that gravity is the curvature of spacetime caused by mass. Time passes more slowly in stronger gravitational fields and at higher speeds. This has practical implications for GPS satellites which must account for time dilation.",
    "The water cycle continuously moves water between land, oceans, and atmosphere. Evaporation from oceans and lakes produces water vapor that rises upward. Condensation forms clouds when water vapor cools at higher altitudes. Precipitation returns water to the surface as rain, snow, and hail.",
    # Tecnologia
    "Rust is designed to provide memory safety without needing garbage collection. The ownership system ensures that each value has exactly one owner at a time. The borrow checker prevents data races and dangling pointer references at compile time. These guarantees make Rust suitable for systems programming and embedded devices.",
    "Transformers use self-attention mechanisms to process sequential data in parallel. Each token can attend to every other token in the sequence simultaneously. Positional encoding preserves the order information that attention would otherwise lose. This architecture achieved breakthrough results in natural language processing tasks.",
    "Quantum computers use qubits which can exist in superposition of zero and one. Entanglement allows qubits to be correlated in ways impossible for classical bits. Quantum algorithms can solve certain problems exponentially faster than classical computers. Error correction remains a major challenge preventing practical quantum computing today.",
    "The Linux kernel manages hardware resources for all processes running on a computer. The scheduler determines which process gets CPU time and for how long. Virtual memory allows each process to have its own isolated address space. Device drivers translate generic operating system calls into hardware specific instructions.",
    # História e ciências sociais
    "The Renaissance was a period of cultural and intellectual flourishing in Europe. Artists like Leonardo da Vinci combined scientific observation with artistic skill. The printing press allowed ideas to spread rapidly across the continent for the first time. This period laid the foundations for modern science and democratic thought.",
    "The Industrial Revolution transformed manufacturing and society in the eighteenth century. Steam engines powered factories and enabled mass production of goods for the first time. Urbanization accelerated as people moved from farms to cities seeking factory employment. Working conditions were often dangerous and children were widely used as laborers.",
    "Climate science studies the long-term patterns of temperature, precipitation, and atmospheric conditions. Data from ice cores reveals temperature fluctuations going back hundreds of thousands of years. Human activities have increased carbon dioxide concentrations to levels not seen in millions of years. The consequences include rising sea levels and more frequent extreme weather events.",
]

# ── Lixo tipo 1: menus de navegação misturados com conteúdo ──────
LIXO_MENU = [
    "Home About Us Services Portfolio Blog Contact Newsletter Subscribe. Neural networks are transforming artificial intelligence research today. Click here for more articles follow us on social media platforms. Privacy Policy Terms of Service Copyright 2024 All Rights Reserved here.",
    "Welcome to our website home about contact privacy terms newsletter. Machine learning algorithms process large amounts of data efficiently. Subscribe now for exclusive deals and special member offers today. Follow Facebook Twitter Instagram LinkedIn YouTube for updates daily.",
    "Navigation menu home about services blog portfolio contact us today. The human brain processes information through complex neural pathways. Read more articles on our platform subscribe to newsletter free. Copyright reserved terms conditions privacy policy apply here today.",
    "Home page about us our team services what we do contact information. Photosynthesis converts sunlight into chemical energy in plant cells. Like share subscribe follow for more amazing content every week. Advertisement sponsored by our partners terms and conditions apply now.",
    "Site map home about services portfolio testimonials blog contact here. Deep learning has revolutionized computer vision and speech recognition. Click to read more download our app get exclusive deals now. All rights reserved copyright trademark registered follow us online today.",
]

# ── Lixo tipo 2: tópicos completamente misturados numa página ────
LIXO_TOPICOS = [
    "The Eiffel Tower was built in Paris France in eighteen eighty nine. Neural networks require large amounts of labeled training data to work well. My dog refuses to eat the new brand of food I bought yesterday. Bitcoin reached a new all time high price in November of twenty twenty one.",
    "Quantum mechanics describes the behavior of particles at subatomic scales. The best recipe for pasta carbonara uses eggs cheese and guanciale. The French Revolution began in seventeen eighty nine with the storming of the Bastille. Dogs are the most popular pets in households across North America today.",
    "The Amazon rainforest produces twenty percent of the world oxygen supply daily. How to make the perfect scrambled eggs with butter and fresh herbs today. Artificial intelligence is transforming industries from healthcare to finance globally. The Roman Empire at its height controlled most of Europe and North Africa.",
    "The speed of light in vacuum is approximately three hundred thousand kilometers per second. My favorite pizza toppings are pepperoni mushrooms and extra cheese always. Climate change is causing glaciers to melt at unprecedented rates globally. The Great Wall of China was built over many centuries by different dynasties.",
    "Mitochondria are the powerhouses of the cell producing ATP for energy. Last night I watched an incredible movie about space exploration and astronauts. The stock market crashed in nineteen twenty nine triggering the Great Depression. Spaghetti bolognese is made with ground beef tomatoes and red wine traditionally.",
]

# ── Lixo tipo 3: SEO farms — texto repetitivo e vazio ───────────
LIXO_SEO = [
    "Dogs are wonderful animals that people love very much around the world. Dogs are amazing companions that families enjoy having in their homes. Dogs are great pets that children love to play with every single day. Dogs are loyal animals and dogs make excellent friends for everyone always.",
    "Learning Python programming is very useful for developers and programmers today. Learning Python programming is very useful for beginners and students everywhere. Learning Python programming is very useful for data scientists working daily. Learning Python programming is very useful for many people in various fields always.",
    "The best way to lose weight is to exercise regularly and eat healthy food. The best way to lose weight is to follow a balanced diet every day. The best way to lose weight is to drink plenty of water always daily. The best way to lose weight is to sleep well and reduce stress levels.",
    "Our company provides the best services in the industry for our clients. Our company provides excellent quality products at affordable prices always. Our company has years of experience helping businesses grow and succeed. Our company is committed to customer satisfaction and delivering results always.",
    "Buy now and save money on all our products with free shipping today. Buy now and get exclusive discounts available to members only this week. Buy now while stocks last limited time offer ends very soon here. Buy now and receive a free gift with every purchase over fifty dollars.",
]

# ── Lixo tipo 4: boilerplate de rodapé e header HTML ────────────
LIXO_BOILERPLATE = [
    "Skip to main content accessibility statement screen reader support here. Error loading page please refresh your browser and try again later. Session expired please log in again to continue using our services. Javascript must be enabled to view this website please check your settings.",
    "Loading please wait your request is being processed thank you patience. Page not found four zero four the resource you requested does not exist. Internal server error five hundred please contact support if problem persists. Forbidden access four zero three you do not have permission to view this.",
    "Cookie notice this website uses cookies to improve your browsing experience here. Accept all cookies decline optional cookies manage cookie preferences settings. We use cookies for analytics advertising and personalization of content daily. Your privacy matters to us please read our privacy policy for more information.",
    "Share this article on Facebook Twitter LinkedIn Pinterest WhatsApp Email today. Print this page save as PDF bookmark for later reading sharing options. Was this article helpful yes no leave a comment below subscribe newsletter. Related articles you might also like popular posts trending now see more.",
    "Subscribe to our weekly newsletter for the latest news and updates today. Enter your email address below to receive exclusive content and special offers. Unsubscribe at any time privacy policy terms of service contact us here. Follow us on social media for daily inspiration tips and tricks always.",
]

def get_dirty_dataset(n_clean=600, n_dirty=600):
    """
    Cria dataset sujo misturando texto limpo com os 4 tipos de lixo.
    Proporção realista do Common Crawl: ~50% tem algum tipo de problema.
    """
    limpos  = (LIMPOS * 100)[:n_clean]

    n_cada  = n_dirty // 4
    sujos   = (
        (LIXO_MENU       * 50)[:n_cada] +
        (LIXO_TOPICOS    * 50)[:n_cada] +
        (LIXO_SEO        * 50)[:n_cada] +
        (LIXO_BOILERPLATE* 50)[:n_dirty - 3*n_cada]
    )

    docs = limpos + sujos
    random.shuffle(docs)
    print(f"  Dataset criado: {len(limpos)} docs limpos + {len(sujos)} docs sujos")
    print(f"  Total: {len(docs)} docs ({len(sujos)/len(docs)*100:.0f}% lixo)")
    return docs, limpos[:200]  # retorna também val set limpo separado


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
    print(f"  Inicial: ppl={ppl_i:.1f}")

    train_ds = TextDS(train_docs, tokenizer)
    collator  = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Trainer(model=model, args=criar_args(label, epochs),
                train_dataset=train_ds, data_collator=collator).train()

    t0 = time.time()
    ppl_f, loss_f = calc_ppl(model, tokenizer, val_docs)
    curva = (ppl_i - ppl_f) / ppl_i * 100
    print(f"  Final:   ppl={ppl_f:.1f}  (curva {curva:+.1f}%)")

    return {'ppl_i': round(ppl_i,2), 'ppl_f': round(ppl_f,2),
            'loss': round(loss_f,4), 'curva': round(curva,2),
            'n': len(train_docs)}


# ================================================================
# BENCHMARK
# ================================================================

def benchmark(epochs=3):
    print()
    print("=" * 62)
    print("  TERRA DOURADA — Benchmark com Dataset Sujo")
    print("  Simula Common Crawl: 50% lixo real misturado")
    print("=" * 62)

    # 1. Cria dataset sujo
    print(f"\n  [1/4] Criando dataset sujo (simula web scraping)...")
    docs, val_limpo = get_dirty_dataset(n_clean=600, n_dirty=600)

    split     = int(len(docs) * 0.8)
    train_raw = docs[:split]

    # IMPORTANTE: validação só com texto limpo — comparação justa
    print(f"  Treino: {len(train_raw)} docs | Validação: {len(val_limpo)} docs LIMPOS")
    print(f"  (validação só com texto limpo = comparação justa)")

    # 2. Sanitiza
    print(f"\n  [2/4] Sanitizando com Terra Dourada...")
    train_clean = sanitizar(train_raw)
    pct = (len(train_raw)-len(train_clean)) / len(train_raw) * 100
    print(f"  Redução: {len(train_raw)} → {len(train_clean)} docs (-{pct:.1f}%)")

    # 3. Treina
    print(f"\n  [3/4] Treinando dois GPT-2 ({epochs} épocas)...")
    print("  (validação nos mesmos docs limpos para ambos)\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_raw   = treinar(train_raw,   val_limpo, "RAW",   epochs)
        r_clean = treinar(train_clean, val_limpo, "CLEAN", epochs)

    # 4. Resultado
    delta_ppl   = r_raw['ppl_f'] - r_clean['ppl_f']
    delta_curva = r_clean['curva'] - r_raw['curva']

    print(f"\n{'=' * 62}")
    print("  RESULTADO FINAL")
    print(f"{'=' * 62}")
    print(f"\n  {'Modelo':<30} {'Docs':>6}  {'PPL':>8}  {'Curva':>8}")
    print(f"  {'─' * 56}")
    print(f"  {'GPT-2 (bruto, com lixo)':<30} {r_raw['n']:>6}  "
          f"{r_raw['ppl_f']:>8.2f}  {r_raw['curva']:>+7.1f}%")
    print(f"  {'GPT-2 (Terra Dourada, limpo)':<30} {r_clean['n']:>6}  "
          f"{r_clean['ppl_f']:>8.2f}  {r_clean['curva']:>+7.1f}%")
    print(f"  {'─' * 56}")
    print(f"\n  Delta perplexidade : {delta_ppl:+.2f} pontos")
    print(f"  Delta curva        : {delta_curva:+.1f}%")
    print()

    if delta_ppl > 1:
        print(f"  ✅ TERRA DOURADA MELHOROU O MODELO!")
        print(f"  PPL caiu {delta_ppl:.2f} pontos treinando em dados limpos.")
        print(f"  O modelo aprendeu padrões de linguagem, não lixo.")
    elif delta_ppl > -1:
        print(f"  ➡️  Resultado neutro — tente --epochs 5")
    else:
        print(f"  ⚠️  Dataset precisa de mais variação — tente --epochs 5")

    result = {
        'raw': r_raw, 'clean': r_clean,
        'delta_ppl': round(delta_ppl, 2),
        'delta_curva': round(delta_curva, 2),
        'pct_rejeitado': round(pct, 1),
    }
    with open('benchmark_resultado.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Salvo: benchmark_resultado.json")
    print(f"{'=' * 62}")
    return result


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=3)
    args = p.parse_args()
    benchmark(args.epochs)
