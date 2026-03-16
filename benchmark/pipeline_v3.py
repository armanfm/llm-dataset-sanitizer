"""
Terra Dourada - Pipeline v3 (Post-Gemini Feedback)
=====================================================
Implements Gemini's suggestions:
1. Word shingles MinHash (not character shingles)
2. Threshold 0.70 for near-duplicate detection
3. Correct order: Lexical -> Language -> MinHash -> FXL Turbo
4. FXL as the "silent" final quality seal

Key findings from testing:
- Lexical filter: 100% precision on nav/spam garbage
- Language detection: catches EN+ES, EN+PT mixing perfectly
- MinHash word shingles: better than char shingles for paraphrases
  BUT paraphrases with different words are still hard without embeddings
- FXL Turbo: works as final seal after other layers clean the corpus

Honest benchmark results (9 test docs):
  Pipeline accuracy: 78% (7/9)
  Two remaining issues:
  - Technical text with rare vocabulary sometimes triggers lang detector
  - Near-paraphrases need MiniLM embeddings for reliable detection

Usage:
    python pipeline_v3.py          # run test suite
    python pipeline_v3.py --corpus my_data.txt  # filter a corpus
"""

import math, hashlib, random, sys, argparse
from collections import Counter, deque

random.seed(42)

# ── Tokenizer ─────────────────────────────────────────────────────
def canon(t):
    o = []
    for c in t.lower():
        o.append(c if (c.isalnum() or c == ' ') else ' ')
    return ' '.join(''.join(o).split())

def tokenize(t): return canon(t).split()

def split_sentences(t):
    s, c = [], []
    for ch in t:
        c.append(ch)
        if ch in '.!?' and len(c) > 15:
            s.append(''.join(c).strip()); c = []
    if ''.join(c).strip(): s.append(''.join(c).strip())
    return s if len(s) >= 2 else [t]

def jaccard(a, b):
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb: return 0.
    return len(sa & sb) / len(sa | sb)


# ── MinHash with WORD shingles ────────────────────────────────────
# Gemini's key suggestion: word shingles beat character shingles
# for capturing near-paraphrases.
#
# Limitation: still struggles with heavy paraphrasing.
# Next step: replace with MiniLM sentence embeddings + cosine.

def minhash(text, n=64, k=3):
    """MinHash with k-word shingles."""
    words = tokenize(text)
    shingles = ({' '.join(words[i:i+k]) for i in range(len(words)-k+1)}
                if len(words) >= k else {canon(text)[:20]})
    return [min(int(hashlib.md5(f"{seed}:{s}".encode()).hexdigest(), 16)
                for s in shingles) for seed in range(n)]

def minhash_sim(sa, sb):
    return sum(1 for a, b in zip(sa, sb) if a == b) / len(sa)

def deduplicate(docs, threshold=0.70):
    """Remove near-duplicates. threshold=0.70 per Gemini's recommendation."""
    if len(docs) < 2: return docs, 0
    sigs = [minhash(d) for d in docs]
    ok, idx = [], []
    for i, (d, s) in enumerate(zip(docs, sigs)):
        if not any(minhash_sim(s, sigs[j]) >= threshold for j in idx):
            ok.append(d); idx.append(i)
    return ok, len(docs) - len(ok)


# ── Language mixing detection ─────────────────────────────────────
# Simple bag-of-function-words detector.
# Works well for: EN+ES, EN+PT, EN+FR mixing.
# Limitation: rare technical vocabulary can look like another language.
#
# Fix: expand TECH_WHITELIST for your domain.

VOCAB = {
    'en': {'the','of','and','in','to','a','is','that','for','on',
           'with','are','was','by','this','it','have','from','or',
           'be','an','not','but','at','they','can','has','were'},
    'pt': {'de','do','da','em','um','uma','para','com','que','por',
           'nao','se','os','as','no','na','ao','dos','das','mas',
           'sao','esta','mais','isso','esse','esta','foi','seu'},
    'es': {'de','la','que','el','en','y','los','del','se','las',
           'un','por','con','una','su','para','es','al','como',
           'mas','pero','sus','le','o','si','porque','este','hay'},
}

# Add your domain-specific rare words here to prevent false positives
TECH_WHITELIST = {
    'mitochondrial','transcription','genomic','cortical','synaptic',
    'backpropagation','tokenization','softmax','transformer','perplexity',
    'qubit','entanglement','photon','thermodynamic','electromagnetic',
    'recursion','parallelism','concurrency','serialization','heuristic',
}

def detect_language(sentence):
    words = set(tokenize(sentence)) - TECH_WHITELIST
    scores = {lang: len(words & vocab) for lang, vocab in VOCAB.items()}
    return max(scores, key=scores.get) if max(scores.values()) > 0 else 'xx'

def has_language_mixing(text, max_foreign_pct=0.30):
    """Returns (is_mixed, dominant_lang, foreign_pct)"""
    sents = split_sentences(text)
    if len(sents) < 2: return False, 'xx', 0.
    langs = [detect_language(s) for s in sents]
    counts = Counter(langs)
    dominant = counts.most_common(1)[0][0]
    foreign_pct = 1. - counts[dominant] / len(langs)
    return foreign_pct >= max_foreign_pct, dominant, foreign_pct


# ── Lexical filter ────────────────────────────────────────────────
SPAM_WORDS = {
    'copyright','subscribe','newsletter','click','terms','privacy',
    'policy','contact','rights','reserved','follow','share',
    'facebook','instagram','loading','login','signup','cookie',
    'home','about','advertisement','sponsored','download',
}

def lexical_filter(doc, sensitivity=0.5):
    """Returns (passed, rejection_reason)"""
    words = tokenize(doc)
    total = max(len(words), 1)
    spam_pct   = sum(1 for w in words if w in SPAM_WORDS) / total
    sents      = [s.strip() for s in doc.split('.') if len(s.strip()) > 5]
    avg_length = sum(len(tokenize(s)) for s in sents) / max(len(sents), 1)
    spam_max   = 0.15 - sensitivity * 0.10
    len_min    = 3.0  + sensitivity * 4.0
    if spam_pct > spam_max:  return False, f"spam={spam_pct:.0%}"
    if avg_length < len_min: return False, f"short_sentences={avg_length:.1f}"
    return True, ""


# ── FXL Turbo ─────────────────────────────────────────────────────
class FXLTurbo:
    """
    Temporal context rupture detection.
    ctx(t) = mean(1 - |sim(t) - sim(t-i)|) over window N
    
    Related to TextTiling (Hearst, 1997) but:
    - measures temporal VARIATION of similarity (not absolute)
    - operates at sentence level (not paragraph)
    - designed for dataset filtering (not document segmentation)
    """
    def __init__(self, window=6, threshold=0.05, max_seq=2):
        self.window    = window
        self.threshold = threshold
        self.max_seq   = max_seq
        self.history   = deque(maxlen=window)
        self.rupt_seq  = 0

    def update(self, sim):
        if not self.history:
            self.history.append(sim)
            return sim, False
        ctx = sum(1. - abs(sim - p) for p in self.history) / len(self.history)
        ctx = max(0., min(1., ctx))
        self.history.append(sim)
        rupt = ctx < self.threshold
        self.rupt_seq = self.rupt_seq + 1 if rupt else 0
        return ctx, self.rupt_seq >= self.max_seq

    def reset(self): self.history.clear(); self.rupt_seq = 0

    def evaluate(self, sentences):
        self.reset()
        blocked = False
        ruptures = 0
        for i in range(len(sentences) - 1):
            sim = jaccard(sentences[i], sentences[i+1])
            _, b = self.update(sim)
            if b:              blocked = True
            if sim < self.threshold: ruptures += 1
        rate = ruptures / max(len(sentences) - 1, 1)
        # Only reject if systematic (blocked AND high rate)
        return blocked and rate > 0.60, rate


# ── Pipeline per document ─────────────────────────────────────────
def filter_doc(doc, fxl, sensitivity=0.5):
    """
    Correct order per Gemini's suggestion:
    Lexical (fast) -> Language (fast) -> FXL (slow, final seal)
    MinHash runs on the full corpus, not per-document.
    """
    # 1. Lexical — kill obvious garbage first
    ok, reason = lexical_filter(doc, sensitivity)
    if not ok: return False, f"lexical:{reason}"

    # 2. Language mixing
    mixed, lang, pct = has_language_mixing(doc)
    if mixed: return False, f"mixed_lang={pct:.0%}({lang})"

    # 3. FXL Turbo — fine-grained narrative check
    rejected, rate = fxl.evaluate(split_sentences(doc))
    if rejected: return False, f"fxl_rupture={rate:.0%}"

    return True, ""


# ── Full corpus pipeline ──────────────────────────────────────────
def filter_corpus(docs, sensitivity=0.5, dedup_threshold=0.70, verbose=True):
    n0 = len(docs)
    if verbose: print(f"  Input          : {n0} docs")

    fxl = FXLTurbo()
    result = list(docs)

    # 1. Lexical
    result = [d for d in result if lexical_filter(d, sensitivity)[0]]
    if verbose: print(f"  After lexical  : {len(result):4d}  (-{n0-len(result)})")

    # 2. Language
    n1 = len(result)
    result = [d for d in result if not has_language_mixing(d)[0]]
    if verbose: print(f"  After language : {len(result):4d}  (-{n1-len(result)})")

    # 3. MinHash deduplication on remaining corpus
    n2 = len(result)
    result, n_dup = deduplicate(result, threshold=dedup_threshold)
    if verbose: print(f"  After MinHash  : {len(result):4d}  (-{n_dup} duplicates)")

    # 4. FXL Turbo
    n3 = len(result)
    result = [d for d in result if not filter_doc(d, fxl)[1].startswith('fxl')]
    if verbose: print(f"  After FXL      : {len(result):4d}  (-{n3-len(result)})")

    total_rej = n0 - len(result)
    if verbose:
        print(f"  Total rejected : {total_rej}/{n0} ({total_rej/n0*100:.0f}%)")
        print(f"  Clean dataset  : {len(result)} docs")
    return result


# ── Test suite ────────────────────────────────────────────────────
TEST_CASES = [
    (True,  "ML coherent",
     "Neural networks learn representations from training data. "
     "Gradient descent minimizes the loss function iteratively. "
     "Backpropagation computes gradients through all network layers. "
     "The optimizer updates model weights to improve predictions."),
    (True,  "Science coherent",
     "Photosynthesis converts sunlight into chemical energy stored in glucose. "
     "Chlorophyll absorbs light in the red and blue spectrum. "
     "Carbon dioxide and water are the raw materials used. "
     "The Calvin cycle produces glucose that fuels plant growth."),
    (True,  "Technical rare words",
     "Mitochondrial ATP synthesis is critical for neuronal function. "
     "Backpropagation through time is used to train recurrent networks. "
     "Quantum entanglement allows qubit superposition in computation. "
     "Thermodynamic equilibrium governs chemical reaction kinetics."),
    (False, "GARBAGE - navigation",
     "Home About Contact Privacy Policy Newsletter Subscribe today. "
     "Click here to read more articles on our website platform. "
     "Follow us on Facebook Twitter Instagram for daily updates. "
     "Copyright 2024 All Rights Reserved Terms and Conditions apply."),
    (False, "GARBAGE - random topics",
     "The Eiffel Tower stands three hundred meters tall in Paris. "
     "My cat refuses to eat dry food for three weeks now. "
     "Bitcoin reached all time high in November twenty twenty one. "
     "Spaghetti carbonara uses eggs cheese guanciale and pepper."),
    (False, "GARBAGE - EN+ES mixing",
     "Machine learning requires large datasets for good performance. "
     "Los algoritmos de aprendizaje son muy poderosos y utiles. "
     "Neural networks have revolutionized artificial intelligence. "
     "Compra ahora con descuento especial para nuevos usuarios."),
    (False, "GARBAGE - EN+PT mixing",
     "Deep learning models achieve state of the art results. "
     "Clique aqui para assinar nossa newsletter gratuita hoje. "
     "Attention mechanisms allow transformers to focus on context. "
     "Aproveite nossa promocao especial com frete gratis agora."),
    (True,  "Near-dup original",
     "How to install Python on Ubuntu Linux in three easy steps. "
     "Update your package manager with apt get update first. "
     "Run apt get install python three to get the interpreter. "
     "Verify with python three dash dash version command."),
    (True,  "Near-dup paraphrase",
     "How to install Python on Ubuntu Linux via three simple steps. "
     "Start by updating your package manager using apt get update. "
     "Next run apt get install python three to setup the interpreter. "
     "Confirm with python three dash dash version command."),
]


def run_tests():
    print("=" * 70)
    print("  Terra Dourada Pipeline v3 — Test Suite")
    print("=" * 70)
    fxl = FXLTurbo()
    correct = 0
    print(f"  {'Document':<42} {'GT':>4}  {'Pred':>5}  Reason")
    print(f"  {'─' * 68}")
    for gt, label, text in TEST_CASES:
        ok, reason = filter_doc(text, fxl)
        if ok == gt: correct += 1
        gt_s = "OK" if gt else "GARBAGE"
        pr_s = "OK" if ok else "GARBAGE"
        ok_s = "✓" if ok == gt else "✗"
        r = (reason[:32] if reason else "passed all layers")
        print(f"  {label[:42]:<42} {gt_s:>7}  {pr_s:>7}  {r} {ok_s}")

    acc = correct / len(TEST_CASES) * 100
    print(f"\n  Accuracy: {acc:.0f}% ({correct}/{len(TEST_CASES)})")
    print(f"\n  Known limitations:")
    print(f"  - Technical rare words sometimes trigger language detector")
    print(f"  - Near-paraphrases need MiniLM for reliable detection")
    print(f"  - FXL with Jaccard can miss semantic coherence (needs TF-IDF)")
    print(f"\n  Next step: integrate MiniLM ONNX for semantic similarity")

    print(f"\n  MinHash word-shingle comparison:")
    sig1 = minhash(TEST_CASES[7][2])
    sig2 = minhash(TEST_CASES[8][2])
    sig3 = minhash(TEST_CASES[0][2])
    sim_dup  = minhash_sim(sig1, sig2)
    sim_diff = minhash_sim(sig1, sig3)
    print(f"  Near-dup sim  : {sim_dup:.3f}  {'→ detected' if sim_dup>=0.7 else f'→ below threshold (MiniLM needed)'}")
    print(f"  Different sim : {sim_diff:.3f}  {'→ correctly distinct' if sim_diff<0.3 else '→ false positive'}")

    print(f"\n  Full corpus pipeline:")
    filter_corpus([t for _,_,t in TEST_CASES])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', default=None, help='Text file to filter')
    parser.add_argument('--sensitivity', type=float, default=0.5)
    parser.add_argument('--dedup', type=float, default=0.70)
    args = parser.parse_args()

    if args.corpus:
        with open(args.corpus) as f:
            docs = [b.strip() for b in f.read().split('\n\n') if b.strip()]
        print(f"Filtering {len(docs)} documents from {args.corpus}")
        clean = filter_corpus(docs, args.sensitivity, args.dedup)
        out = args.corpus.replace('.txt', '_clean.txt')
        with open(out, 'w') as f:
            f.write('\n\n'.join(clean))
        print(f"Saved: {out}")
    else:
        run_tests()
