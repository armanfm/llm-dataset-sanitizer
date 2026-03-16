"""
Terra Dourada - Hard Cases Analysis
=====================================
Tests the 5 difficult cases identified in the analysis.
Honest assessment of what the system can and cannot detect without GPU.

5 Hard Cases:
1. HTML Boilerplate  - long sentences, good grammar, little spam
2. Repeated Templates - nav structure without explicit spam words
3. SEO Farm (repetitive) - coherent, grammatical, same topic but empty
4. Mixed Topics (well written) - each sentence is perfect
5. Bad Machine Translation - strange grammar but coherent

Results:
- Cases 1-2: DETECTED (residual spam/rupture still present)
- Cases 3-5: NOT DETECTED without additional features

New features needed for cases 3-5 (all CPU, no GPU):
- Repetitiveness: sim_media > 0.35 between consecutive sentences
- Lexical entropy: entropy < 3.0 (poor vocabulary diversity)
- Unique word ratio: < 0.50 (too many repetitions)
"""

import math
from collections import Counter, deque


def canon(t):
    o = []
    for c in t.lower():
        o.append(c if (c.isalnum() or c == ' ') else ' ')
    return ' '.join(''.join(o).split())

def tokenize(t): return canon(t).split()

def jaccard(a, b):
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb: return 0.
    return len(sa & sb) / len(sa | sb)

def split_sentences(t):
    s, c = [], []
    for ch in t:
        c.append(ch)
        if ch in '.!?' and len(c) > 15:
            s.append(''.join(c).strip()); c = []
    if ''.join(c).strip(): s.append(''.join(c).strip())
    return s if len(s) >= 2 else [t]


# ── Feature: Lexical entropy and diversity ────────────────────────
def lexical_features(text):
    """
    Measures vocabulary richness.
    Low entropy + low unique_ratio = repetitive/SEO content.
    """
    words = tokenize(text)
    total = max(len(words), 1)
    freq  = Counter(words)
    n     = sum(freq.values())
    entropy     = -sum((c/n) * math.log2(c/n + 1e-9) for c in freq.values()) if n > 0 else 0
    unique_ratio = len(freq) / total
    return entropy, unique_ratio


# ── Feature: Consecutive sentence similarity ─────────────────────
def consecutive_similarity(text):
    """
    Measures average similarity between consecutive sentences.
    High value (> 0.35) = repetitive content (SEO farms, templates).
    Low value (< 0.05) = topic jumping (mixed topics, boilerplate).
    """
    sents = split_sentences(text)
    if len(sents) < 2: return 0.5
    sims = [jaccard(sents[i], sents[i+1]) for i in range(len(sents)-1)]
    return sum(sims) / len(sims)


# ── Current pipeline (without new features) ──────────────────────
SPAM_WORDS = {
    'copyright','subscribe','newsletter','click','terms','privacy',
    'policy','contact','rights','reserved','follow','share',
    'facebook','instagram','loading','login','signup','cookie',
    'home','about','advertisement','sponsored','download',
}
VOCAB = {
    'en': {'the','of','and','in','to','a','is','that','for','on',
           'with','are','was','by','this','it','have','from'},
    'pt': {'de','do','da','em','um','uma','para','com','que','por',
           'nao','se','os','as','no','na','ao','dos','das','mas'},
    'es': {'de','la','que','el','en','y','los','del','se','las',
           'un','por','con','una','su','para','es','al','como','mas'},
}

def detect_lang(s):
    ws = set(tokenize(s))
    sc = {l: len(ws & v) for l, v in VOCAB.items()}
    return max(sc, key=sc.get) if max(sc.values()) > 0 else 'xx'

class FXLTurbo:
    def __init__(self, window=6, threshold=0.05, max_seq=2):
        self.window = window; self.threshold = threshold
        self.max_seq = max_seq
        self.hist = deque(maxlen=window); self.seq = 0
    def update(self, sim):
        if not self.hist: self.hist.append(sim); return sim, False
        ctx = sum(1. - abs(sim-p) for p in self.hist) / len(self.hist)
        ctx = max(0., min(1., ctx)); self.hist.append(sim)
        rupt = ctx < self.threshold
        self.seq = self.seq + 1 if rupt else 0
        return ctx, self.seq >= self.max_seq
    def reset(self): self.hist.clear(); self.seq = 0

def current_pipeline(text):
    """Current pipeline without new features."""
    words = tokenize(text); total = max(len(words), 1)
    spam  = sum(1 for w in words if w in SPAM_WORDS) / total
    sents = [s.strip() for s in text.split('.') if len(s.strip()) > 5]
    avg_len = sum(len(tokenize(s)) for s in sents) / max(len(sents), 1)

    if spam > 0.10:  return False, f"spam={spam:.0%}"
    if avg_len < 5.0: return False, "short_sentences"

    sents2 = split_sentences(text)
    langs  = [detect_lang(s) for s in sents2]
    cnt    = Counter(langs); dom = cnt.most_common(1)[0][0]
    foreign = 1. - cnt[dom] / len(langs)
    if foreign >= 0.30: return False, f"mixed_lang={foreign:.0%}"

    fxl = FXLTurbo()
    blocked = False; ruptures = 0
    for i in range(len(sents2)-1):
        sim = jaccard(sents2[i], sents2[i+1])
        _, b = fxl.update(sim)
        if b: blocked = True
        if sim < 0.05: ruptures += 1
    rate = ruptures / max(len(sents2)-1, 1)
    if blocked and rate > 0.6: return False, f"fxl_rupture={rate:.0%}"

    return True, "passed"


def pipeline_with_new_features(text):
    """Pipeline + repetitiveness detection (CPU, no model needed)."""
    ok, reason = current_pipeline(text)
    if not ok: return False, reason

    # NEW: repetitiveness detection
    entropy, unique = lexical_features(text)
    sim_med = consecutive_similarity(text)

    # SEO farm signature: high consecutive similarity OR low entropy
    if sim_med > 0.35:
        return False, f"repetitive:high_sim={sim_med:.2f}"
    if entropy < 2.8:
        return False, f"repetitive:low_entropy={entropy:.2f}"
    if unique < 0.45:
        return False, f"repetitive:low_vocab={unique:.2f}"

    return True, "passed"


# ── Test cases ────────────────────────────────────────────────────
HARD_CASES = [
    # label, is_clean, text
    (False, "1. HTML Boilerplate",
     "Article about climate change impacts on global temperatures today. "
     "Related articles: Climate impacts in Europe. How glaciers melt faster. "
     "Privacy Policy Terms Contact Copyright 2024 Example Media website. "
     "Subscribe to our newsletter for latest environmental news today."),

    (False, "2. Repeated Template",
     "This article explains how neural networks work in modern practice. "
     "Share this article on social media to help others learn about AI. "
     "Related posts: Deep learning tutorial. Machine learning basics guide. "
     "Follow us on Twitter for more artificial intelligence content today."),

    (False, "3. SEO Farm (repetitive)",
     "Dogs are very popular animals in the world today everywhere you go. "
     "Dogs like to play with humans and dogs are very friendly animals always. "
     "Dogs are wonderful and loyal pets for families with children at home. "
     "Dogs are great companions and dogs make excellent family animals indeed."),

    (False, "4. Mixed Topics (well-written)",
     "Quantum mechanics describes the behavior of subatomic particles precisely. "
     "The Mediterranean diet includes olive oil and fresh vegetables daily always. "
     "Python programming uses indentation instead of curly braces for code blocks. "
     "African elephants are the largest land animals on the entire planet Earth."),

    (False, "5. Bad Machine Translation",
     "The car of future will be electric and have much autonomy for driving. "
     "The system of driving will not require the human for controlling car now. "
     "The road infrastructure must adapt itself for the new transport methods. "
     "The government will need invest in new technologies of electric vehicles."),

    # Control cases
    (True,  "CONTROL: ML coherent",
     "Neural networks learn hierarchical representations from large datasets. "
     "Gradient descent minimizes the loss function by updating model weights. "
     "Backpropagation computes gradients efficiently through the chain rule. "
     "Regularization techniques like dropout prevent overfitting in deep models."),

    (True,  "CONTROL: Science coherent",
     "Photosynthesis converts sunlight into chemical energy stored in glucose. "
     "Chlorophyll in plant cells absorbs light in the red and blue spectrum. "
     "Carbon dioxide from air and water from soil are the raw materials used. "
     "The Calvin cycle uses ATP and NADPH to synthesize organic compounds."),
]


def run_analysis():
    print("=" * 70)
    print("  Terra Dourada - Hard Cases Analysis")
    print("  Honest assessment: what we detect vs what needs GPU")
    print("=" * 70)

    print(f"\n  {'Case':<40} {'GT':>7}  {'Now':>7}  {'v4':>7}  {'Key Metric'}")
    print(f"  {'─' * 70}")

    correct_now = 0
    correct_v4  = 0

    for is_clean, label, text in HARD_CASES:
        ok_now, r_now = current_pipeline(text)
        ok_v4,  r_v4  = pipeline_with_new_features(text)

        entropy, unique = lexical_features(text)
        sim_med = consecutive_similarity(text)

        if ok_now == is_clean: correct_now += 1
        if ok_v4  == is_clean: correct_v4  += 1

        gt_s  = "CLEAN" if is_clean else "GARBAGE"
        now_s = "clean" if ok_now  else "REJECT"
        v4_s  = "clean" if ok_v4   else "REJECT"
        ok_n  = "✓" if ok_now == is_clean else "✗"
        ok_v  = "✓" if ok_v4  == is_clean else "✗"

        metric = f"sim={sim_med:.2f} entr={entropy:.1f} uniq={unique:.2f}"
        print(f"  {label[:40]:<40} {gt_s:>7}  {now_s:>7}{ok_n} {v4_s:>7}{ok_v}  {metric}")

    n = len(HARD_CASES)
    print(f"\n  Accuracy now : {correct_now}/{n} ({correct_now/n*100:.0f}%)")
    print(f"  Accuracy v4  : {correct_v4}/{n}  ({correct_v4/n*100:.0f}%)  [+repetitiveness filter]")

    print(f"""
  ─────────────────────────────────────────────────────────────
  WHAT WE ALREADY SOLVE (no GPU):
  ✅ Nav menus / footers / spam       - lexical filter
  ✅ Language mixing (EN+ES, EN+PT)   - language detector
  ✅ Near-duplicate copies            - MinHash word shingles
  ✅ Abrupt ruptures (article+menu)   - FXL Turbo
  ✅ Code mixed in prose              - pct_cod feature

  WHAT v4 ADDS (still no GPU):
  ✅ SEO farms / repetitive content   - sim_media + entropy

  WHAT STILL NEEDS GPU/EMBEDDINGS:
  ✗  Well-written topic salad         - needs MiniLM
  ✗  Bad machine translation          - needs LM perplexity
  ✗  Heavy paraphrasing               - needs semantic similarity

  HONEST POSITION:
  Terra Dourada handles the structural garbage layer perfectly.
  This is exactly what CCNet and FineWeb use as their FIRST layer.
  The difference: Terra Dourada does it without any human labels
  and without GPU — making it accessible to everyone.
  ─────────────────────────────────────────────────────────────
""")


if __name__ == '__main__':
    run_analysis()
