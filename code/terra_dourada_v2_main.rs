// ================================================================
// TERRA DOURADA — Dataset Sanitizer v2.0
// ================================================================
//
// Detecta lixo de web scraping em datasets de texto e código.
// Duas camadas de detecção:
//   1. Features lexicais (comp_med, pct_lixo, pct_nav, pct_cod)
//      → detecta lixo pela composição do vocabulário
//   2. FXL Turbo (janela histórica de contexto)
//      → detecta rupturas abruptas de assunto
//
// Compilar:
//   cargo build --release
//
// Uso:
//   ./tds --demo
//   ./tds -i dataset.txt   -o limpo.txt
//   ./tds -i dataset.jsonl -o limpo.jsonl
//   ./tds -i enorme.jsonl  -o limpo.jsonl --threads 8 --relatorio rel.json
//   ./tds -i codigo.jsonl  -o limpo.jsonl --codigo
// ================================================================

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use clap::Parser;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ================================================================
// CLI
// ================================================================

#[derive(Parser, Debug)]
#[command(
    name  = "tds",
    about = "Terra Dourada Dataset Sanitizer v2 — detecta lixo de web scraping"
)]
struct Cli {
    /// Arquivo de entrada (.txt ou .jsonl)
    #[arg(short, long)]
    input: Option<String>,

    /// Arquivo de saída para documentos aprovados
    #[arg(short, long)]
    output: Option<String>,

    /// Salvar documentos rejeitados aqui (opcional)
    #[arg(short, long)]
    rejeitados: Option<String>,

    /// Salvar relatório JSON aqui (opcional)
    #[arg(long)]
    relatorio: Option<String>,

    /// Sensibilidade: 0.0 = permissivo, 1.0 = rigoroso (default: 0.5)
    #[arg(short = 's', long, default_value = "0.5")]
    sensibilidade: f64,

    /// Janela do FXL Turbo em sentenças (default: 10)
    #[arg(short, long, default_value = "10")]
    janela: usize,

    /// Número de threads (default: CPUs disponíveis)
    #[arg(long, default_value = "0")]
    threads: usize,

    /// Máximo de documentos a processar (0 = todos)
    #[arg(long, default_value = "0")]
    max_docs: usize,

    /// Modo código-fonte (tokenização específica)
    #[arg(long)]
    codigo: bool,

    /// Mostra exemplos e explica cada critério
    #[arg(long)]
    demo: bool,

    /// Mostra score de cada documento durante o processamento
    #[arg(short, long)]
    verbose: bool,
}

// ================================================================
// PALAVRAS-CHAVE — a base das features lexicais
// ================================================================

/// Palavras típicas de lixo de navegação/spam.
/// Presença alta = documento suspeito.
fn lixo_words() -> HashSet<&'static str> {
    [
        "copyright","subscribe","newsletter","click","terms","privacy",
        "policy","contact","advertisement","sponsored","rights","reserved",
        "follow","download","share","facebook","instagram","twitter",
        "loading","error","login","signup","cookie","accept","affiliate",
        "rate","review","app","limited","offer","today","now","free",
        "home","about","sign","related","posts","articles","youtube",
        "tiktok","linkedin","pinterest","whatsapp","telegram","discord",
        "promo","coupon","discount","sale","deal","checkout","cart",
        "gdpr","unsubscribe","optout","spam","phishing","malware",
    ].iter().copied().collect()
}

/// Tokens típicos de código-fonte misturado com texto.
fn code_tokens() -> HashSet<&'static str> {
    [
        "def","class","return","import","function","select","from","where",
        "html","body","docker","git","sudo","npm","pip","const","let",
        "var","fn","pub","struct","impl","http","localhost","sql","json",
        "api","server","client","curl","chmod","bash","install","require",
        "include","namespace","printf","scanf","malloc","typedef","pragma",
        "endif","ifndef","ifdef","cmake","makefile","dockerfile","gradle",
        "webpack","babel","eslint","pytest","cargo","rustup","rustc",
    ].iter().copied().collect()
}

/// Palavras típicas de menus/navegação de sites.
fn nav_words() -> HashSet<&'static str> {
    [
        "home","about","contact","privacy","terms","service","newsletter",
        "subscribe","follow","copyright","reserved","cookie","download",
        "sponsored","affiliate","advertisement","sitemap","breadcrumb",
        "pagination","prev","next","read","more","back","top","menu",
        "navigation","sidebar","footer","header","widget","modal","popup",
    ].iter().copied().collect()
}

// ================================================================
// TOKENIZADOR
// ================================================================

fn canon(texto: &str) -> String {
    let mut out = String::with_capacity(texto.len());
    let mut espaco = true;
    for ch in texto.chars() {
        if ch.is_alphanumeric() {
            out.push(ch.to_lowercase().next().unwrap());
            espaco = false;
        } else if !espaco {
            out.push(' ');
            espaco = true;
        }
    }
    out.trim_end().to_string()
}

fn tokenize(texto: &str) -> Vec<String> {
    canon(texto)
        .split_whitespace()
        .filter(|w| w.len() >= 2)
        .map(String::from)
        .collect()
}

fn tokenize_codigo(texto: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut atual = String::new();
    for ch in texto.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            atual.push(ch.to_lowercase().next().unwrap());
        } else {
            if atual.len() >= 2 {
                // token completo
                tokens.push(atual.clone());
                // expande snake_case
                for part in atual.split('_') {
                    if part.len() >= 2 && part != atual.as_str() {
                        tokens.push(part.to_string());
                    }
                }
            }
            atual.clear();
        }
    }
    if atual.len() >= 2 {
        tokens.push(atual);
    }
    tokens
}

// ================================================================
// FXL TURBO — janela histórica (fiel ao original .rs)
// ================================================================

#[derive(Clone)]
struct FXLTurbo {
    janela:    usize,
    limiar:    f64,
    max_seq:   usize,
    historico: VecDeque<f64>,
    rupt_seq:  usize,
    total_rupt: usize,
}

impl FXLTurbo {
    fn new(janela: usize, limiar: f64, max_seq: usize) -> Self {
        Self {
            janela,
            limiar,
            max_seq,
            historico: VecDeque::with_capacity(janela),
            rupt_seq: 0,
            total_rupt: 0,
        }
    }

    /// contexto(t) = (1/N) * Σ [ 1 - |sim(t) - sim(t-i)| ]
    fn calcular_contexto(&self, sim: f64) -> f64 {
        if self.historico.is_empty() { return sim; }
        let n = self.historico.len() as f64;
        let est: f64 = self.historico.iter()
            .map(|&p| 1.0 - (sim - p).abs())
            .sum::<f64>() / n;
        est.clamp(0.0, 1.0)
    }

    fn update(&mut self, sim: f64) -> (f64, bool) {
        let ctx = self.calcular_contexto(sim);
        if self.historico.len() >= self.janela {
            self.historico.pop_front();
        }
        self.historico.push_back(sim);
        let rupt = ctx < self.limiar;
        if rupt { self.rupt_seq += 1; self.total_rupt += 1; }
        else    { self.rupt_seq = 0; }
        let bloqueado = self.rupt_seq >= self.max_seq;
        (ctx, bloqueado)
    }

    fn reset(&mut self) {
        self.historico.clear();
        self.rupt_seq = 0;
    }
}

// ================================================================
// FEATURES LEXICAIS
// ================================================================

#[derive(Debug, Clone, Serialize)]
pub struct FeatureLexical {
    // Lexicais diretas
    pub pct_lixo:    f64,   // % palavras de spam/nav
    pub pct_cod:     f64,   // % tokens de código
    pub pct_nav:     f64,   // % palavras de menu/navegação
    pub limpeza:     f64,   // score de limpeza combinado
    pub riqueza:     f64,   // diversidade de vocabulário
    pub entropia:    f64,   // entropia do vocabulário
    // Estruturais
    pub comp_med:    f64,   // comprimento médio de frases (palavras)
    pub comp_var:    f64,   // variância do comprimento
    pub n_sents:     usize, // número de frases
    // FXL Turbo
    pub taxa_rupt:   f64,   // taxa de rupturas detectadas
    pub ctx_med:     f64,   // contexto FXL médio
    pub bloqueado:   bool,  // FXL detectou ruptura sistêmica
}

fn dividir_sentencas(texto: &str) -> Vec<String> {
    let mut sents  = Vec::new();
    let mut atual  = String::new();

    for ch in texto.chars() {
        atual.push(ch);
        if matches!(ch, '.' | '!' | '?') && atual.len() > 20 {
            let s = atual.trim().to_string();
            if !s.is_empty() { sents.push(s); }
            atual.clear();
        }
    }
    let tail = atual.trim().to_string();
    if tail.len() > 10 { sents.push(tail); }

    if sents.len() >= 2 { return sents; }

    // fallback: parágrafos
    let parags: Vec<String> = texto.lines()
        .map(|l| l.trim().to_string())
        .filter(|l| l.len() > 20)
        .collect();
    if parags.len() >= 2 { return parags; }

    // fallback: divide ao meio
    if texto.len() > 60 {
        let mid = texto.len() / 2;
        return vec![texto[..mid].trim().to_string(),
                    texto[mid..].trim().to_string()];
    }
    vec![texto.to_string()]
}

fn jaccard_words(a: &str, b: &str) -> f64 {
    let sa: HashSet<String> = tokenize(a).into_iter().collect();
    let sb: HashSet<String> = tokenize(b).into_iter().collect();
    if sa.is_empty() && sb.is_empty() { return 1.0; }
    if sa.is_empty() || sb.is_empty() { return 0.0; }
    let inter = sa.intersection(&sb).count() as f64;
    let union = sa.union(&sb).count() as f64;
    inter / union
}

fn extrair_features(texto: &str, fxl: &mut FXLTurbo, modo_cod: bool) -> FeatureLexical {
    fxl.reset();

    let lixo = lixo_words();
    let code = code_tokens();
    let nav  = nav_words();

    let tokens: Vec<String> = if modo_cod {
        tokenize_codigo(texto)
    } else {
        tokenize(texto)
    };

    let tot = tokens.len().max(1) as f64;
    let unique = tokens.iter().collect::<HashSet<_>>().len() as f64;

    let n_lixo = tokens.iter().filter(|w| lixo.contains(w.as_str())).count();
    let n_cod  = tokens.iter().filter(|w| code.contains(w.as_str())).count();
    let n_nav  = tokens.iter().filter(|w| nav.contains(w.as_str())).count();

    let pct_lixo = n_lixo as f64 / tot;
    let pct_cod  = n_cod  as f64 / tot;
    let pct_nav  = n_nav  as f64 / tot;
    let limpeza  = (1.0 - pct_lixo * 0.5 - pct_cod * 0.3 - pct_nav * 0.2).clamp(0.0, 1.0);
    let riqueza  = unique / tot;

    // Entropia
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for t in &tokens { *freq.entry(t).or_insert(0) += 1; }
    let n_tot = freq.values().sum::<usize>() as f64;
    let entropia = if n_tot > 0.0 {
        -freq.values()
            .map(|&c| { let p = c as f64 / n_tot; p * p.ln() })
            .sum::<f64>()
    } else { 0.0 };

    // Comprimento de frases
    let sents = dividir_sentencas(texto);
    let comprimentos: Vec<f64> = sents.iter()
        .map(|s| tokenize(s).len() as f64)
        .collect();
    let comp_med = if comprimentos.is_empty() { 0.0 }
        else { comprimentos.iter().sum::<f64>() / comprimentos.len() as f64 };
    let comp_var = if comprimentos.len() > 1 {
        let v = comprimentos.iter()
            .map(|&c| (c - comp_med).powi(2))
            .sum::<f64>() / comprimentos.len() as f64;
        v.sqrt()
    } else { 0.0 };

    // FXL Turbo com Jaccard entre pares consecutivos
    let mut taxa_rupt = 0.0;
    let mut ctx_med   = 1.0;
    let mut bloqueado = false;

    if sents.len() >= 2 {
        let sims: Vec<f64> = (0..sents.len()-1)
            .map(|i| jaccard_words(&sents[i], &sents[i+1]))
            .collect();
        let mut ctxs = Vec::with_capacity(sims.len());
        let mut rupt_count = 0usize;
        for &sim in &sims {
            let (ctx, blk) = fxl.update(sim);
            ctxs.push(ctx);
            if sim < fxl.limiar { rupt_count += 1; }
            if blk { bloqueado = true; }
        }
        taxa_rupt = rupt_count as f64 / sims.len() as f64;
        ctx_med   = ctxs.iter().sum::<f64>() / ctxs.len() as f64;
    }

    FeatureLexical {
        pct_lixo, pct_cod, pct_nav, limpeza,
        riqueza, entropia,
        comp_med, comp_var, n_sents: sents.len(),
        taxa_rupt, ctx_med, bloqueado,
    }
}

// ================================================================
// DECISÃO — combina features em score final
// ================================================================

/// Converte sensibilidade 0–1 em thresholds internos.
struct Thresholds {
    comp_med_min:   f64,  // comprimento mínimo de frases
    pct_lixo_max:   f64,  // % máximo de palavras de lixo
    pct_cod_max:    f64,  // % máximo de tokens de código (em texto)
    pct_nav_max:    f64,  // % máximo de palavras de nav
    taxa_rupt_max:  f64,  // taxa máxima de rupturas FXL
    limpeza_min:    f64,  // score mínimo de limpeza
}

impl Thresholds {
    /// sensibilidade: 0.0 = muito permissivo, 1.0 = muito rigoroso
    fn from_sensibilidade(s: f64) -> Self {
        let s = s.clamp(0.0, 1.0);
        Self {
            comp_med_min:  3.0  + s * 4.0,   // 3–7 palavras mínimo por frase
            pct_lixo_max:  0.15 - s * 0.10,  // 15% → 5%
            pct_cod_max:   0.15 - s * 0.08,  // 15% → 7% (em modo texto)
            pct_nav_max:   0.20 - s * 0.12,  // 20% → 8%
            taxa_rupt_max: 0.50 - s * 0.25,  // 50% → 25%
            limpeza_min:   0.60 + s * 0.20,  // 60% → 80%
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ResultadoDoc {
    pub id:       usize,
    pub aprovado: bool,
    pub score:    f64,   // 0 = lixo, 1 = perfeito
    pub motivo:   String,
    pub features: FeatureLexical,
}

fn avaliar_documento(
    id: usize,
    texto: &str,
    fxl: &mut FXLTurbo,
    thresh: &Thresholds,
    modo_cod: bool,
) -> ResultadoDoc {
    let feats = extrair_features(texto, fxl, modo_cod);

    // Calcula score composto (0–1, maior = melhor qualidade)
    let score = calc_score(&feats);

    // Critérios de rejeição
    let motivo = checar_criterios(&feats, thresh, modo_cod);
    let aprovado = motivo.is_empty();

    ResultadoDoc { id, aprovado, score, motivo, features: feats }
}

fn calc_score(f: &FeatureLexical) -> f64 {
    // Componentes positivos
    let s_limpeza  = f.limpeza;
    let s_comp     = (f.comp_med / 15.0).min(1.0);
    let s_riqueza  = f.riqueza.min(1.0);
    let s_ctx      = f.ctx_med;

    // Componentes negativos
    let pen_lixo   = f.pct_lixo * 2.0;
    let pen_cod    = f.pct_cod  * 1.5;
    let pen_nav    = f.pct_nav  * 1.5;
    let pen_rupt   = f.taxa_rupt;
    let pen_blk    = if f.bloqueado { 0.3 } else { 0.0 };

    let pos = (s_limpeza * 0.35 + s_comp * 0.30 + s_riqueza * 0.20 + s_ctx * 0.15)
              .clamp(0.0, 1.0);
    let neg = (pen_lixo + pen_cod + pen_nav + pen_rupt + pen_blk)
              .clamp(0.0, 1.0);

    (pos - neg * 0.5).clamp(0.0, 1.0)
}

fn checar_criterios(f: &FeatureLexical, t: &Thresholds, modo_cod: bool) -> String {
    // Bloqueio FXL (ruptura sistêmica)
    if f.bloqueado {
        return "FXL: ruptura sistêmica de contexto".to_string();
    }

    // Frases muito curtas (típico de lixo de navegação)
    if f.comp_med < t.comp_med_min && f.n_sents > 1 {
        return format!("frases curtas: {:.1} palavras/frase (min {:.1})",
                       f.comp_med, t.comp_med_min);
    }

    // Muitas palavras de lixo
    if f.pct_lixo > t.pct_lixo_max {
        return format!("pct_lixo: {:.0}% (max {:.0}%)",
                       f.pct_lixo * 100.0, t.pct_lixo_max * 100.0);
    }

    // Muitas palavras de navegação
    if f.pct_nav > t.pct_nav_max {
        return format!("pct_nav: {:.0}% (max {:.0}%)",
                       f.pct_nav * 100.0, t.pct_nav_max * 100.0);
    }

    // Código misturado em modo texto
    if !modo_cod && f.pct_cod > t.pct_cod_max {
        return format!("pct_codigo: {:.0}% (max {:.0}%)",
                       f.pct_cod * 100.0, t.pct_cod_max * 100.0);
    }

    // Limpeza baixa
    if f.limpeza < t.limpeza_min {
        return format!("limpeza: {:.2} (min {:.2})",
                       f.limpeza, t.limpeza_min);
    }

    // Muitas rupturas FXL
    if f.taxa_rupt > t.taxa_rupt_max && f.n_sents >= 3 {
        return format!("taxa_ruptura: {:.0}% (max {:.0}%)",
                       f.taxa_rupt * 100.0, t.taxa_rupt_max * 100.0);
    }

    String::new() // aprovado
}

// ================================================================
// LEITURA DE DATASET
// ================================================================

#[derive(Deserialize)]
struct JsonlDoc {
    #[serde(alias = "text", alias = "content", alias = "body",
            alias = "document", alias = "passage")]
    text: Option<String>,
}

enum Formato { Txt, Jsonl }

fn detectar_formato(path: &str) -> Formato {
    if path.ends_with(".jsonl") || path.ends_with(".json") {
        Formato::Jsonl
    } else {
        Formato::Txt
    }
}

fn ler_documentos(path: &str, max: usize) -> Vec<String> {
    let file   = File::open(path).unwrap_or_else(|_| panic!("Arquivo não encontrado: {}", path));
    let reader = BufReader::new(file);

    let mut docs = Vec::new();

    match detectar_formato(path) {
        Formato::Jsonl => {
            for linha in reader.lines().flatten() {
                if linha.trim().is_empty() { continue; }
                if let Ok(obj) = serde_json::from_str::<JsonlDoc>(&linha) {
                    if let Some(t) = obj.text {
                        let t = t.trim().to_string();
                        if !t.is_empty() {
                            docs.push(t);
                            if max > 0 && docs.len() >= max { break; }
                        }
                    }
                }
            }
        }
        Formato::Txt => {
            let mut atual: Vec<String> = Vec::new();
            for linha in reader.lines().flatten() {
                if linha.trim().is_empty() {
                    if !atual.is_empty() {
                        docs.push(atual.join(" "));
                        atual.clear();
                        if max > 0 && docs.len() >= max { break; }
                    }
                } else {
                    atual.push(linha.trim().to_string());
                }
            }
            if !atual.is_empty() {
                docs.push(atual.join(" "));
            }
        }
    }
    docs
}

fn escrever_doc(writer: &mut impl Write, texto: &str, fmt: &Formato) {
    match fmt {
        Formato::Jsonl => {
            let obj = serde_json::json!({"text": texto});
            writeln!(writer, "{}", obj).unwrap();
        }
        Formato::Txt => {
            writeln!(writer, "{}\n", texto).unwrap();
        }
    }
}

// ================================================================
// RELATÓRIO
// ================================================================

#[derive(Serialize)]
struct Relatorio {
    config:     ConfigRel,
    resumo:     ResumoRel,
    documentos: Vec<ResultadoDoc>,
}
#[derive(Serialize)]
struct ConfigRel {
    sensibilidade: f64,
    janela_fxl:   usize,
    modo_codigo:  bool,
    input:        String,
}
#[derive(Serialize)]
struct ResumoRel {
    total:           usize,
    aprovados:       usize,
    rejeitados:      usize,
    taxa_aprovacao:  f64,
    score_medio_ok:  f64,
    score_medio_rej: f64,
    tempo_segundos:  f64,
    docs_por_segundo:f64,
}

// ================================================================
// PIPELINE PRINCIPAL
// ================================================================

fn pipeline(cli: &Cli) {
    let input  = cli.input.as_ref().expect("--input obrigatório");
    let output = cli.output.as_ref().expect("--output obrigatório");
    let thresh = Thresholds::from_sensibilidade(cli.sensibilidade);

    // Limiar FXL: adapta à sensibilidade
    let fxl_limiar = 0.05 + cli.sensibilidade * 0.10;

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║     TERRA DOURADA — Dataset Sanitizer v2.0           ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!("  Input        : {}", input);
    println!("  Output       : {}", output);
    println!("  Sensibilidade: {:.1} (0=permissivo 1=rigoroso)", cli.sensibilidade);
    println!("  Janela FXL   : {}", cli.janela);
    println!("  Modo         : {}", if cli.codigo { "CÓDIGO" } else { "TEXTO" });
    println!("──────────────────────────────────────────────────────");

    let t0 = Instant::now();

    // 1. Leitura
    print!("  [1/3] Lendo... ");
    let _ = std::io::stdout().flush();
    let docs = ler_documentos(input, cli.max_docs);
    println!("{} documentos", docs.len());
    if docs.is_empty() {
        eprintln!("  Nenhum documento encontrado em {}", input);
        return;
    }

    // 2. Avaliação paralela
    println!("  [2/3] Avaliando...");
    if cli.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build_global()
            .ok();
    }

    let total     = docs.len();
    let progresso = Arc::new(Mutex::new(0usize));
    let fxl_lim   = fxl_limiar;
    let fxl_jan   = cli.janela;
    let cod       = cli.codigo;
    let th        = Arc::new(thresh);

    let mut resultados: Vec<(usize, ResultadoDoc)> = docs
        .par_iter()
        .enumerate()
        .map(|(i, texto)| {
            let mut fxl   = FXLTurbo::new(fxl_jan, fxl_lim, 2);
            let thresh_l  = Arc::clone(&th);
            let mut res   = avaliar_documento(i+1, texto, &mut fxl, &thresh_l, cod);
            res.id = i + 1;

            let mut p = progresso.lock().unwrap();
            *p += 1;
            if *p % 500 == 0 || *p == total {
                let pct = *p as f64 / total as f64 * 100.0;
                let bar_len = (*p * 30 / total).min(30);
                let bar = "█".repeat(bar_len) + &"░".repeat(30 - bar_len);
                print!("\r  [2/3] [{}] {:.0}%  ", bar, pct);
                let _ = std::io::stdout().flush();
            }
            drop(p);
            (i, res)
        })
        .collect();

    resultados.sort_by_key(|(i, _)| *i);
    println!("\r  [2/3] Avaliação concluída ({} docs)          ", total);

    // 3. Escrita
    print!("  [3/3] Escrevendo saída...");
    let fmt_out = detectar_formato(output);
    let mut f_out = BufWriter::new(
        File::create(output).expect("Não foi possível criar arquivo de saída")
    );
    let mut f_rej = cli.rejeitados.as_ref().map(|p| {
        BufWriter::new(File::create(p).expect("Não foi possível criar arquivo de rejeitados"))
    });

    let mut n_apr = 0usize; let mut n_rej = 0usize;
    let mut soma_ok = 0.0f64; let mut soma_rej = 0.0f64;

    for (i, res) in &resultados {
        let texto = &docs[*i];
        if res.aprovado {
            escrever_doc(&mut f_out, texto, &fmt_out);
            n_apr  += 1;
            soma_ok += res.score;
        } else {
            n_rej  += 1;
            soma_rej += res.score;
            if let Some(ref mut fw) = f_rej {
                match fmt_out {
                    Formato::Jsonl => {
                        let obj = serde_json::json!({
                            "text":   texto,
                            "motivo": res.motivo,
                            "score":  res.score,
                        });
                        writeln!(fw, "{}", obj).unwrap();
                    }
                    Formato::Txt => {
                        writeln!(fw, "[REJEITADO score={:.3} motivo={}]\n{}\n",
                                 res.score, res.motivo, texto).unwrap();
                    }
                }
            }
            if cli.verbose {
                println!("  ✗ doc#{:5}  score={:.3}  {}", res.id, res.score, res.motivo);
            }
        }
    }
    println!(" ok");

    let elapsed = t0.elapsed().as_secs_f64();
    let dps     = total as f64 / elapsed.max(0.001);

    // Resultado
    println!();
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  RESULTADO                                           ║");
    println!("╠══════════════════════════════════════════════════════╣");
    println!("║  Total processado : {:>8}                          ║", total);
    println!("║  Aprovados        : {:>8}  ({:>5.1}%)               ║",
             n_apr, n_apr as f64 / total as f64 * 100.0);
    println!("║  Rejeitados       : {:>8}  ({:>5.1}%)               ║",
             n_rej, n_rej as f64 / total as f64 * 100.0);
    println!("║  Score médio OK   : {:>8.4}                          ║",
             soma_ok / n_apr.max(1) as f64);
    println!("║  Score médio REJ  : {:>8.4}                          ║",
             soma_rej / n_rej.max(1) as f64);
    println!("║  Tempo            : {:>7.1}s                          ║", elapsed);
    println!("║  Velocidade       : {:>7.0} docs/s                   ║", dps);
    println!("╠══════════════════════════════════════════════════════╣");
    println!("║  Estimativas de escala:                               ║");
    for (n, lbl) in [(10_000u64,"10k"),(100_000,"100k"),(1_000_000,"1M"),(100_000_000,"100M")] {
        let t = n as f64 / dps;
        let ts = if t < 60.0 { format!("{:.0}s",t) }
                 else if t < 3600.0 { format!("{:.1}min",t/60.0) }
                 else { format!("{:.1}h",t/3600.0) };
        println!("║    {:>10} docs → {:>10}                      ║", lbl, ts);
    }
    println!("╚══════════════════════════════════════════════════════╝");
    println!("  Dataset limpo: {}", output);
    if let Some(p) = &cli.rejeitados { println!("  Rejeitados   : {}", p); }

    // Relatório JSON
    if let Some(rel_path) = &cli.relatorio {
        let rel = Relatorio {
            config: ConfigRel {
                sensibilidade: cli.sensibilidade,
                janela_fxl: cli.janela,
                modo_codigo: cli.codigo,
                input: input.clone(),
            },
            resumo: ResumoRel {
                total, aprovados: n_apr, rejeitados: n_rej,
                taxa_aprovacao: n_apr as f64 / total as f64 * 100.0,
                score_medio_ok:  soma_ok  / n_apr.max(1) as f64,
                score_medio_rej: soma_rej / n_rej.max(1) as f64,
                tempo_segundos: elapsed,
                docs_por_segundo: dps,
            },
            documentos: resultados.iter().map(|(_,r)| r.clone()).collect(),
        };
        let json = serde_json::to_string_pretty(&rel).unwrap();
        std::fs::write(rel_path, json).expect("Não foi possível salvar relatório");
        println!("  Relatório    : {}", rel_path);
    }
}

// ================================================================
// DEMO
// ================================================================

fn demo() {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║     TERRA DOURADA v2 — Demo                          ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    let casos: &[(&str, bool, &str)] = &[
        ("ML coerente", true,
         "Neural networks learn representations from large datasets. Deep learning uses multiple layers to extract hierarchical features. Training involves minimizing a loss function through gradient descent. The optimizer updates weights based on computed gradients each step. Regularization prevents the model from memorizing training examples."),
        ("Ciência coerente", true,
         "Photosynthesis converts sunlight into chemical energy stored in glucose. Chlorophyll absorbs light primarily in the red and blue spectrum bands. Carbon dioxide from air and water from soil are the raw materials used. The Calvin cycle uses ATP and NADPH to synthesize organic compounds. Glucose produced fuels plant growth and cellular metabolic processes."),
        ("Código Rust coerente", true,
         "pub fn calcular_ctx(hist: &[f64], sim: f64) -> f64 { hist.iter().map(|&p| 1.0 - (sim-p).abs()).sum::<f64>() / hist.len() as f64 }\npub fn detectar_ruptura(ctx: f64, limiar: f64) -> bool { ctx < limiar }\npub fn atualizar_hist(h: &mut Vec<f64>, sim: f64, max: usize) { h.push(sim); if h.len() > max { h.remove(0); } }"),
        ("LIXO — navegação/spam", false,
         "Home About Contact Privacy Policy Terms of Service Newsletter Subscribe. Click here to read more related articles on our website today. Follow us on Facebook Twitter Instagram YouTube for daily updates. Copyright 2024 All Rights Reserved Terms and Conditions apply. Subscribe now for exclusive deals discounts and free offers available."),
        ("LIXO — tópicos aleatórios", false,
         "The Eiffel Tower stands three hundred meters tall in central Paris. My cat refuses to eat dry food for the past three weeks now. Bitcoin reached its all time high price in November of 2021. Spaghetti carbonara is made with eggs cheese guanciale and pepper. Jupiter has ninety five known moons orbiting around it."),
        ("LIXO — código misturado", false,
         "SELECT * FROM users WHERE active = 1 ORDER BY created_at DESC. def hello(): print('Hello World') return None exit. git commit -m 'fix critical bug' && git push origin main. docker run -d -p 8080:80 --name app nginx:latest always. import pandas as pd; df = pd.read_csv('data.csv') head."),
        ("LIXO — scraping com copyright", false,
         "Machine learning is transforming industries globally everywhere. Copyright 2024 All Rights Reserved Terms and Conditions Privacy. Neural networks use multiple layers for feature extraction tasks. Subscribe newsletter click here for more tutorials and free content. Gradient descent optimizes parameters by following loss surface."),
    ];

    let thresh = Thresholds::from_sensibilidade(0.5);
    let fxl_lim = 0.10;

    println!("  {:<35} {:>7}  {:>7}  {:<10}", "Documento", "Score", "GT", "Status");
    println!("  {}", "─".repeat(65));

    let mut corretos = 0;
    for (label, gt, texto) in casos {
        let mut fxl = FXLTurbo::new(8, fxl_lim, 2);
        let res = avaliar_documento(0, texto, &mut fxl, &thresh, false);
        let acerto = res.aprovado == *gt;
        if acerto { corretos += 1; }

        let gt_str  = if *gt { "COERENT" } else { "LIXO   " };
        let st_str  = if res.aprovado { "✓ APROVADO" } else { "✗ REJEITADO" };
        let err_str = if acerto { "" } else { " ← ERRO" };
        println!("  {:<35} {:>7.3}  {:>7}  {}{}",
                 label, res.score, gt_str, st_str, err_str);

        if !res.aprovado && !res.motivo.is_empty() {
            println!("  {:<35}          motivo: {}", "", res.motivo);
        }
    }

    let acc = corretos as f64 / casos.len() as f64 * 100.0;
    println!("\n  Acurácia demo: {:.0}% ({}/{})", acc, corretos, casos.len());

    println!("\n  Níveis de sensibilidade:");
    println!("  --sensibilidade 0.2  → permissivo  (~10% rejeitado)");
    println!("  --sensibilidade 0.5  → balanceado  (~25% rejeitado) ← default");
    println!("  --sensibilidade 0.8  → rigoroso    (~40% rejeitado)");
    println!("\n  Uso:");
    println!("  ./tds -i dataset.txt  -o limpo.txt");
    println!("  ./tds -i big.jsonl    -o limpo.jsonl --threads 8");
    println!("  ./tds -i code.jsonl   -o limpo.jsonl --codigo");
}

// ================================================================
// MAIN
// ================================================================

fn main() {
    let cli = Cli::parse();
    if cli.demo || cli.input.is_none() {
        demo();
    } else {
        pipeline(&cli);
    }
}
