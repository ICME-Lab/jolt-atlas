use std::{collections::BTreeMap, error::Error};

use qwen2_prover::float::Rotary;

const TOP_K: usize = 16;

fn main() -> Result<(), Box<dyn Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let tok = root.join("atlas-onnx-tracer/models/qwen/tokenizer.json");
    let model = root.join("atlas-onnx-tracer/models/qwen/model.safetensors");

    let mut layers = qwen2_prover::LAYERS;
    let mut texts = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--layers" => {
                let value = args.next().ok_or("--layers requires a value")?;
                layers = value.parse::<usize>()?.min(qwen2_prover::LAYERS);
            }
            "--text" => {
                texts.push(args.next().ok_or("--text requires a value")?);
            }
            other => texts.push(other.to_string()),
        }
    }
    if texts.is_empty() {
        texts = vec![
            "hello world this is a test".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
            "mathematics and proofs require careful reasoning".to_string(),
        ];
    }

    let bytes = std::fs::read(model)?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    let r = Rotary::new();

    let mut runs = Vec::new();
    for text in &texts {
        let ids = qwen2_prover::text::tokenize(&tok, text)?;
        let mut x = qwen2_prover::float::embed_from_safetensors(&st, &ids)?;
        let mut sites = BTreeMap::new();
        for layer in 0..layers {
            let w = qwen2_prover::float::load_layer_from_safetensors(&st, layer)?;
            x = inspect_layer(layer, &x, &w, &r, &mut sites);
        }
        runs.push(Run {
            text: text.clone(),
            sites,
        });
    }

    print_summary(&runs);
    Ok(())
}

struct Run {
    text: String,
    sites: BTreeMap<String, Vec<Outlier>>,
}

#[derive(Clone)]
struct Outlier {
    token: usize,
    channel: usize,
    abs: f32,
}

fn inspect_layer(
    layer: usize,
    x: &[f32],
    w: &qwen2_prover::float::LayerWeights,
    r: &Rotary,
    sites: &mut BTreeMap<String, Vec<Outlier>>,
) -> Vec<f32> {
    let n1 = qwen2_prover::float::rms_norm(x, &w.ln1, qwen2_prover::SEQ, qwen2_prover::HIDDEN);

    let mut q = matmul_site(
        layer,
        "q_proj",
        &n1,
        &w.wq,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::HIDDEN,
        sites,
    );
    qwen2_prover::float::add_rows(&mut q, &w.bq);
    let mut k = matmul_site(
        layer,
        "k_proj",
        &n1,
        &w.wk,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM,
        sites,
    );
    qwen2_prover::float::add_rows(&mut k, &w.bk);
    let mut v = matmul_site(
        layer,
        "v_proj",
        &n1,
        &w.wv,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM,
        sites,
    );
    qwen2_prover::float::add_rows(&mut v, &w.bv);

    let q = qwen2_prover::float::rope(&q, &r.rq);
    let k = qwen2_prover::float::rope(&k, &r.rk);
    let s = qwen2_prover::float::score_qk(&q, &k);
    let p = qwen2_prover::float::softmax(
        &s,
        qwen2_prover::HEADS * qwen2_prover::SEQ,
        qwen2_prover::SEQ,
    );
    let c = qwen2_prover::float::attn_v(&p, &v);
    let a = matmul_site(
        layer,
        "o_proj",
        &c,
        &w.wo,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::HIDDEN,
        sites,
    );
    let h = qwen2_prover::float::add(x, &a);
    let n2 = qwen2_prover::float::rms_norm(&h, &w.ln2, qwen2_prover::SEQ, qwen2_prover::HIDDEN);

    let g = matmul_site(
        layer,
        "gate_proj",
        &n2,
        &w.wg,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
        sites,
    );
    let u = matmul_site(
        layer,
        "up_proj",
        &n2,
        &w.wu,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
        sites,
    );
    let m = qwen2_prover::float::silu_mul(&g, &u);
    record_site(
        layer,
        "mlp_mid",
        &m,
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        sites,
    );
    let d = matmul_site(
        layer,
        "down_proj",
        &m,
        &w.wd,
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        qwen2_prover::HIDDEN,
        sites,
    );
    qwen2_prover::float::add(&h, &d)
}

fn matmul_site(
    layer: usize,
    name: &str,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    sites: &mut BTreeMap<String, Vec<Outlier>>,
) -> Vec<f32> {
    let y = qwen2_prover::float::matmul(a, b, m, k, n);
    record_site(layer, name, &y, m, n, sites);
    y
}

fn record_site(
    layer: usize,
    name: &str,
    y: &[f32],
    rows: usize,
    cols: usize,
    sites: &mut BTreeMap<String, Vec<Outlier>>,
) {
    let mut top = Vec::new();
    for row in 0..rows {
        for col in 0..cols {
            let abs = y[row * cols + col].abs();
            if top.len() < TOP_K {
                top.push(Outlier {
                    token: row,
                    channel: col,
                    abs,
                });
                top.sort_by(|a, b| b.abs.total_cmp(&a.abs));
            } else if abs > top.last().unwrap().abs {
                *top.last_mut().unwrap() = Outlier {
                    token: row,
                    channel: col,
                    abs,
                };
                top.sort_by(|a, b| b.abs.total_cmp(&a.abs));
            }
        }
    }
    sites.insert(format!("L{layer}.{name}"), top);
}

fn print_summary(runs: &[Run]) {
    for (idx, run) in runs.iter().enumerate() {
        println!("=== text {idx}: {:?} ===", run.text);
    }
    println!();
    println!(
        "{:<16} {:>9} {:>9} {:>9}  {}",
        "site", "ch_ovlp", "pos_ovlp", "max_ch", "top channels per text"
    );

    let Some(base) = runs.first() else {
        return;
    };
    for site in base.sites.keys() {
        let Some(all) = runs
            .iter()
            .map(|run| run.sites.get(site))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        let ch_overlap = overlap_channels(&all);
        let pos_overlap = overlap_positions(&all);
        let max_ch_same = all
            .windows(2)
            .all(|pair| pair[0].first().unwrap().channel == pair[1].first().unwrap().channel);
        let desc = all
            .iter()
            .map(|tops| {
                tops.iter()
                    .take(4)
                    .map(|o| format!("t{}:c{}:{:.1}", o.token, o.channel, o.abs))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .collect::<Vec<_>>()
            .join(" | ");
        println!(
            "{site:<16} {ch_overlap:>8.2}% {pos_overlap:>8.2}% {:>9}  {desc}",
            if max_ch_same { "yes" } else { "no" }
        );
    }
}

fn overlap_channels(all: &[&Vec<Outlier>]) -> f64 {
    let base = all[0];
    let mut hits = 0usize;
    for item in base {
        if all[1..]
            .iter()
            .all(|tops| tops.iter().any(|other| other.channel == item.channel))
        {
            hits += 1;
        }
    }
    100.0 * hits as f64 / base.len() as f64
}

fn overlap_positions(all: &[&Vec<Outlier>]) -> f64 {
    let base = all[0];
    let mut hits = 0usize;
    for item in base {
        if all[1..].iter().all(|tops| {
            tops.iter()
                .any(|other| other.token == item.token && other.channel == item.channel)
        }) {
            hits += 1;
        }
    }
    100.0 * hits as f64 / base.len() as f64
}
