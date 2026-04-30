use std::sync::mpsc;

fn bytes_i32(xs: &[i32]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

fn cfg2(a: u32, b: i32) -> Vec<u8> {
    let mut xs = Vec::with_capacity(16);
    xs.extend(a.to_ne_bytes());
    xs.extend(b.to_ne_bytes());
    xs.resize(16, 0);
    xs
}

fn cfg1(a: u32) -> Vec<u8> {
    let mut xs = Vec::with_capacity(16);
    xs.extend(a.to_ne_bytes());
    xs.resize(16, 0);
    xs
}

fn cfg3(a: u32, b: u32, c: i32) -> Vec<u8> {
    let mut xs = Vec::with_capacity(16);
    xs.extend(a.to_ne_bytes());
    xs.extend(b.to_ne_bytes());
    xs.extend(c.to_ne_bytes());
    xs.resize(16, 0);
    xs
}

fn cfg5(a: u32, b: u32, c: u32, d: u32, e: i32) -> Vec<u8> {
    let mut xs = Vec::with_capacity(32);
    xs.extend(a.to_ne_bytes());
    xs.extend(b.to_ne_bytes());
    xs.extend(c.to_ne_bytes());
    xs.extend(d.to_ne_bytes());
    xs.extend(e.to_ne_bytes());
    xs.resize(32, 0);
    xs
}

fn div_ceil(a: usize, b: usize) -> u32 {
    a.div_ceil(b) as u32
}

fn rot(seq: usize, heads: usize, head_dim: usize) -> Vec<i32> {
    let mut xs = vec![0; seq * heads * head_dim];
    for pos in 0..seq {
        for h in 0..heads {
            for p in 0..(head_dim / 2) {
                let freq = qwen2_prover::ROPE_THETA.powf(-((2 * p) as f64) / head_dim as f64);
                let t = pos as f64 * freq;
                let i = (pos * heads + h) * head_dim + 2 * p;
                xs[i] = (t.cos() * 256.0).round() as i32;
                xs[i + 1] = (t.sin() * 256.0).round() as i32;
            }
        }
    }
    xs
}

struct G {
    dev: wgpu::Device,
    queue: wgpu::Queue,
}

impl G {
    async fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("no wgpu adapter found");
        let (dev, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("qwen2-prover-layer0"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("failed to request wgpu device");
        Self { dev, queue }
    }

    fn buf(&self, label: &str, xs: &[i32]) -> wgpu::Buffer {
        let b = self.dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (xs.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&b, 0, &bytes_i32(xs));
        b
    }

    fn run(
        &self,
        src: &str,
        cfg: &[u8],
        ins: &[&wgpu::Buffer],
        out_len: usize,
        wg: (u32, u32, u32),
    ) -> wgpu::Buffer {
        let cfg_buf = self.dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cfg"),
            size: cfg.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&cfg_buf, 0, cfg);

        let out = self.dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: (out_len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut layout_entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        for i in 0..ins.len() {
            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: (i + 1) as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: (ins.len() + 1) as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let layout = self
            .dev
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("layout"),
                entries: &layout_entries,
            });

        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: cfg_buf.as_entire_binding(),
        }];
        for (i, b) in ins.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (i + 1) as u32,
                resource: b.as_entire_binding(),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: (ins.len() + 1) as u32,
            resource: out.as_entire_binding(),
        });
        let bind = self.dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind"),
            layout: &layout,
            entries: &entries,
        });
        let shader = self.dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let pl = self
            .dev
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pl"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipe = self
            .dev
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pipe"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let mut enc = self
            .dev
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(wg.0, wg.1, wg.2);
        }
        self.queue.submit([enc.finish()]);
        self.dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        out
    }

    fn read(&self, buf: &wgpu::Buffer, len: usize) -> Vec<i32> {
        let read = self.dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read"),
            size: (len * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .dev
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(buf, 0, &read, 0, (len * 4) as u64);
        self.queue.submit([enc.finish()]);

        let slice = read.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let view = slice.get_mapped_range();
        let xs = view
            .chunks_exact(4)
            .map(|b| i32::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        drop(view);
        read.unmap();
        xs
    }
}

fn run_layer(
    g: &G,
    x: &wgpu::Buffer,
    w: &[wgpu::Buffer],
    rsqrt: &wgpu::Buffer,
    silu: &wgpu::Buffer,
    exp: &wgpu::Buffer,
    rq: &wgpu::Buffer,
    rk: &wgpu::Buffer,
    rz: i32,
    sz: i32,
    ez: i32,
) -> wgpu::Buffer {
    let norm1 = g.run(
        qwen2_prover::RMS_NORM,
        &cfg3(qwen2_prover::SEQ as u32, qwen2_prover::HIDDEN as u32, rz),
        &[x, &w[qwen2_prover::weights::LN1], rsqrt],
        qwen2_prover::X_LEN,
        (qwen2_prover::SEQ as u32, 1, 1),
    );
    let q = g.run(
        qwen2_prover::MATMUL,
        &cfg3(
            qwen2_prover::SEQ as u32,
            qwen2_prover::HIDDEN as u32,
            qwen2_prover::HIDDEN as i32,
        ),
        &[&norm1, &w[qwen2_prover::weights::WQ]],
        qwen2_prover::Q_LEN,
        (
            div_ceil(qwen2_prover::HIDDEN, 16),
            div_ceil(qwen2_prover::SEQ, 16),
            1,
        ),
    );
    let k = g.run(
        qwen2_prover::MATMUL,
        &cfg3(
            qwen2_prover::SEQ as u32,
            (qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM) as u32,
            qwen2_prover::HIDDEN as i32,
        ),
        &[&norm1, &w[qwen2_prover::weights::WK]],
        qwen2_prover::KV_LEN,
        (
            div_ceil(qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM, 16),
            div_ceil(qwen2_prover::SEQ, 16),
            1,
        ),
    );
    let v = g.run(
        qwen2_prover::MATMUL,
        &cfg3(
            qwen2_prover::SEQ as u32,
            (qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM) as u32,
            qwen2_prover::HIDDEN as i32,
        ),
        &[&norm1, &w[qwen2_prover::weights::WV]],
        qwen2_prover::KV_LEN,
        (
            div_ceil(qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM, 16),
            div_ceil(qwen2_prover::SEQ, 16),
            1,
        ),
    );
    let q = g.run(
        qwen2_prover::ROPE,
        &cfg2((qwen2_prover::Q_LEN / 2) as u32, 0),
        &[&q, rq],
        qwen2_prover::Q_LEN,
        (div_ceil(qwen2_prover::Q_LEN / 2, 256), 1, 1),
    );
    let k = g.run(
        qwen2_prover::ROPE,
        &cfg2((qwen2_prover::KV_LEN / 2) as u32, 0),
        &[&k, rk],
        qwen2_prover::KV_LEN,
        (div_ceil(qwen2_prover::KV_LEN / 2, 256), 1, 1),
    );
    let score = g.run(
        qwen2_prover::SCORE_QK,
        &cfg5(
            qwen2_prover::SEQ as u32,
            qwen2_prover::HEADS as u32,
            qwen2_prover::KV_HEADS as u32,
            qwen2_prover::HEAD_DIM as u32,
            -32768,
        ),
        &[&q, &k],
        qwen2_prover::SCORE_LEN,
        (
            div_ceil(qwen2_prover::SEQ, 8),
            div_ceil(qwen2_prover::SEQ, 8),
            qwen2_prover::HEADS as u32,
        ),
    );
    let prob = g.run(
        qwen2_prover::SOFTMAX,
        &cfg3(
            (qwen2_prover::HEADS * qwen2_prover::SEQ) as u32,
            qwen2_prover::SEQ as u32,
            ez,
        ),
        &[&score, exp],
        qwen2_prover::SCORE_LEN,
        ((qwen2_prover::HEADS * qwen2_prover::SEQ) as u32, 1, 1),
    );
    let ctx = g.run(
        qwen2_prover::ATTN_V,
        &cfg5(
            qwen2_prover::SEQ as u32,
            qwen2_prover::HEADS as u32,
            qwen2_prover::KV_HEADS as u32,
            qwen2_prover::HEAD_DIM as u32,
            0,
        ),
        &[&prob, &v],
        qwen2_prover::X_LEN,
        (
            div_ceil(qwen2_prover::HEAD_DIM, 8),
            div_ceil(qwen2_prover::SEQ, 8),
            qwen2_prover::HEADS as u32,
        ),
    );
    let attn = g.run(
        qwen2_prover::MATMUL,
        &cfg3(
            qwen2_prover::SEQ as u32,
            qwen2_prover::HIDDEN as u32,
            qwen2_prover::HIDDEN as i32,
        ),
        &[&ctx, &w[qwen2_prover::weights::WO]],
        qwen2_prover::X_LEN,
        (
            div_ceil(qwen2_prover::HIDDEN, 16),
            div_ceil(qwen2_prover::SEQ, 16),
            1,
        ),
    );
    let h = g.run(
        qwen2_prover::ADD,
        &cfg1(qwen2_prover::X_LEN as u32),
        &[x, &attn],
        qwen2_prover::X_LEN,
        (div_ceil(qwen2_prover::X_LEN, 256), 1, 1),
    );
    let norm2 = g.run(
        qwen2_prover::RMS_NORM,
        &cfg3(qwen2_prover::SEQ as u32, qwen2_prover::HIDDEN as u32, rz),
        &[&h, &w[qwen2_prover::weights::LN2], rsqrt],
        qwen2_prover::X_LEN,
        (qwen2_prover::SEQ as u32, 1, 1),
    );
    let gate = g.run(
        qwen2_prover::MATMUL,
        &cfg3(
            qwen2_prover::SEQ as u32,
            qwen2_prover::INTERMEDIATE as u32,
            qwen2_prover::HIDDEN as i32,
        ),
        &[&norm2, &w[qwen2_prover::weights::WG]],
        qwen2_prover::MLP_LEN,
        (
            div_ceil(qwen2_prover::INTERMEDIATE, 16),
            div_ceil(qwen2_prover::SEQ, 16),
            1,
        ),
    );
    let up = g.run(
        qwen2_prover::MATMUL,
        &cfg3(
            qwen2_prover::SEQ as u32,
            qwen2_prover::INTERMEDIATE as u32,
            qwen2_prover::HIDDEN as i32,
        ),
        &[&norm2, &w[qwen2_prover::weights::WU]],
        qwen2_prover::MLP_LEN,
        (
            div_ceil(qwen2_prover::INTERMEDIATE, 16),
            div_ceil(qwen2_prover::SEQ, 16),
            1,
        ),
    );
    let mid = g.run(
        qwen2_prover::SILU_MUL,
        &cfg2(qwen2_prover::MLP_LEN as u32, sz),
        &[&gate, &up, silu],
        qwen2_prover::MLP_LEN,
        (div_ceil(qwen2_prover::MLP_LEN, 256), 1, 1),
    );
    let mlp = g.run(
        qwen2_prover::MATMUL,
        &cfg3(
            qwen2_prover::SEQ as u32,
            qwen2_prover::HIDDEN as u32,
            qwen2_prover::INTERMEDIATE as i32,
        ),
        &[&mid, &w[qwen2_prover::weights::WD]],
        qwen2_prover::X_LEN,
        (
            div_ceil(qwen2_prover::HIDDEN, 16),
            div_ceil(qwen2_prover::SEQ, 16),
            1,
        ),
    );
    g.run(
        qwen2_prover::ADD,
        &cfg1(qwen2_prover::X_LEN as u32),
        &[&h, &mlp],
        qwen2_prover::X_LEN,
        (div_ceil(qwen2_prover::X_LEN, 256), 1, 1),
    )
}

#[test]
#[ignore = "loads the downloaded Qwen2-0.5B safetensors and runs a full layer on Metal"]
fn layer0_text_embedding_smoke() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let path = root.join("atlas-onnx-tracer/models/qwen/model.safetensors");
    let tok = root.join("atlas-onnx-tracer/models/qwen/tokenizer.json");
    let ws = qwen2_prover::weights::load_layer(&path, 0).unwrap();
    let g = pollster::block_on(G::new());

    let x = qwen2_prover::text::embed_text(tok, &path, "hello").unwrap();
    let x = g.buf("x", &x);
    let w: Vec<_> = ws
        .iter()
        .enumerate()
        .map(|(i, x)| g.buf(&format!("w{i}"), x))
        .collect();
    let (rsqrt, rz) = qwen2_prover::lut::rsqrt(0, 1_000_000);
    let (silu, sz) = qwen2_prover::lut::silu(-32768, 32767);
    let (exp, ez) = qwen2_prover::lut::exp(-32768, 0);
    let rsqrt = g.buf("rsqrt", &rsqrt);
    let silu = g.buf("silu", &silu);
    let exp = g.buf("exp", &exp);
    let rq = g.buf(
        "rot_q",
        &rot(
            qwen2_prover::SEQ,
            qwen2_prover::HEADS,
            qwen2_prover::HEAD_DIM,
        ),
    );
    let rk = g.buf(
        "rot_k",
        &rot(
            qwen2_prover::SEQ,
            qwen2_prover::KV_HEADS,
            qwen2_prover::HEAD_DIM,
        ),
    );

    let norm1 = g.run(
        qwen2_prover::RMS_NORM,
        &cfg3(qwen2_prover::SEQ as u32, qwen2_prover::HIDDEN as u32, rz),
        &[&x, &w[qwen2_prover::weights::LN1], &rsqrt],
        qwen2_prover::X_LEN,
        (qwen2_prover::SEQ as u32, 1, 1),
    );
    let q = g.run(
        qwen2_prover::MATMUL,
        &cfg3(
            qwen2_prover::SEQ as u32,
            qwen2_prover::HIDDEN as u32,
            qwen2_prover::HIDDEN as i32,
        ),
        &[&norm1, &w[qwen2_prover::weights::WQ]],
        qwen2_prover::Q_LEN,
        (
            div_ceil(qwen2_prover::HIDDEN, 16),
            div_ceil(qwen2_prover::SEQ, 16),
            1,
        ),
    );
    let k = g.run(
        qwen2_prover::MATMUL,
        &cfg3(
            qwen2_prover::SEQ as u32,
            (qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM) as u32,
            qwen2_prover::HIDDEN as i32,
        ),
        &[&norm1, &w[qwen2_prover::weights::WK]],
        qwen2_prover::KV_LEN,
        (
            div_ceil(qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM, 16),
            div_ceil(qwen2_prover::SEQ, 16),
            1,
        ),
    );
    let v = g.run(
        qwen2_prover::MATMUL,
        &cfg3(
            qwen2_prover::SEQ as u32,
            (qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM) as u32,
            qwen2_prover::HIDDEN as i32,
        ),
        &[&norm1, &w[qwen2_prover::weights::WV]],
        qwen2_prover::KV_LEN,
        (
            div_ceil(qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM, 16),
            div_ceil(qwen2_prover::SEQ, 16),
            1,
        ),
    );
    let q = g.run(
        qwen2_prover::ROPE,
        &cfg2((qwen2_prover::Q_LEN / 2) as u32, 0),
        &[&q, &rq],
        qwen2_prover::Q_LEN,
        (div_ceil(qwen2_prover::Q_LEN / 2, 256), 1, 1),
    );
    let k = g.run(
        qwen2_prover::ROPE,
        &cfg2((qwen2_prover::KV_LEN / 2) as u32, 0),
        &[&k, &rk],
        qwen2_prover::KV_LEN,
        (div_ceil(qwen2_prover::KV_LEN / 2, 256), 1, 1),
    );
    let score = g.run(
        qwen2_prover::SCORE_QK,
        &cfg5(
            qwen2_prover::SEQ as u32,
            qwen2_prover::HEADS as u32,
            qwen2_prover::KV_HEADS as u32,
            qwen2_prover::HEAD_DIM as u32,
            -32768,
        ),
        &[&q, &k],
        qwen2_prover::SCORE_LEN,
        (
            div_ceil(qwen2_prover::SEQ, 8),
            div_ceil(qwen2_prover::SEQ, 8),
            qwen2_prover::HEADS as u32,
        ),
    );
    let prob = g.run(
        qwen2_prover::SOFTMAX,
        &cfg3(
            (qwen2_prover::HEADS * qwen2_prover::SEQ) as u32,
            qwen2_prover::SEQ as u32,
            ez,
        ),
        &[&score, &exp],
        qwen2_prover::SCORE_LEN,
        ((qwen2_prover::HEADS * qwen2_prover::SEQ) as u32, 1, 1),
    );
    let ctx = g.run(
        qwen2_prover::ATTN_V,
        &cfg5(
            qwen2_prover::SEQ as u32,
            qwen2_prover::HEADS as u32,
            qwen2_prover::KV_HEADS as u32,
            qwen2_prover::HEAD_DIM as u32,
            0,
        ),
        &[&prob, &v],
        qwen2_prover::X_LEN,
        (
            div_ceil(qwen2_prover::HEAD_DIM, 8),
            div_ceil(qwen2_prover::SEQ, 8),
            qwen2_prover::HEADS as u32,
        ),
    );
    let attn = g.run(
        qwen2_prover::MATMUL,
        &cfg3(
            qwen2_prover::SEQ as u32,
            qwen2_prover::HIDDEN as u32,
            qwen2_prover::HIDDEN as i32,
        ),
        &[&ctx, &w[qwen2_prover::weights::WO]],
        qwen2_prover::X_LEN,
        (
            div_ceil(qwen2_prover::HIDDEN, 16),
            div_ceil(qwen2_prover::SEQ, 16),
            1,
        ),
    );
    let h = g.run(
        qwen2_prover::ADD,
        &cfg1(qwen2_prover::X_LEN as u32),
        &[&x, &attn],
        qwen2_prover::X_LEN,
        (div_ceil(qwen2_prover::X_LEN, 256), 1, 1),
    );
    let norm2 = g.run(
        qwen2_prover::RMS_NORM,
        &cfg3(qwen2_prover::SEQ as u32, qwen2_prover::HIDDEN as u32, rz),
        &[&h, &w[qwen2_prover::weights::LN2], &rsqrt],
        qwen2_prover::X_LEN,
        (qwen2_prover::SEQ as u32, 1, 1),
    );
    let gate = g.run(
        qwen2_prover::MATMUL,
        &cfg3(
            qwen2_prover::SEQ as u32,
            qwen2_prover::INTERMEDIATE as u32,
            qwen2_prover::HIDDEN as i32,
        ),
        &[&norm2, &w[qwen2_prover::weights::WG]],
        qwen2_prover::MLP_LEN,
        (
            div_ceil(qwen2_prover::INTERMEDIATE, 16),
            div_ceil(qwen2_prover::SEQ, 16),
            1,
        ),
    );
    let up = g.run(
        qwen2_prover::MATMUL,
        &cfg3(
            qwen2_prover::SEQ as u32,
            qwen2_prover::INTERMEDIATE as u32,
            qwen2_prover::HIDDEN as i32,
        ),
        &[&norm2, &w[qwen2_prover::weights::WU]],
        qwen2_prover::MLP_LEN,
        (
            div_ceil(qwen2_prover::INTERMEDIATE, 16),
            div_ceil(qwen2_prover::SEQ, 16),
            1,
        ),
    );
    let mid = g.run(
        qwen2_prover::SILU_MUL,
        &cfg2(qwen2_prover::MLP_LEN as u32, sz),
        &[&gate, &up, &silu],
        qwen2_prover::MLP_LEN,
        (div_ceil(qwen2_prover::MLP_LEN, 256), 1, 1),
    );
    let mlp = g.run(
        qwen2_prover::MATMUL,
        &cfg3(
            qwen2_prover::SEQ as u32,
            qwen2_prover::HIDDEN as u32,
            qwen2_prover::INTERMEDIATE as i32,
        ),
        &[&mid, &w[qwen2_prover::weights::WD]],
        qwen2_prover::X_LEN,
        (
            div_ceil(qwen2_prover::HIDDEN, 16),
            div_ceil(qwen2_prover::SEQ, 16),
            1,
        ),
    );
    let y = g.run(
        qwen2_prover::ADD,
        &cfg1(qwen2_prover::X_LEN as u32),
        &[&h, &mlp],
        qwen2_prover::X_LEN,
        (div_ceil(qwen2_prover::X_LEN, 256), 1, 1),
    );
    let y = g.read(&y, qwen2_prover::X_LEN);
    assert_eq!(y.len(), qwen2_prover::X_LEN);
    assert!(y.iter().any(|&v| v != 0));
}

#[test]
#[ignore = "runs text embedding through all 24 Qwen2-0.5B layers and CPU tied lm head"]
fn full_text_next_token_smoke() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let path = root.join("atlas-onnx-tracer/models/qwen/model.safetensors");
    let tok = root.join("atlas-onnx-tracer/models/qwen/tokenizer.json");
    let bytes = std::fs::read(&path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let g = pollster::block_on(G::new());

    let ids = qwen2_prover::text::tokenize(&tok, "hello world this is a test").unwrap();
    let x = qwen2_prover::text::embed_from_safetensors(&st, &ids).unwrap();
    let mut x = g.buf("x", &x);

    let (rsqrt, rz) = qwen2_prover::lut::rsqrt(0, 1_000_000);
    let (silu, sz) = qwen2_prover::lut::silu(-32768, 32767);
    let (exp, ez) = qwen2_prover::lut::exp(-32768, 0);
    let rsqrt = g.buf("rsqrt", &rsqrt);
    let silu = g.buf("silu", &silu);
    let exp = g.buf("exp", &exp);
    let rq = g.buf(
        "rot_q",
        &rot(
            qwen2_prover::SEQ,
            qwen2_prover::HEADS,
            qwen2_prover::HEAD_DIM,
        ),
    );
    let rk = g.buf(
        "rot_k",
        &rot(
            qwen2_prover::SEQ,
            qwen2_prover::KV_HEADS,
            qwen2_prover::HEAD_DIM,
        ),
    );

    for layer in 0..qwen2_prover::LAYERS {
        let ws = qwen2_prover::weights::load_layer_from_safetensors(&st, layer).unwrap();
        let w: Vec<_> = ws
            .iter()
            .enumerate()
            .map(|(i, x)| g.buf(&format!("l{layer}_w{i}"), x))
            .collect();
        x = run_layer(&g, &x, &w, &rsqrt, &silu, &exp, &rq, &rk, rz, sz, ez);
    }

    let norm = qwen2_prover::weights::final_norm(&st).unwrap();
    let norm = g.buf("final_norm", &norm);
    let y = g.run(
        qwen2_prover::RMS_NORM,
        &cfg3(qwen2_prover::SEQ as u32, qwen2_prover::HIDDEN as u32, rz),
        &[&x, &norm, &rsqrt],
        qwen2_prover::X_LEN,
        (qwen2_prover::SEQ as u32, 1, 1),
    );
    let y = g.read(&y, qwen2_prover::X_LEN);
    let last = &y
        [(qwen2_prover::SEQ - 1) * qwen2_prover::HIDDEN..qwen2_prover::SEQ * qwen2_prover::HIDDEN];
    let id = qwen2_prover::text::argmax_tied_lm_head_from_safetensors(&st, last).unwrap();
    assert!((id as usize) < qwen2_prover::VOCAB);
    let ppl = qwen2_prover::text::perplexity_tied_lm_head_prefix_from_safetensors(&st, &y, &ids, 3)
        .unwrap();
    assert!(ppl.is_finite());
    let decoded = qwen2_prover::text::decode(&tok, id).unwrap();
    println!("next token: {id} {decoded:?}");
    println!("ppl(first 3 targets): {ppl}");
}
