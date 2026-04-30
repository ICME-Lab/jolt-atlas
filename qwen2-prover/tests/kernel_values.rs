use std::sync::mpsc;

fn u32s(xs: &[u32]) -> Vec<u8> {
    let mut bs = Vec::with_capacity(16);
    for x in xs {
        bs.extend(x.to_ne_bytes());
    }
    bs.resize(16, 0);
    bs
}

fn cfg2(a: u32, b: i32) -> Vec<u8> {
    let mut bs = Vec::with_capacity(16);
    bs.extend(a.to_ne_bytes());
    bs.extend(b.to_ne_bytes());
    bs.resize(16, 0);
    bs
}

fn cfg3(a: u32, b: u32, c: i32) -> Vec<u8> {
    let mut bs = Vec::with_capacity(16);
    bs.extend(a.to_ne_bytes());
    bs.extend(b.to_ne_bytes());
    bs.extend(c.to_ne_bytes());
    bs.resize(16, 0);
    bs
}

fn cfg5(a: u32, b: u32, c: u32, d: u32, e: i32) -> Vec<u8> {
    let mut bs = Vec::with_capacity(32);
    bs.extend(a.to_ne_bytes());
    bs.extend(b.to_ne_bytes());
    bs.extend(c.to_ne_bytes());
    bs.extend(d.to_ne_bytes());
    bs.extend(e.to_ne_bytes());
    bs.resize(32, 0);
    bs
}

fn i32s(xs: &[i32]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

fn qmul(a: i32, b: i32) -> i32 {
    (a * b) >> 8
}

fn at(xs: &[i32], z: i32, x: i32) -> i32 {
    xs[(x + z) as usize]
}

fn run(src: &str, cfg: &[u8], ins: &[&[i32]], out_len: usize, wg: (u32, u32, u32)) -> Vec<i32> {
    pollster::block_on(async {
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
                label: Some("qwen2-prover-values-test"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("failed to request wgpu device");

        let cfg_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cfg"),
            size: cfg.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&cfg_buf, 0, cfg);

        let mut in_bufs = Vec::new();
        for xs in ins {
            let buf = dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("in"),
                size: (xs.len() * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, &i32s(xs));
            in_bufs.push(buf);
        }

        let out = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: (out_len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read"),
            size: (out_len * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
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
        let layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layout"),
            entries: &layout_entries,
        });

        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: cfg_buf.as_entire_binding(),
        }];
        for (i, buf) in in_bufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (i + 1) as u32,
                resource: buf.as_entire_binding(),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: (ins.len() + 1) as u32,
            resource: out.as_entire_binding(),
        });
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind"),
            layout: &layout,
            entries: &entries,
        });

        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipe = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipe"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(wg.0, wg.1, wg.2);
        }
        enc.copy_buffer_to_buffer(&out, 0, &read, 0, (out_len * 4) as u64);
        queue.submit([enc.finish()]);

        let slice = read.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();

        let view = slice.get_mapped_range();
        let got = view
            .chunks_exact(4)
            .map(|b| i32::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        drop(view);
        read.unmap();
        got
    })
}

#[test]
fn add_values() {
    let a = [1, -2, 3, 4];
    let b = [5, 6, -7, 8];
    let got = run(qwen2_prover::ADD, &u32s(&[4]), &[&a, &b], 4, (1, 1, 1));
    assert_eq!(got, vec![6, 4, -4, 12]);
}

#[test]
fn mul_values_q8_8() {
    let a = [384, -512, 128];
    let b = [512, 384, -256];
    let got = run(qwen2_prover::MUL, &u32s(&[3]), &[&a, &b], 3, (1, 1, 1));
    assert_eq!(
        got,
        a.iter()
            .zip(b)
            .map(|(&x, y)| qmul(x, y))
            .collect::<Vec<_>>()
    );
}

#[test]
fn matmul_values_q8_8() {
    let a = [256, 512, -256, 128, 256, 384];
    let b = [256, -256, 512, 128, 256, 256];
    let got = run(
        qwen2_prover::MATMUL,
        &u32s(&[2, 2, 3]),
        &[&a, &b],
        4,
        (1, 1, 1),
    );
    let mut exp = vec![0; 4];
    for r in 0..2 {
        for c in 0..2 {
            let mut acc = 0;
            for p in 0..3 {
                acc += qmul(a[r * 3 + p], b[p * 2 + c]);
            }
            exp[r * 2 + c] = acc;
        }
    }
    assert_eq!(got, exp);
}

#[test]
fn rope_values_q8_8() {
    let x = [256, 512, -256, 128];
    let rot = [256, 0, 256, 0];
    let got = run(qwen2_prover::ROPE, &u32s(&[2]), &[&x, &rot], 4, (1, 1, 1));
    assert_eq!(got, x);
}

#[test]
fn silu_mul_uses_lut() {
    let gate = [-1, 0, 1];
    let up = [256, 512, -256];
    let (lut, z) = qwen2_prover::lut::silu(-1, 1);
    let got = run(
        qwen2_prover::SILU_MUL,
        &cfg2(3, z),
        &[&gate, &up, &lut],
        3,
        (1, 1, 1),
    );
    assert_eq!(
        got,
        vec![
            qmul(at(&lut, z, -1), 256),
            qmul(at(&lut, z, 0), 512),
            qmul(at(&lut, z, 1), -256)
        ]
    );
}

#[test]
fn rms_norm_uses_lut() {
    let x = [16, 0, 16, 16];
    let w = [256, 512];
    let (lut, z) = qwen2_prover::lut::rsqrt(0, 256);
    let got = run(
        qwen2_prover::RMS_NORM,
        &cfg3(2, 2, z),
        &[&x, &w, &lut],
        4,
        (2, 1, 1),
    );

    let mean0 = (qmul(16, 16) + qmul(0, 0)) / 2;
    let mean1 = (qmul(16, 16) + qmul(16, 16)) / 2;
    let inv0 = at(&lut, z, mean0);
    let inv1 = at(&lut, z, mean1);
    assert_eq!(
        got,
        vec![
            qmul(qmul(16, inv0), 256),
            0,
            qmul(qmul(16, inv1), 256),
            qmul(qmul(16, inv1), 512)
        ]
    );
}

#[test]
fn softmax_uses_exp_lut() {
    let x = [0, -1, 1, 1];
    let (lut, z) = qwen2_prover::lut::exp(-2, 0);
    let got = run(
        qwen2_prover::SOFTMAX,
        &cfg3(2, 2, z),
        &[&x, &lut],
        4,
        (2, 1, 1),
    );
    let e0 = at(&lut, z, -1);
    let e1 = at(&lut, z, -2);
    let s0 = e0 + e1;
    let e2 = at(&lut, z, 0);
    let s1 = e2 + e2;
    assert_eq!(
        got,
        vec![
            (e0 << 8) / s0,
            (e1 << 8) / s0,
            (e2 << 8) / s1,
            (e2 << 8) / s1
        ]
    );
}

#[test]
fn score_qk_values_q8_8_with_gqa_and_mask() {
    let seq = 3usize;
    let heads = 2usize;
    let kv_heads = 1usize;
    let head_dim = 2usize;
    let mask = -12345;
    let q = [
        256, 0, 0, 256, //
        512, 0, 0, 512, //
        256, 256, 512, 512,
    ];
    let k = [
        256, 0, //
        0, 256, //
        256, 256,
    ];
    let got = run(
        qwen2_prover::SCORE_QK,
        &cfg5(
            seq as u32,
            heads as u32,
            kv_heads as u32,
            head_dim as u32,
            mask,
        ),
        &[&q, &k],
        heads * seq * seq,
        (1, 1, heads as u32),
    );

    let mut exp = vec![0; heads * seq * seq];
    for h in 0..heads {
        for qp in 0..seq {
            for kp in 0..seq {
                let yi = (h * seq + qp) * seq + kp;
                if kp > qp {
                    exp[yi] = mask;
                    continue;
                }
                let mut acc = 0;
                for d in 0..head_dim {
                    let qi = (qp * heads + h) * head_dim + d;
                    let ki = kp * head_dim + d;
                    acc += qmul(q[qi], k[ki]);
                }
                exp[yi] = acc;
            }
        }
    }
    assert_eq!(got, exp);
}

#[test]
fn attn_v_values_q8_8_with_gqa() {
    let seq = 2usize;
    let heads = 2usize;
    let kv_heads = 1usize;
    let head_dim = 2usize;
    let p = [
        256, 0, //
        128, 128, //
        0, 256, //
        256, 0,
    ];
    let v = [
        256, 512, //
        768, 1024,
    ];
    let got = run(
        qwen2_prover::ATTN_V,
        &cfg5(
            seq as u32,
            heads as u32,
            kv_heads as u32,
            head_dim as u32,
            0,
        ),
        &[&p, &v],
        seq * heads * head_dim,
        (1, 1, heads as u32),
    );

    let mut exp = vec![0; seq * heads * head_dim];
    for pos in 0..seq {
        for h in 0..heads {
            for d in 0..head_dim {
                let mut acc = 0;
                for kp in 0..seq {
                    let pi = (h * seq + pos) * seq + kp;
                    let vi = kp * head_dim + d;
                    acc += qmul(p[pi], v[vi]);
                }
                exp[(pos * heads + h) * head_dim + d] = acc;
            }
        }
    }
    assert_eq!(got, exp);
}
