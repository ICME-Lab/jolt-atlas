#[test]
fn kernels_compile() {
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

        let (dev, _) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("qwen2-prover-kernels-test"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("failed to request wgpu device");

        for (name, src) in [
            ("add", qwen2_prover::ADD),
            ("copy", qwen2_prover::COPY),
            ("matmul", qwen2_prover::MATMUL),
            ("mul", qwen2_prover::MUL),
            ("rms_norm", qwen2_prover::RMS_NORM),
            ("rope", qwen2_prover::ROPE),
            ("silu_mul", qwen2_prover::SILU_MUL),
            ("softmax", qwen2_prover::SOFTMAX),
            ("score_qk", qwen2_prover::SCORE_QK),
            ("attn_v", qwen2_prover::ATTN_V),
        ] {
            let _ = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
        }
    });
}
