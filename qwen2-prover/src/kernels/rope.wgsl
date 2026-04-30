struct Cfg {
    n_pair: u32,
}

@group(0) @binding(0)
var<uniform> cfg: Cfg;

@group(0) @binding(1)
var<storage, read> x: array<i32>;

@group(0) @binding(2)
var<storage, read> rot: array<i32>;

@group(0) @binding(3)
var<storage, read_write> y: array<i32>;

fn qmul(a: i32, b: i32) -> i32 {
    return (a * b) >> 8u;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = id.x;
    if (p >= cfg.n_pair) {
        return;
    }

    let i = p * 2u;
    let a = x[i];
    let b = x[i + 1u];
    let c = rot[i];
    let s = rot[i + 1u];
    y[i] = qmul(a, c) - qmul(b, s);
    y[i + 1u] = qmul(a, s) + qmul(b, c);
}
