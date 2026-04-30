struct Cfg {
    n: u32,
}

@group(0) @binding(0)
var<uniform> cfg: Cfg;

@group(0) @binding(1)
var<storage, read> a: array<i32>;

@group(0) @binding(2)
var<storage, read> b: array<i32>;

@group(0) @binding(3)
var<storage, read_write> y: array<i32>;

fn qmul(a: i32, b: i32) -> i32 {
    return (a * b) >> 8u;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= cfg.n) {
        return;
    }
    y[i] = qmul(a[i], b[i]);
}
