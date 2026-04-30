struct Cfg {
    m: u32,
    n: u32,
    k: u32,
}

@group(0) @binding(0)
var<uniform> cfg: Cfg;

@group(0) @binding(1)
var<storage, read> a: array<i32>;

@group(0) @binding(2)
var<storage, read> b: array<i32>;

@group(0) @binding(3)
var<storage, read_write> y: array<i32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.y;
    let col = id.x;
    if (row >= cfg.m || col >= cfg.n) {
        return;
    }

    var acc = 0i;
    for (var p = 0u; p < cfg.k; p = p + 1u) {
        let av = a[row * cfg.k + p];
        let bv = b[p * cfg.n + col];
        acc = acc + ((av * bv) >> 8u);
    }
    y[row * cfg.n + col] = acc;
}
