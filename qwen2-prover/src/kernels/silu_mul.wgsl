struct Cfg {
    n: u32,
    lut_zero: i32,
}

@group(0) @binding(0)
var<uniform> cfg: Cfg;

@group(0) @binding(1)
var<storage, read> gate: array<i32>;

@group(0) @binding(2)
var<storage, read> up: array<i32>;

@group(0) @binding(3)
var<storage, read> silu_lut: array<i32>;

@group(0) @binding(4)
var<storage, read_write> y: array<i32>;

fn qmul(a: i32, b: i32) -> i32 {
    return (a * b) >> 8u;
}

fn lut(x: i32, zero: i32) -> i32 {
    let n = i32(arrayLength(&silu_lut));
    let i = min(max(x + zero, 0i), n - 1i);
    return silu_lut[u32(i)];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= cfg.n) {
        return;
    }
    y[i] = qmul(lut(gate[i], cfg.lut_zero), up[i]);
}
