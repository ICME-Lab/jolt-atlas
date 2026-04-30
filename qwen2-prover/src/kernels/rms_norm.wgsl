struct Cfg {
    rows: u32,
    cols: u32,
    lut_zero: i32,
}

@group(0) @binding(0)
var<uniform> cfg: Cfg;

@group(0) @binding(1)
var<storage, read> x: array<i32>;

@group(0) @binding(2)
var<storage, read> w: array<i32>;

@group(0) @binding(3)
var<storage, read> rsqrt_lut: array<i32>;

@group(0) @binding(4)
var<storage, read_write> y: array<i32>;

fn qmul(a: i32, b: i32) -> i32 {
    return (a * b) >> 8u;
}

fn lut(x: i32, zero: i32) -> i32 {
    let n = i32(arrayLength(&rsqrt_lut));
    let i = min(max(x + zero, 0i), n - 1i);
    return rsqrt_lut[u32(i)];
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let r = id.x;
    if (r >= cfg.rows) {
        return;
    }

    var ss = 0i;
    for (var c = 0u; c < cfg.cols; c = c + 1u) {
        let v = x[r * cfg.cols + c];
        ss = ss + ((v * v) >> 8u);
    }
    let mean = ss / i32(cfg.cols);
    let inv = lut(mean, cfg.lut_zero);

    for (var c = 0u; c < cfg.cols; c = c + 1u) {
        let i = r * cfg.cols + c;
        y[i] = qmul(qmul(x[i], inv), w[c]);
    }
}
