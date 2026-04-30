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
var<storage, read> exp_lut: array<i32>;

@group(0) @binding(3)
var<storage, read_write> y: array<i32>;

fn lut(x: i32, zero: i32) -> i32 {
    let n = i32(arrayLength(&exp_lut));
    let i = min(max(x + zero, 0i), n - 1i);
    return exp_lut[u32(i)];
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let r = id.x;
    if (r >= cfg.rows) {
        return;
    }

    var mx = x[r * cfg.cols];
    for (var c = 1u; c < cfg.cols; c = c + 1u) {
        mx = max(mx, x[r * cfg.cols + c]);
    }

    var sum = 0i;
    for (var c = 0u; c < cfg.cols; c = c + 1u) {
        let e = lut(x[r * cfg.cols + c] - mx, cfg.lut_zero);
        y[r * cfg.cols + c] = e;
        sum = sum + e;
    }

    for (var c = 0u; c < cfg.cols; c = c + 1u) {
        let i = r * cfg.cols + c;
        y[i] = (y[i] << 8u) / sum;
    }
}
