struct Cfg {
    seq: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    mask: i32,
}

@group(0) @binding(0)
var<uniform> cfg: Cfg;

@group(0) @binding(1)
var<storage, read> q: array<i32>;

@group(0) @binding(2)
var<storage, read> k: array<i32>;

@group(0) @binding(3)
var<storage, read_write> y: array<i32>;

fn q_idx(pos: u32, head: u32, dim: u32) -> u32 {
    return (pos * cfg.heads + head) * cfg.head_dim + dim;
}

fn k_idx(pos: u32, kv_head: u32, dim: u32) -> u32 {
    return (pos * cfg.kv_heads + kv_head) * cfg.head_dim + dim;
}

fn y_idx(head: u32, q_pos: u32, k_pos: u32) -> u32 {
    return (head * cfg.seq + q_pos) * cfg.seq + k_pos;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let k_pos = id.x;
    let q_pos = id.y;
    let head = id.z;
    if (k_pos >= cfg.seq || q_pos >= cfg.seq || head >= cfg.heads) {
        return;
    }

    if (k_pos > q_pos) {
        y[y_idx(head, q_pos, k_pos)] = cfg.mask;
        return;
    }

    let group = cfg.heads / cfg.kv_heads;
    let kv_head = head / group;
    var acc = 0i;
    for (var d = 0u; d < cfg.head_dim; d = d + 1u) {
        acc = acc + ((q[q_idx(q_pos, head, d)] * k[k_idx(k_pos, kv_head, d)]) >> 8u);
    }
    y[y_idx(head, q_pos, k_pos)] = acc;
}
