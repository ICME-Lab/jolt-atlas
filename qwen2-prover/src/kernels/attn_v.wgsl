struct Cfg {
    seq: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
}

@group(0) @binding(0)
var<uniform> cfg: Cfg;

@group(0) @binding(1)
var<storage, read> p: array<i32>;

@group(0) @binding(2)
var<storage, read> v: array<i32>;

@group(0) @binding(3)
var<storage, read_write> y: array<i32>;

fn p_idx(head: u32, q_pos: u32, k_pos: u32) -> u32 {
    return (head * cfg.seq + q_pos) * cfg.seq + k_pos;
}

fn v_idx(pos: u32, kv_head: u32, dim: u32) -> u32 {
    return (pos * cfg.kv_heads + kv_head) * cfg.head_dim + dim;
}

fn y_idx(pos: u32, head: u32, dim: u32) -> u32 {
    return (pos * cfg.heads + head) * cfg.head_dim + dim;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = id.x;
    let pos = id.y;
    let head = id.z;
    if (dim >= cfg.head_dim || pos >= cfg.seq || head >= cfg.heads) {
        return;
    }

    let group = cfg.heads / cfg.kv_heads;
    let kv_head = head / group;
    var acc = 0i;
    for (var k_pos = 0u; k_pos < cfg.seq; k_pos = k_pos + 1u) {
        acc = acc + ((p[p_idx(head, pos, k_pos)] * v[v_idx(k_pos, kv_head, dim)]) >> 8u);
    }
    y[y_idx(pos, head, dim)] = acc;
}
