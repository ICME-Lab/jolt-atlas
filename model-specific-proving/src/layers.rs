#[derive(Clone, Copy)]
pub struct Projection<'a> {
    pub weights: &'a [i32],
    pub bias: &'a [i32],
}

#[derive(Clone, Copy)]
pub struct Rotary<'a> {
    pub cos: &'a [i32],
    pub sin: &'a [i32],
}

#[derive(Clone, Copy)]
pub struct Activation<'a> {
    pub lower: i32,
    pub correction: &'a [i32],
}

#[derive(Clone, Copy)]
pub struct Mlp<'a> {
    pub up: Projection<'a>,
    pub gate: Option<Projection<'a>>,
    pub down: Projection<'a>,
    pub activation: Activation<'a>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NormalizationKind {
    Layer,
    Rms,
}

#[derive(Clone, Copy)]
pub struct NormalizationParameters<'a> {
    pub kind: NormalizationKind,
    pub gain: &'a [i32],
    pub bias: &'a [i32],
    pub epsilon_variance_units: i64,
}

#[derive(Clone, Copy)]
pub struct SoftmaxTables<'a> {
    pub probabilities: &'a [i32],
    pub sums: &'a [i32],
    pub limit: i32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SoftmaxMask {
    Full,
    Causal,
}

pub struct Attention<'a> {
    pub queries: Vec<Projection<'a>>,
    pub groups: Vec<[Projection<'a>; 2]>,
    pub output: Projection<'a>,
    pub rotary: Option<Rotary<'a>>,
    pub mask: SoftmaxMask,
    pub tables: SoftmaxTables<'a>,
    pub factor: i32,
}

pub(crate) mod attention;
pub(crate) mod linear;
pub(crate) mod mlp;
pub(crate) mod norm;
pub(crate) mod softmax;
