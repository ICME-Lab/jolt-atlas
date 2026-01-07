use allocative::Allocative;

pub mod consts;

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum CommittedPolynomial {
    /// The index (i.e. the 0 field) refers to this node's one-hot-encoded output MLE.
    NodeOutputRa(usize),
    DivNodeQuotient(usize),
    DivNodeRemainder(usize),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum VirtualPolynomial {
    /// The index (i.e. the 0 field) refers to this node's output MLE.
    NodeOutput(usize),
    /// Fields:
    /// 0 - node index
    /// 1 - d
    NodeOutputRaD(usize, usize),
}
