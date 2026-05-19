#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProveResult<C, P> {
    pub claims: C,
    pub proof: P,
}

impl<C, P> ProveResult<C, P> {
    pub fn new(claims: C, proof: P) -> Self {
        Self { claims, proof }
    }
}
