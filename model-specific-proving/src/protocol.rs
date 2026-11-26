use serde::{Deserialize, Serialize};
use std::ops::Range;

pub use crate::channel::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candidate {
    pub id: usize,
    pub logit: i32,
}

#[derive(Clone)]
pub struct TopKConfig {
    pub positions: Range<usize>,
    pub k: usize,
    pub rows_per_block: usize,
    pub vocab_per_block: usize,
}

pub struct Proof {
    pub(crate) header: Header,
    pub(crate) evaluations: Vec<crate::bitz::Proof>,
}

impl Proof {
    pub(crate) fn evaluation(
        &self,
        index: usize,
    ) -> Result<std::borrow::Cow<'_, crate::bitz::Proof>> {
        self.evaluations
            .get(index)
            .map(std::borrow::Cow::Borrowed)
            .ok_or("missing BitZ evaluation")
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Header {
    pub(crate) widths: Vec<u8>,
    pub(crate) state: crate::channel::Messages,
    #[serde(with = "serde_bytes")]
    pub(crate) transcript: Vec<u8>,
    pub(crate) roots: Vec<[u8; 32]>,
}
