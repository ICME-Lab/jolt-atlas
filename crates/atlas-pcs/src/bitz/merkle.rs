//! A binary Merkle tree over the rows of a matrix, with BLAKE3.
//!
//! A leaf is the hash of one row's symbols; leaf and node hashes are domain-separated so a row
//! cannot be passed off as an inner node.

use super::gf128::Gf128;
use rayon::prelude::*;

pub type Digest = [u8; 32];

const LEAF_TAG: u8 = 0;
const NODE_TAG: u8 = 1;

fn hash_row(row: &[Gf128]) -> Digest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&[LEAF_TAG]);
    for symbol in row {
        hasher.update(&symbol.0.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn hash_node(left: &Digest, right: &Digest) -> Digest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&[NODE_TAG]);
    hasher.update(left);
    hasher.update(right);
    *hasher.finalize().as_bytes()
}

/// The tree over `2^depth` rows: `layers[0]` holds the leaf hashes and the last layer the root.
#[derive(Clone, Debug)]
pub struct Tree {
    layers: Vec<Vec<Digest>>,
}

impl Tree {
    /// Commits to the rows of a row-major matrix of `width` symbols per row. The number of
    /// rows is a power of two.
    pub fn commit(matrix: &[Gf128], width: usize) -> Self {
        assert!(
            width > 0 && matrix.len().is_multiple_of(width),
            "matrix shape"
        );
        let rows = matrix.len() / width;
        assert!(
            rows.is_power_of_two(),
            "the tree is over a power of two rows"
        );
        let mut layers = vec![matrix.par_chunks(width).map(hash_row).collect::<Vec<_>>()];
        while layers[layers.len() - 1].len() > 1 {
            let below = &layers[layers.len() - 1];
            let above = below
                .par_chunks(2)
                .map(|pair| hash_node(&pair[0], &pair[1]))
                .collect();
            layers.push(above);
        }
        Self { layers }
    }

    pub fn root(&self) -> Digest {
        self.layers[self.layers.len() - 1][0]
    }

    /// The sibling hashes from row `index` up to the root.
    pub fn open(&self, index: usize) -> Vec<Digest> {
        let mut position = index;
        let mut path = Vec::with_capacity(self.layers.len() - 1);
        for layer in &self.layers[..self.layers.len() - 1] {
            path.push(layer[position ^ 1]);
            position >>= 1;
        }
        path
    }
}

/// Whether `row` is row `index` of the matrix committed to by `root`, a tree of `2^depth` rows.
pub fn verify_row(
    root: &Digest,
    depth: usize,
    index: usize,
    row: &[Gf128],
    path: &[Digest],
) -> bool {
    if path.len() != depth || (depth < usize::BITS as usize && index >> depth != 0) {
        return false;
    }
    let mut position = index;
    let mut digest = hash_row(row);
    for sibling in path {
        digest = if position & 1 == 0 {
            hash_node(&digest, sibling)
        } else {
            hash_node(sibling, &digest)
        };
        position >>= 1;
    }
    digest == *root
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: usize, width: usize) -> Vec<Gf128> {
        (0..rows * width)
            .map(|i| Gf128(i as u128 * 0x9e37_79b9 + 1))
            .collect()
    }

    #[test]
    fn every_row_opens_and_nothing_else_does() {
        for (rows, width) in [(1usize, 3usize), (2, 1), (16, 4)] {
            let m = matrix(rows, width);
            let tree = Tree::commit(&m, width);
            let (root, depth) = (tree.root(), rows.trailing_zeros() as usize);
            for index in 0..rows {
                let row = &m[index * width..(index + 1) * width];
                let path = tree.open(index);
                assert!(verify_row(&root, depth, index, row, &path));
                let mut wrong = row.to_vec();
                wrong[0] += Gf128::ONE;
                assert!(!verify_row(&root, depth, index, &wrong, &path));
                assert!(!verify_row(&root, depth, index + rows, row, &path));
                if rows > 1 {
                    assert!(!verify_row(&root, depth, index ^ 1, row, &path));
                    let mut bad_path = path.clone();
                    bad_path[0][0] ^= 1;
                    assert!(!verify_row(&root, depth, index, row, &bad_path));
                    assert!(!verify_row(&root, depth, index, row, &path[1..]));
                }
            }
        }
    }

    #[test]
    fn a_row_is_not_an_inner_node() {
        // Two leaf hashes, presented as a one-row matrix of their bytes, do not reproduce the
        // root of the two-row tree.
        let m = matrix(2, 2);
        let tree = Tree::commit(&m, 2);
        let as_symbols: Vec<Gf128> = tree.layers[0]
            .iter()
            .flat_map(|d| {
                [
                    Gf128(u128::from_le_bytes(d[..16].try_into().unwrap())),
                    Gf128(u128::from_le_bytes(d[16..].try_into().unwrap())),
                ]
            })
            .collect();
        assert_ne!(Tree::commit(&as_symbols, 4).root(), tree.root());
    }
}
