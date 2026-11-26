use crate::{field::F, fold::basis, tensor::Shape};

#[derive(Clone, Copy)]
pub(crate) struct Tile {
    pub row: usize,
    pub col: usize,
    pub size: usize,
}

impl Tile {
    pub fn for_attention(rows: usize, causal: bool) -> Vec<Self> {
        fn visit(rows: usize, causal: bool, tile: Tile, output: &mut Vec<Tile>) {
            let Tile { row, col, size } = tile;
            if row >= rows || col >= rows || causal && col >= row + size {
                return;
            }
            if size <= 64 || size <= 512 && (!causal || col + size - 1 <= row) {
                output.push(tile);
            } else {
                let half = size / 2;
                for row in [row, row + half] {
                    for col in [col, col + half] {
                        visit(
                            rows,
                            causal,
                            Tile {
                                row,
                                col,
                                size: half,
                            },
                            output,
                        );
                    }
                }
            }
        }
        let mut output = Vec::new();
        visit(
            rows,
            causal,
            Tile {
                row: 0,
                col: 0,
                size: rows.next_power_of_two(),
            },
            &mut output,
        );
        output
    }

    pub fn shape(self) -> Shape {
        Shape {
            rows: self.size,
            cols: self.size,
        }
    }

    pub fn restrict(self, rows: &[F], cols: &[F]) -> (F, Vec<F>) {
        let width = self.size.ilog2() as usize;
        let (rp, r) = rows.split_at(rows.len() - width);
        let (cp, c) = cols.split_at(cols.len() - width);
        (
            basis(rp)[self.row / self.size] * basis(cp)[self.col / self.size],
            [r, c].concat(),
        )
    }

    pub fn at(self, i: usize) -> (usize, usize) {
        (self.row + i / self.size, self.col + i % self.size)
    }
}
