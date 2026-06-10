#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixShape {
    pub rows: usize,
    pub cols: usize,
}

impl MatrixShape {
    pub fn new(rows: usize, cols: usize) -> Option<Self> {
        (rows.is_power_of_two() && cols.is_power_of_two()).then_some(Self { rows, cols })
    }

    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    pub fn row_vars(&self) -> usize {
        self.rows.ilog2() as usize
    }

    pub fn col_vars(&self) -> usize {
        self.cols.ilog2() as usize
    }

    pub fn point_len(&self) -> usize {
        self.row_vars() + self.col_vars()
    }
}
