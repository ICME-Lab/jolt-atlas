use jolt_core::{
    field::JoltField,
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
};

pub struct ProgramIOPolynomial<F: JoltField> {
    poly: MultilinearPolynomial<F>,
}

impl<F: JoltField> ProgramIOPolynomial<F> {
    pub fn new(output_vals: &[F]) -> Self {
        Self {
            poly: output_vals.to_vec().into(),
        }
    }

    pub fn evaluate(&self, r_address: &[F]) -> F {
        let (r_hi, r_lo) = r_address.split_at(r_address.len() - self.poly.get_num_vars());
        debug_assert_eq!(r_lo.len(), self.poly.get_num_vars());

        let mut result = self.poly.evaluate(r_lo);
        for r_i in r_hi.iter() {
            result *= F::one() - r_i;
        }

        result
    }
}
