use ark_bn254::Fq;
use ark_ff::{Field, One, UniformRand, Zero};
use ark_std::test_rng;
use jolt_optimizations::{
    eval_multilinear,
    fq12_poly::{eval_poly_vec, to_multilinear_evals},
};

/// Generate random polynomial coefficients for testing
fn random_poly12_coeffs() -> [Fq; 12] {
    let mut rng = test_rng();
    let mut coeffs = [Fq::zero(); 12];
    for c in coeffs.iter_mut() {
        *c = Fq::rand(&mut rng);
    }
    coeffs
}

#[test]
fn test_mle_agreement_with_univariate() {
    let coeffs = random_poly12_coeffs();
    let mle_evals = to_multilinear_evals(&coeffs);

    for i in 0..16 {
        let x = Fq::from(i as u64);
        let univariate_eval = eval_poly_vec(&coeffs[..], &x);

        assert_eq!(
            univariate_eval, mle_evals[i],
            "MLE evaluation doesn't match univariate at point {}",
            i
        );
        let binary_point = vec![
            Fq::from((i & 1) as u64),
            Fq::from(((i >> 1) & 1) as u64),
            Fq::from(((i >> 2) & 1) as u64),
            Fq::from(((i >> 3) & 1) as u64),
        ];
        let mle_eval = eval_multilinear(&mle_evals, &binary_point);

        assert_eq!(
            univariate_eval, mle_eval,
            "eval_multilinear doesn't agree with univariate at point {}",
            i
        );
    }
}

#[test]
fn test_mle_is_multilinear() {
    let mut rng = test_rng();
    let coeffs = random_poly12_coeffs();
    let mle_evals = to_multilinear_evals(&coeffs);

    for var_idx in 0..4 {
        let point = vec![
            Fq::rand(&mut rng),
            Fq::rand(&mut rng),
            Fq::rand(&mut rng),
            Fq::rand(&mut rng),
        ];

        let mut p0 = point.clone();
        p0[var_idx] = Fq::zero();
        let eval0 = eval_multilinear(&mle_evals, &p0);

        let mut p1 = point.clone();
        p1[var_idx] = Fq::one();
        let eval1 = eval_multilinear(&mle_evals, &p1);

        let t = Fq::rand(&mut rng);
        let mut pt = point.clone();
        pt[var_idx] = t;
        let eval_t = eval_multilinear(&mle_evals, &pt);
        let expected = eval0 * (Fq::one() - t) + eval1 * t;

        assert_eq!(
            eval_t, expected,
            "MLE is not linear in variable {}",
            var_idx
        );
    }
}

#[test]
fn test_mle_special_cases() {
    let zero_coeffs = [Fq::zero(); 12];
    let mle = to_multilinear_evals(&zero_coeffs);
    assert!(
        mle.iter().all(|&x| x.is_zero()),
        "Zero polynomial MLE should be all zeros"
    );

    let const_val = Fq::from(42u64);
    let mut const_coeffs = [Fq::zero(); 12];
    const_coeffs[0] = const_val;
    let mle = to_multilinear_evals(&const_coeffs);
    assert!(
        mle.iter().all(|&x| x == const_val),
        "Constant polynomial MLE should be constant"
    );

    let mut linear_coeffs = [Fq::zero(); 12];
    linear_coeffs[1] = Fq::one();
    let mle = to_multilinear_evals(&linear_coeffs);
    for i in 0..16 {
        assert_eq!(
            mle[i],
            Fq::from(i as u64),
            "Linear polynomial p(x)=x should evaluate to {} at {}",
            i,
            i
        );
    }
    let mut quad_coeffs = [Fq::zero(); 12];
    quad_coeffs[2] = Fq::one();
    let mle = to_multilinear_evals(&quad_coeffs);
    for i in 0..16 {
        let expected = Fq::from((i * i) as u64);
        assert_eq!(
            mle[i], expected,
            "Quadratic polynomial p(x)=x² should evaluate correctly at {}",
            i
        );
    }
}

#[test]
fn test_mle_high_degree() {
    let mut coeffs = [Fq::zero(); 12];
    coeffs[11] = Fq::one();

    let mle = to_multilinear_evals(&coeffs);
    for i in 0..16 {
        let x = Fq::from(i as u64);
        let expected = x.pow([11u64]);
        assert_eq!(
            mle[i], expected,
            "p(x) = x^11 should evaluate correctly at {}",
            i
        );
    }
}
