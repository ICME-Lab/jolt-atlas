use super::*;
use crypto_primitives::LiftElement;

#[test]
fn fp119_matches_world_bitz() {
    let mut state = 42u128;
    let mut draw = || {
        state = state
            .wrapping_mul(47026247687942121848144207491837523525)
            .wrapping_add(1);
        state ^ (state >> 61)
    };
    let edge = [
        0,
        1,
        68,
        69,
        1 << 55,
        1 << 64,
        MODULUS - 1,
        MODULUS,
        u128::MAX,
    ];
    let check = |x: u128, y: u128| {
        let (a, b) = (F::from_u128(x), F::from_u128(y));
        let (c, d) = (Scalar::from(x), Scalar::from(y));
        assert_eq!((a + b).0, (c + d).lift());
        assert_eq!((a - b).0, (c - d).lift());
        assert_eq!((a * b).0, (c * d).lift());
        let integer = i128::from(y as i64);
        assert_eq!(a.times(integer), a * F::from_i128(integer));
    };
    for x in edge {
        for y in edge {
            check(x, y);
        }
    }
    for _ in 0..2000 {
        check(draw(), draw());
    }
}

#[test]
fn parallel_work_selects_the_first_jolt_nonce() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(|| {
            let tag = protocol_tag(b"fp119 work parity");
            for bits in [14, 15, 17] {
                let mut reference = prover_transcript::<Keccak>(&tag, b"");
                let mut parallel = prover_transcript::<Keccak>(&tag, b"");
                reference.grind(bits).unwrap();
                parallel.work(bits).unwrap();
                assert_eq!(reference.narg_string(), parallel.narg_string());
                let expected: Vec<F> = reference.challenge_vector(4).unwrap();
                let actual: Vec<F> = parallel.challenge_vector(4).unwrap();
                assert_eq!(actual, expected);
            }
        });
}

#[test]
fn packed_digits_commit_like_materialized_digits() {
    let shape = Shape::new(8, 8).unwrap();
    let mut a = Advice::new(true).unwrap();
    let mut b = Advice::new(true).unwrap();
    let code = a.pack(shape, 6, |i| i as i128).unwrap();
    for shift in [0, 3] {
        a.digit(shape, code, shift, 3);
        b.column(shape, Width::Unsigned(8), |i| 1 << ((i >> shift) & 7))
            .unwrap();
    }
    a.seal().unwrap();
    b.seal().unwrap();
    let root = |advice: &Advice| {
        Group::new(advice, vec![0, 1])
            .unwrap()
            .commit(advice)
            .unwrap()
            .root
    };
    assert_eq!(root(&a), root(&b));
}

#[test]
fn signed_and_unsigned_openings_keep_their_offsets() {
    let shape = Shape::new(2, 2).unwrap();
    let x = [-5, 8, 4, -1];
    let y = [0, 7, 3, 2];
    let mut a = Advice::new(true).unwrap();
    let x_slot = a.column(shape, Width::Signed(5), |i| x[i].into()).unwrap();
    let y_slot = a
        .column(shape, Width::Unsigned(3), |i| y[i].into())
        .unwrap();
    a.seal().unwrap();
    let group = Group::new(&a, vec![0, 1]).unwrap();
    let root = group.commit(&a).unwrap().root;
    let queries = || {
        let mut requests = Vec::new();
        for (key, v, point) in [(x_slot, x, [3, 5]), (y_slot, y, [7, 11])] {
            let p = point.map(F::from_u64);
            ask(&mut requests, key, &p, shape.eval(&v, &p));
        }
        requests
    };
    let tag = protocol_tag(b"BitZ integer parity");
    let mut prover = prover_transcript::<Keccak>(&tag, &root);
    let proof = group
        .prove(&a, root, queries(), &mut prover)
        .unwrap()
        .finish(&mut Cache::new())
        .unwrap();
    let mut checker = verifier_transcript::<Keccak>(&tag, &root, prover.narg_string());
    group.verify(root, queries(), proof, &mut checker).unwrap();
    checker.check_eof().unwrap();
}

#[test]
fn branch_output_order_is_checked() {
    let tag = protocol_tag(b"branch order parent");
    let domain = b"branch order child";
    let a = Advice::new(false).unwrap();
    let run = |i, io: &mut Channel<'_, '_>| {
        let [answer] = io.values(|| [F::from_u64(i as u64)])?;
        if answer != F::from_u64(i as u64) {
            return Err("incorrect branch output");
        }
        Ok(answer)
    };
    let mut prover = prover_transcript::<Keccak>(&tag, b"");
    let mut messages = Messages::default();
    let expected = Channel::Send(&mut prover, &mut messages, &a)
        .fork(domain, 2, run)
        .unwrap();
    let read = |m: &Messages| -> Result<Vec<F>> {
        let mut t = verifier_transcript::<Keccak>(&tag, b"", prover.narg_string());
        let mut io = Channel::Check(&mut t, m.0.cursor(&m.1, &m.2));
        let outputs = io.fork(domain, 2, run)?;
        io.used()?;
        Ok(outputs)
    };
    assert_eq!(read(&messages), Ok(expected));
    messages.1.swap(0, 1);
    assert!(read(&messages).is_err());
}

#[test]
fn projection_combines_distinct_column_points_at_a_shared_row() {
    let shape = Shape::new(3, 2).unwrap();
    let x = [3, -2, 5, 1, -2, -3];
    let p = Projection {
        weights: &[4, 1, -3, 5],
        bias: &[0, 2],
    };
    let q = Projection {
        weights: &[1, -3, 5, 2],
        bias: &[-1, 0],
    };
    let mut advice = Advice::new(true).unwrap();
    let p_witness = Rounded::apply(&mut advice, shape, 3, p, Some(&x), true).unwrap();
    let q_witness = Rounded::apply(&mut advice, shape, 3, q, Some(&x), true).unwrap();
    advice.seal().unwrap();
    let target = |w: &Rounded, col| {
        let point = [7, 9, col].map(F::from_u64);
        at(&point, shape.eval(w.values().unwrap(), &point))
    };
    let left = target(&p_witness, 11);
    let right = target(&q_witness, 13);
    let tag = protocol_tag(b"shared projection input");
    let mut prover = prover_transcript::<Keccak>(&tag, b"");
    let mut messages = Messages::default();
    let mut io = Channel::Send(&mut prover, &mut messages, &advice);
    let batch = vec![(&p_witness, p, left), (&q_witness, q, right)];
    let claims = projections(&mut io, &mut Vec::new(), shape, 3, &batch, Some(&x)).unwrap();
    assert_eq!(claims.len(), 1);
    assert_eq!(claims[0].value, shape.eval(&x, &claims[0].point));
}

#[test]
fn normalization_preserves_pending_input_claims() {
    let shape = Shape::new(3, 3).unwrap();
    let x = [21, -12, 4, -16, 13, 11, 4, -14, 3];
    for kind in [NormalizationKind::Layer, NormalizationKind::Rms] {
        let p = NormalizationParameters {
            kind,
            gain: &[16; 3],
            bias: &[0; 3],
            epsilon_variance_units: 5,
        };
        let mut a = Advice::new(true).unwrap();
        let w = Normalized::record(&mut a, shape, 4, p, Some(&x)).unwrap();
        let claim = |v: &[i32], n| {
            let point: Vec<_> = (n..n + 4).map(F::from_u64).collect();
            at(&point, shape.eval(v, &point))
        };
        let claims = [
            claim(w.values().unwrap(), 7),
            claim(w.values().unwrap(), 13),
        ];
        let carry = [claim(&x, 19)];
        let tag = protocol_tag(b"normalization carry");
        let mut t = prover_transcript::<Keccak>(&tag, b"");
        let mut messages = Messages::default();
        let mut io = Channel::Send(&mut t, &mut messages, &a);
        let result = w
            .reverse(&mut io, &mut Vec::new(), p, Some(&x), &claims, &carry)
            .unwrap();
        assert_eq!(shape.eval(&x, &result.point), result.value);
    }
}

#[test]
fn rejects_unsafe_custom_tables() {
    assert!(ActivationTable::new(Activation {
        lower: -2,
        correction: &[0, 0, 0, 1],
    })
    .is_err());
    assert!(ActivationTable::new(Activation {
        lower: i32::MAX - 2,
        correction: &[0, 0, 1, 0],
    })
    .is_err());
    assert!(ActivationTable::new(Activation {
        lower: -2,
        correction: &[1, 2, 3, 0],
    })
    .is_ok());
    assert!(Softmax::new(
        2,
        SoftmaxMask::Full,
        SoftmaxTables {
            probabilities: &[4, 3, 2, 1],
            sums: &[4, 3, 2, 1],
            limit: 8,
        }
    )
    .is_err());
}

#[test]
fn lookup_identity_uses_table_contents() {
    let a = vec![1, 2, 3, 0];
    let b = a.clone();
    assert!(Table::Activation(&a).same(Table::Activation(&b)));
    assert!(!Table::Activation(&a).same(Table::Activation(&[1, 2, 2, 0])));
}

#[test]
fn independent_coefficient_separates_softmax_residual() {
    let error = F::from_u64(7);
    // The old equation accepts equal nonzero residuals for every row point.
    assert_eq!(error - error, F::zero());
    let residual = |rho: F| error - rho * error;
    assert_ne!(residual(F::from_u64(2)), F::zero());
    assert_ne!(residual(F::from_u64(2)), residual(F::from_u64(3)));
}

#[test]
fn rejects_footer_width_count_before_decoding() {
    let mut proof = Vec::from(MAGIC);
    proof.extend_from_slice(&16u64.to_le_bytes());
    proof.extend_from_slice(&u64::MAX.to_le_bytes());
    let mut advice = Advice::new(false).unwrap();
    assert!(footer(std::io::Cursor::new(proof), &mut advice).is_err());
}
