use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use ark_bn254::{Bn254, Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup, pairing::Pairing, scalar_mul::fixed_base::FixedBase};
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress};
use ark_std::{UniformRand, Zero};
use common::CommittedPoly;
use joltworks::poly::{
    commitment::hyperkzg::{
        HyperKZGCommitment, HyperKZGProverKey,
        kzg::{KZGProverKey, SRS},
    },
    one_hot_polynomial::OneHotPolynomial,
};
use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};
use rayon::prelude::*;

use crate::{ProverError, Result};

const FLAT_G1_SRS_MAGIC: &[u8; 8] = b"Q3G1SRS1";
const FLAT_G1_SRS_VERSION: u64 = 1;
const FLAT_G1_SRS_HEADER_BYTES: u64 = 8 + 8 + 8 + 8;
const HYPERKZG_SEED: &[u8; 32] = b"HyperKZG_POLY_COMMITMENT_SCHEMEE";

#[derive(Debug, Clone)]
pub struct FlatG1SrsReader {
    path: PathBuf,
    point_count: usize,
    point_size: usize,
}

impl FlatG1SrsReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut reader = BufReader::new(File::open(&path)?);
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != FLAT_G1_SRS_MAGIC {
            return Err(ProverError::InvalidInput(format!(
                "{} has invalid flat G1 SRS magic",
                path.display()
            )));
        }
        let version = read_u64(&mut reader)?;
        if version != FLAT_G1_SRS_VERSION {
            return Err(ProverError::InvalidInput(format!(
                "{} has unsupported flat G1 SRS version {version}",
                path.display()
            )));
        }
        let point_count = read_u64(&mut reader)? as usize;
        let point_size = read_u64(&mut reader)? as usize;
        let expected_size = G1Affine::identity().serialized_size(Compress::No);
        if point_size != expected_size {
            return Err(ProverError::InvalidInput(format!(
                "{} point size mismatch: expected {expected_size}, got {point_size}",
                path.display()
            )));
        }
        Ok(Self {
            path,
            point_count,
            point_size,
        })
    }

    pub fn point_count(&self) -> usize {
        self.point_count
    }

    pub fn point_size(&self) -> usize {
        self.point_size
    }

    pub fn read_chunk_bytes(&self, start: usize, len: usize) -> Result<ReadChunkBytesResult> {
        if start > self.point_count || start + len > self.point_count {
            return Err(ProverError::InvalidInput(format!(
                "flat SRS chunk [{start}, {}) exceeds point_count {}",
                start + len,
                self.point_count
            )));
        }
        let mut reader = BufReader::new(File::open(&self.path)?);
        reader.seek(SeekFrom::Start(
            FLAT_G1_SRS_HEADER_BYTES + (start as u64) * (self.point_size as u64),
        ))?;
        let mut bytes = vec![0u8; len * self.point_size];
        let t0 = Instant::now();
        reader.read_exact(&mut bytes)?;
        let read_bytes = t0.elapsed();
        Ok(ReadChunkBytesResult { bytes, read_bytes })
    }
}

#[derive(Debug)]
pub struct ReadChunkBytesResult {
    pub bytes: Vec<u8>,
    pub read_bytes: Duration,
}

pub fn write_flat_g1_srs(path: impl AsRef<Path>, num_vars: usize, chunk_len: usize) -> Result<()> {
    if chunk_len == 0 {
        return Err(ProverError::InvalidInput(
            "flat SRS chunk_len must be nonzero".to_string(),
        ));
    }
    let point_count = (1usize << num_vars) + 1;
    let point_size = G1Affine::identity().serialized_size(Compress::No);
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(FLAT_G1_SRS_MAGIC)?;
    write_u64(&mut writer, FLAT_G1_SRS_VERSION)?;
    write_u64(&mut writer, point_count as u64)?;
    write_u64(&mut writer, point_size as u64)?;

    let mut rng = ChaCha20Rng::from_seed(*HYPERKZG_SEED);
    let beta = Fr::rand(&mut rng);
    let g1 = <Bn254 as Pairing>::G1::rand(&mut rng);
    let _g2 = <Bn254 as Pairing>::G2::rand(&mut rng);

    let scalar_bits = Fr::MODULUS_BIT_SIZE as usize;
    let window_size = FixedBase::get_mul_window_size(point_count - 1);
    let table = FixedBase::get_window_table(scalar_bits, window_size, g1);

    // Match `joltworks::SRS::setup`: the first stored G1 power is beta * G,
    // not G.  The flat file must match that deterministic setup exactly so
    // streaming onehot commitments can be mixed with regular HyperKZG commits.
    let mut beta_power = beta;
    let mut written = 0usize;
    while written < point_count {
        let len = chunk_len.min(point_count - written);
        let mut scalars = Vec::with_capacity(len);
        for _ in 0..len {
            scalars.push(beta_power);
            beta_power *= beta;
        }
        let projective = FixedBase::msm(scalar_bits, window_size, &table, &scalars);
        let affine = <Bn254 as Pairing>::G1::normalize_batch(&projective);
        for point in affine {
            point.serialize_uncompressed(&mut writer)?;
        }
        written += len;
    }
    writer.flush()?;
    Ok(())
}

pub fn load_hyperkzg_setup_from_flat_g1_srs(
    reader: &FlatG1SrsReader,
    num_vars: usize,
    chunk_len: usize,
) -> Result<HyperKZGProverKey<Bn254>> {
    if chunk_len == 0 {
        return Err(ProverError::InvalidInput(
            "flat SRS load chunk_len must be nonzero".to_string(),
        ));
    }
    let required_points = (1usize << num_vars) + 1;
    if reader.point_count() < required_points {
        return Err(ProverError::InvalidInput(format!(
            "flat SRS has {} G1 points but vars={num_vars} needs {required_points}",
            reader.point_count()
        )));
    }

    let mut g1_powers = Vec::with_capacity(required_points);
    let mut start = 0usize;
    while start < required_points {
        let len = chunk_len.min(required_points - start);
        let chunk = reader.read_chunk_bytes(start, len)?;
        for point_bytes in chunk.bytes.chunks_exact(reader.point_size()) {
            let point = G1Affine::deserialize_uncompressed_unchecked(point_bytes)?;
            g1_powers.push(point);
        }
        start += len;
    }

    let mut rng = ChaCha20Rng::from_seed(*HYPERKZG_SEED);
    let beta = Fr::rand(&mut rng);
    let _g1 = <Bn254 as Pairing>::G1::rand(&mut rng);
    let g2 = <Bn254 as Pairing>::G2::rand(&mut rng);
    let mut beta_power = beta;
    let mut g2_powers = Vec::with_capacity(3);
    for _ in 0..3 {
        g2_powers.push((g2 * beta_power).into_affine());
        beta_power *= beta;
    }

    let srs = Arc::new(SRS::<Bn254> {
        g1_powers,
        g2_powers,
        // HyperKZG's current commit/open/verify paths do not read this
        // prefix-sum table. Leaving it empty avoids rebuilding the expensive
        // full setup during proof execution.
        g_products: Vec::new(),
    });
    Ok(HyperKZGProverKey {
        kzg_pk: KZGProverKey::new(srs, 0, required_points),
    })
}

#[derive(Debug)]
pub struct StreamingOneHotCommitter {
    reader: FlatG1SrsReader,
    chunk_len: usize,
    threads: Option<usize>,
    sort_indices: bool,
    report_metrics: bool,
    requests: Vec<OneHotCommitRequest>,
}

#[derive(Debug)]
struct OneHotCommitRequest {
    poly: CommittedPoly,
    buckets: Vec<Vec<usize>>,
    sum: G1Projective,
}

impl StreamingOneHotCommitter {
    pub fn new(reader: FlatG1SrsReader, chunk_len: usize) -> Result<Self> {
        Self::with_threads(reader, chunk_len, None)
    }

    pub fn with_threads(
        reader: FlatG1SrsReader,
        chunk_len: usize,
        threads: Option<usize>,
    ) -> Result<Self> {
        if chunk_len == 0 {
            return Err(ProverError::InvalidInput(
                "streaming onehot chunk_len must be nonzero".to_string(),
            ));
        }
        if threads == Some(0) {
            return Err(ProverError::InvalidInput(
                "streaming onehot threads must be nonzero".to_string(),
            ));
        }
        Ok(Self {
            reader,
            chunk_len,
            threads,
            sort_indices: false,
            report_metrics: false,
            requests: Vec::new(),
        })
    }

    pub fn with_sorted_indices(mut self, sort_indices: bool) -> Self {
        self.sort_indices = sort_indices;
        self
    }

    pub fn with_metrics(mut self, report_metrics: bool) -> Self {
        self.report_metrics = report_metrics;
        self
    }

    pub fn add_one_hot(
        &mut self,
        poly: CommittedPoly,
        one_hot: &OneHotPolynomial<Fr>,
    ) -> Result<()> {
        let chunk_count = self.reader.point_count().div_ceil(self.chunk_len);
        let mut buckets = vec![Vec::new(); chunk_count];
        let t_len = one_hot.nonzero_indices.len();
        for (t, k) in one_hot.nonzero_indices.iter().enumerate() {
            let Some(k) = k else {
                continue;
            };
            let global = (*k as usize) * t_len + t;
            if global >= self.reader.point_count() {
                return Err(ProverError::InvalidInput(format!(
                    "onehot index {global} for {poly:?} exceeds flat SRS point_count {}",
                    self.reader.point_count()
                )));
            }
            let chunk = global / self.chunk_len;
            let local = global % self.chunk_len;
            buckets[chunk].push(local);
        }
        if self.sort_indices {
            for bucket in &mut buckets {
                bucket.sort_unstable();
            }
        }
        self.requests.push(OneHotCommitRequest {
            poly,
            buckets,
            sum: G1Projective::zero(),
        });
        Ok(())
    }

    pub fn commit_all(mut self) -> Result<BTreeMap<CommittedPoly, HyperKZGCommitment<Bn254>>> {
        if let Some(threads) = self.threads {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|err| ProverError::InvalidInput(err.to_string()))?;
            return pool.install(|| self.commit_all_inner());
        }
        self.commit_all_inner()
    }

    fn commit_all_inner(&mut self) -> Result<BTreeMap<CommittedPoly, HyperKZGCommitment<Bn254>>> {
        let mut metrics = StreamingOneHotMetrics::default();
        let chunk_count = self.reader.point_count().div_ceil(self.chunk_len);
        for chunk in 0..chunk_count {
            let has_work = self
                .requests
                .iter()
                .any(|request| !request.buckets[chunk].is_empty());
            if !has_work {
                continue;
            }
            let start = chunk * self.chunk_len;
            let len = self.chunk_len.min(self.reader.point_count() - start);
            let srs_chunk = self.reader.read_chunk_bytes(start, len)?;
            metrics.read_bytes += srs_chunk.read_bytes;

            let t0 = Instant::now();
            let point_additions = if self.threads == Some(1) {
                self.add_chunk_selected_sequential(
                    chunk,
                    &srs_chunk.bytes,
                    self.reader.point_size(),
                )?
            } else {
                self.add_chunk_selected_parallel(chunk, &srs_chunk.bytes, self.reader.point_size())?
            };
            metrics.selected_deserialize_sum += t0.elapsed();
            metrics.chunks += 1;
            metrics.point_additions += point_additions;
        }
        if self.report_metrics {
            eprintln!(
                "streaming_onehot_metrics: chunks={} point_additions={} read_bytes={:.3}s selected_deserialize_sum={:.3}s",
                metrics.chunks,
                metrics.point_additions,
                metrics.read_bytes.as_secs_f64(),
                metrics.selected_deserialize_sum.as_secs_f64(),
            );
        }
        Ok(self
            .requests
            .iter()
            .map(|request| {
                (
                    request.poly,
                    HyperKZGCommitment::<Bn254>(request.sum.into_affine()),
                )
            })
            .collect())
    }

    fn add_chunk_selected_sequential(
        &mut self,
        chunk: usize,
        srs_bytes: &[u8],
        point_size: usize,
    ) -> Result<usize> {
        let mut point_additions = 0usize;
        for request in &mut self.requests {
            let mut partial = G1Projective::zero();
            for &local in &request.buckets[chunk] {
                let offset = local * point_size;
                let point = G1Affine::deserialize_uncompressed_unchecked(
                    &srs_bytes[offset..offset + point_size],
                )?;
                partial += point.into_group();
                point_additions += 1;
            }
            request.sum += partial;
        }
        Ok(point_additions)
    }

    fn add_chunk_selected_parallel(
        &mut self,
        chunk: usize,
        srs_bytes: &[u8],
        point_size: usize,
    ) -> Result<usize> {
        self.requests
            .par_iter_mut()
            .map(|request| -> Result<usize> {
                let mut partial = G1Projective::zero();
                for &local in &request.buckets[chunk] {
                    let offset = local * point_size;
                    let point = G1Affine::deserialize_uncompressed_unchecked(
                        &srs_bytes[offset..offset + point_size],
                    )?;
                    partial += point.into_group();
                }
                request.sum += partial;
                Ok(request.buckets[chunk].len())
            })
            .try_reduce(|| 0usize, |lhs, rhs| Ok(lhs + rhs))
    }
}

#[derive(Debug, Default)]
struct StreamingOneHotMetrics {
    chunks: usize,
    point_additions: usize,
    read_bytes: Duration,
    selected_deserialize_sum: Duration,
}

pub fn streaming_one_hot_commitment(
    reader: &FlatG1SrsReader,
    chunk_len: usize,
    poly: CommittedPoly,
    one_hot: &OneHotPolynomial<Fr>,
) -> Result<HyperKZGCommitment<Bn254>> {
    let mut committer = StreamingOneHotCommitter::new(reader.clone(), chunk_len)?;
    committer.add_one_hot(poly, one_hot)?;
    let mut commitments = committer.commit_all()?;
    commitments
        .remove(&poly)
        .ok_or_else(|| ProverError::InvalidInput(format!("missing streaming commitment {poly:?}")))
}

pub fn flat_srs_matches_setup(reader: &FlatG1SrsReader, setup: &HyperKZGProverKey<Bn254>) -> bool {
    reader.point_count() >= setup.kzg_pk.g1_powers().len()
}

fn read_u64(reader: &mut impl Read) -> std::io::Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn write_u64(writer: &mut impl Write, value: u64) -> std::io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

#[cfg(test)]
mod tests {
    use ark_bn254::{Bn254, Fr};
    use common::CommittedPoly;
    use joltworks::poly::{
        commitment::{commitment_scheme::CommitmentScheme, hyperkzg::HyperKZG},
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        one_hot_polynomial::OneHotPolynomial,
    };
    use joltworks::transcripts::Blake2bTranscript;

    use super::{
        FlatG1SrsReader, load_hyperkzg_setup_from_flat_g1_srs, streaming_one_hot_commitment,
        write_flat_g1_srs,
    };

    #[test]
    fn streaming_onehot_commit_matches_hyperkzg() {
        let path = std::env::temp_dir().join(format!(
            "qwen3-layer-prover-srs-{}.flat",
            std::process::id()
        ));
        write_flat_g1_srs(&path, 8, 17).unwrap();
        let reader = FlatG1SrsReader::open(&path).unwrap();

        let one_hot = OneHotPolynomial::<Fr>::from_indices(
            vec![
                Some(3),
                Some(5),
                None,
                Some(1),
                Some(0),
                Some(7),
                Some(2),
                Some(4),
            ],
            8,
        );
        let streaming =
            streaming_one_hot_commitment(&reader, 11, CommittedPoly::QwenLayerTensor(0), &one_hot)
                .unwrap();

        type PCS = HyperKZG<Bn254>;
        let setup = PCS::setup_prover(8);
        let direct = PCS::commit(&MultilinearPolynomial::OneHot(one_hot), &setup).0;
        assert_eq!(streaming, direct);

        let loaded_setup = load_hyperkzg_setup_from_flat_g1_srs(&reader, 8, 17).unwrap();
        let dense_poly = MultilinearPolynomial::from(vec![Fr::from(3); 8]);
        let loaded = PCS::commit(&dense_poly, &loaded_setup).0;
        let direct = PCS::commit(&dense_poly, &setup).0;
        assert_eq!(loaded, direct);

        type Challenge = <Fr as joltworks::field::JoltField>::Challenge;
        let point = vec![
            Challenge::from(2u128),
            Challenge::from(5u128),
            Challenge::from(7u128),
        ];
        let opening = dense_poly.evaluate(&point);
        let mut prover_transcript = Blake2bTranscript::default();
        let proof = PCS::prove(&loaded_setup, &dense_poly, &point, None, &mut prover_transcript);
        let verifier_setup = PCS::setup_verifier(&loaded_setup);
        let mut verifier_transcript = Blake2bTranscript::default();
        PCS::verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &point,
            &opening,
            &loaded,
        )
        .unwrap();
        let _ = std::fs::remove_file(path);
    }
}
