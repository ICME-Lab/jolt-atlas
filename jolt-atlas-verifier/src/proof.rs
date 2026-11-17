//! The proof: the byte string the prover's transcript wrote.

/// Proof for an ONNX neural network computation: the NARG string of the prover's transcript,
/// which the verifier's transcript reads front to back.
///
/// The bytes are the prover's messages in the order the verifier needs them, with no key, tag
/// or count between them:
///
/// 1. the batch commitment, then one commitment per polynomial the model's schedule opens;
/// 2. the output claim;
/// 3. for each step of the schedule, what the step reads: an opening's claims, an evaluation
///    reduction's polynomial, a sumcheck's round messages (each followed by its proof-of-work
///    nonce) and then its members' produced claims, a tie's produced claims;
/// 4. the opening reduction's sumcheck and its reduced claims, then the joint opening proof,
///    when the schedule opens a committed polynomial.
///
/// How many values there are and how long each is comes from the schedule, never from the
/// bytes; the one length the bytes state is that of a framed value (a commitment, the joint
/// opening proof), bounded by the bytes left. A missing value is a short read and a surplus
/// one is trailing bytes: both are rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ONNXProof(pub Vec<u8>);
