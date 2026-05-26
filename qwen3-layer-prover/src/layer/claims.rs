use joltworks::{field::JoltField, transcripts::Transcript};

use super::types::LayerShape;

// Claims that start or leave the layer IOP.
//
// The output claim is not caller-provided.  It is sampled after all layer
// commitments have been absorbed, then evaluated from the committed
// `hidden_out` witness.  The verifier repeats the same challenge draw and
// checks that the proof uses that point.

pub(crate) fn draw_hidden_out_point<F, T>(transcript: &mut T, shape: &LayerShape) -> Vec<F>
where
    F: JoltField,
    T: Transcript,
{
    transcript.challenge_vector::<F>(shape.hidden_shape().padded_power_of_two().point_len())
}
