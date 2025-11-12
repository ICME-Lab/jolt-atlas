use core::fmt;

use tabled::Tabled;

use crate::{tensor::Tensor, trace_types::ONNXInstr};

/// Helper function to format optional values for display
fn display_option<T: fmt::Display>(opt: &Option<T>) -> String {
    match opt {
        Some(val) => val.to_string(),
        None => String::new(),
    }
}

/// Helper function to format immediate tensors with truncation for large values
fn display_imm(imm: &Option<Tensor<i32>>) -> String {
    match imm {
        None => String::new(),
        Some(tensor) => {
            const MAX_DISPLAY: usize = 6;
            const SHOW_EACH_SIDE: usize = 2;
            let len = tensor.inner.len();
            if len <= MAX_DISPLAY {
                format!("{:?}", tensor.inner)
            } else {
                let start: String = tensor
                    .inner
                    .iter()
                    .take(SHOW_EACH_SIDE)
                    .map(|n| format!("{n}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let end: String = tensor
                    .inner
                    .iter()
                    .skip(len - SHOW_EACH_SIDE)
                    .map(|n| format!("{n}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("[{start}...{end}] ({len})")
            }
        }
    }
}

/// Helper function to format inputs as a compact string
fn display_inputs(ts1: &Option<usize>, ts2: &Option<usize>, ts3: &Option<usize>) -> String {
    let mut inputs = Vec::new();
    if let Some(t1) = ts1 {
        inputs.push(format!("ts1={t1}"));
    }
    if let Some(t2) = ts2 {
        inputs.push(format!("ts2={t2}"));
    }
    if let Some(t3) = ts3 {
        inputs.push(format!("ts3={t3}"));
    }
    if inputs.is_empty() {
        String::new()
    } else {
        inputs.join(", ")
    }
}

impl fmt::Debug for ONNXInstr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ONNXInstr")
            .field("address", &self.address)
            .field("opcode", &self.opcode)
            .field("ts1", &self.ts1)
            .field("ts2", &self.ts2)
            .field("ts3", &self.ts3)
            .field("td", &self.td)
            .field("imm", &display_imm(&self.imm))
            .field("virtual_seq_remaining", &self.virtual_sequence_remaining)
            .field("output_dims", &self.output_dims)
            .finish()
    }
}

impl Tabled for ONNXInstr {
    const LENGTH: usize = 7;

    fn headers() -> Vec<std::borrow::Cow<'static, str>> {
        vec![
            std::borrow::Cow::Borrowed("address"),
            std::borrow::Cow::Borrowed("opcode"),
            std::borrow::Cow::Borrowed("inputs"),
            std::borrow::Cow::Borrowed("td"),
            std::borrow::Cow::Borrowed("imm"),
            std::borrow::Cow::Borrowed("output_dims"),
            std::borrow::Cow::Borrowed("active_elems"),
        ]
    }

    fn fields(&self) -> Vec<std::borrow::Cow<'_, str>> {
        vec![
            std::borrow::Cow::Owned(self.address.to_string()),
            std::borrow::Cow::Owned(format!("{:?}", self.opcode)),
            std::borrow::Cow::Owned(display_inputs(&self.ts1, &self.ts2, &self.ts3)),
            std::borrow::Cow::Owned(display_option(&self.td)),
            std::borrow::Cow::Owned(display_imm(&self.imm)),
            std::borrow::Cow::Owned(format!("{:?}", self.output_dims)),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_display_imm() {
        let small_imm = Tensor::new(Some(&[1, 2, 3]), &[1, 3]).ok();
        let large_imm = Tensor::new(Some(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), &[1, 10]).ok();
        let none_imm: Option<Tensor<i32>> = None;
        assert_eq!(display_imm(&small_imm), "[1, 2, 3]");
        assert_eq!(display_imm(&large_imm), "[0, 1...8, 9] (10)");
        assert_eq!(display_imm(&none_imm), "");
    }
}
