use crate::ops::InputType;
use serde::{ser::SerializeStruct, Deserialize, Deserializer, Serialize, Serializer};
use std::{io::Read, panic::UnwindSafe};
use tract_onnx::tract_core::{
    tract_data::{prelude::Tensor as TractTensor, TVec},
    value::TValue,
};

///
#[derive(Clone, Debug, PartialOrd, PartialEq)]
pub enum FileSourceInner {
    /// Inner elements of float inputs coming from a file
    Float(f64),
    /// Inner elements of bool inputs coming from a file
    Bool(bool),
}

impl FileSourceInner {
    ///
    pub fn is_float(&self) -> bool {
        matches!(self, FileSourceInner::Float(_))
    }
    ///
    pub fn is_bool(&self) -> bool {
        matches!(self, FileSourceInner::Bool(_))
    }
}

impl Serialize for FileSourceInner {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            FileSourceInner::Bool(data) => data.serialize(serializer),
            FileSourceInner::Float(data) => data.serialize(serializer),
        }
    }
}

// !!! ALWAYS USE JSON SERIALIZATION FOR GRAPH INPUT
// UNTAGGED ENUMS WONT WORK :( as highlighted here:
impl<'de> Deserialize<'de> for FileSourceInner {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let this_json: Box<serde_json::value::RawValue> = Deserialize::deserialize(deserializer)?;

        let bool_try: Result<bool, _> = serde_json::from_str(this_json.get());
        if let Ok(t) = bool_try {
            return Ok(FileSourceInner::Bool(t));
        }
        let float_try: Result<f64, _> = serde_json::from_str(this_json.get());
        if let Ok(t) = float_try {
            return Ok(FileSourceInner::Float(t));
        }

        Err(serde::de::Error::custom(
            "failed to deserialize FileSourceInner",
        ))
    }
}

/// Elements of inputs coming from a file
pub type FileSource = Vec<Vec<FileSourceInner>>;

impl FileSourceInner {
    /// Create a new FileSourceInner
    pub fn new_float(f: f64) -> Self {
        FileSourceInner::Float(f)
    }

    /// Create a new FileSourceInner
    pub fn new_bool(f: bool) -> Self {
        FileSourceInner::Bool(f)
    }

    ///
    pub fn as_type(&mut self, input_type: &InputType) {
        match self {
            FileSourceInner::Float(f) => input_type.roundtrip(f),
            FileSourceInner::Bool(_) => assert!(matches!(input_type, InputType::Bool)),
        }
    }

    /// Convert to a float
    pub fn to_float(&self) -> f64 {
        match self {
            FileSourceInner::Float(f) => *f,
            FileSourceInner::Bool(f) => {
                if *f {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

/// Enum that defines source of the inputs/outputs to the EZKL model
#[derive(Clone, Debug, Serialize, PartialOrd, PartialEq)]
#[serde(untagged)]
pub enum DataSource {
    /// .json File data source.
    File(FileSource),
}

impl Default for DataSource {
    fn default() -> Self {
        DataSource::File(vec![vec![]])
    }
}

impl From<FileSource> for DataSource {
    fn from(data: FileSource) -> Self {
        DataSource::File(data)
    }
}

impl From<Vec<Vec<f64>>> for DataSource {
    fn from(data: Vec<Vec<f64>>) -> Self {
        DataSource::File(
            data.iter()
                .map(|e| e.iter().map(|e| FileSourceInner::Float(*e)).collect())
                .collect(),
        )
    }
}

// !!! ALWAYS USE JSON SERIALIZATION FOR GRAPH INPUT
// UNTAGGED ENUMS WONT WORK :( as highlighted here:
impl<'de> Deserialize<'de> for DataSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let this_json: Box<serde_json::value::RawValue> = Deserialize::deserialize(deserializer)?;

        let first_try: Result<FileSource, _> = serde_json::from_str(this_json.get());

        if let Ok(t) = first_try {
            return Ok(DataSource::File(t));
        }

        Err(serde::de::Error::custom("failed to deserialize DataSource"))
    }
}

/// Input to graph as a datasource
/// Always use JSON serialization for GraphData. Seriously.
#[derive(Clone, Debug, Deserialize, Default, PartialEq)]
pub struct GraphData {
    /// Inputs to the model / computational graph (can be empty vectors if inputs are
    /// coming from on-chain).
    pub input_data: DataSource,
    /// Outputs of the model / computational graph (can be empty vectors if outputs are
    /// coming from on-chain).
    pub output_data: Option<DataSource>,
}

impl UnwindSafe for GraphData {}

impl GraphData {
    // not wasm
    #[cfg(not(target_arch = "wasm32"))]
    /// Convert the input data to tract data
    pub fn to_tract_data(
        &self,
        shapes: &[Vec<usize>],
        datum_types: &[tract_onnx::prelude::DatumType],
    ) -> Result<TVec<TValue>, Box<dyn std::error::Error>> {
        let mut inputs = TVec::new();
        match &self.input_data {
            DataSource::File(data) => {
                for (i, input) in data.iter().enumerate() {
                    if !input.is_empty() {
                        let dt = datum_types[i];
                        let input = input.iter().map(|e| e.to_float()).collect::<Vec<f64>>();
                        let tt = TractTensor::from_shape(&shapes[i], &input)?;
                        let tt = tt.cast_to_dt(dt)?;
                        inputs.push(tt.into_owned().into());
                    }
                }
            }
        }
        Ok(inputs)
    }

    ///
    pub fn new(input_data: DataSource) -> Self {
        GraphData {
            input_data,
            output_data: None,
        }
    }

    /// Load the model input from a file
    pub fn from_path(path: std::path::PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = std::fs::File::open(path.clone())
            .map_err(|_| format!("failed to open input at {}", path.display()))?;
        let mut data = String::new();
        file.read_to_string(&mut data)?;
        serde_json::from_str(&data).map_err(|e| e.into())
    }

    /// Save the model input to a file
    pub fn save(&self, path: std::path::PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        serde_json::to_writer(std::fs::File::create(path)?, &self).map_err(|e| e.into())
    }
}

impl Serialize for GraphData {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("GraphData", 4)?;
        state.serialize_field("input_data", &self.input_data)?;
        state.serialize_field("output_data", &self.output_data)?;
        state.end()
    }
}
