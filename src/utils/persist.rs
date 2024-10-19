use std::fs;
use std::path::Path;
use serde::{Serialize, Deserialize};
use bincode;
use anyhow::Result;

#[cfg(not(feature = "shortdeck"))]
const PREFIX: &str = "cache/holdem-";

#[cfg(feature = "shortdeck")]
const PREFIX: &str = "cache/shortdeck-";

pub fn persisted_function<F, T>(name: &str, f: F) -> Result<T>
where
    F: FnOnce() -> T,
    T: Serialize + for<'de> Deserialize<'de>,
{
    let path = format!("{}{}.lz4", &PREFIX, name);
    let path = Path::new(&path);

    if path.exists() {
        log::info!("Loading from cache: {}", name);
        let compressed_data = fs::read(path)?;
        let decompressed_data = lz4_flex::decompress_size_prepended(&compressed_data).unwrap();
        let result: T = bincode::deserialize(&decompressed_data)?;
        Ok(result)
    } else {
        let result = f();
        let serialized_data = bincode::serialize(&result)?;
        let compressed_data = lz4_flex::compress_prepend_size(&serialized_data);
        fs::write(path, &compressed_data)?;
        log::info!("Saved to cache: {}", name);
        Ok(result)
    }
}

pub fn try_load<T>(path: &Path) -> Result<T>
where
    T: Serialize + for<'de> Deserialize<'de>,
{
    if path.exists() {
        log::info!("Loading from cache: {:?}", path);
        let compressed_data = fs::read(path)?;
        let decompressed_data = lz4_flex::decompress_size_prepended(&compressed_data).unwrap();
        let result: T = bincode::deserialize(&decompressed_data)?;
        Ok(result)
    } else {
        anyhow::bail!("Cache not found: {:?}", path);
    }
}