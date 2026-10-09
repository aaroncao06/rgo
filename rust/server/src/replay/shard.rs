//! Export selected packed record fields as board-size arrays in one NPZ.

use npyz::{AutoSerialize, WriterBuilder, half::f16};
use rgo_artifacts::{
    SUPPORTED_BOARD_DIMS,
    chunk::{NUM_GLOBAL_FEATURES, NUM_SPATIAL_FEATURES},
};
use std::io::{self, BufWriter, Seek, Write};
use zip::write::FileOptions;

use super::{Result, chunk_reader::Record};

pub(super) fn write(
    groups: [Vec<Record<'_>>; SUPPORTED_BOARD_DIMS.len()],
    output: &mut (impl Write + Seek),
) -> Result<()> {
    let mut archive = zip::ZipWriter::new(output);
    array(&mut archive, "format_version", &[], [1u32])?;
    let board_sizes: Vec<_> = SUPPORTED_BOARD_DIMS
        .into_iter()
        .zip(&groups)
        .filter(|(_, rows)| !rows.is_empty())
        .map(|(dim, _)| dim as u8)
        .collect();
    array(
        &mut archive,
        "board_sizes",
        &[board_sizes.len() as u64],
        board_sizes,
    )?;
    for (dim, rows) in SUPPORTED_BOARD_DIMS.into_iter().zip(groups) {
        if rows.is_empty() {
            continue;
        }
        let n = rows.len() as u64;
        let area = dim * dim;
        let prefix = dim.to_string();
        array(
            &mut archive,
            &format!("{prefix}/spatial"),
            &[n, NUM_SPATIAL_FEATURES as u64, area.div_ceil(8) as u64],
            rows.iter()
                .flat_map(|record| record.spatial_bytes().iter().copied()),
        )?;
        array(
            &mut archive,
            &format!("{prefix}/global"),
            &[n, NUM_GLOBAL_FEATURES as u64],
            rows.iter().flat_map(Record::global),
        )?;
        array(
            &mut archive,
            &format!("{prefix}/policy"),
            &[n, (area + 1) as u64],
            rows.iter()
                .flat_map(|record| record.policy_bits().map(f16::from_bits)),
        )?;
        array(
            &mut archive,
            &format!("{prefix}/value"),
            &[n, 2],
            rows.iter().flat_map(Record::value),
        )?;
        array(
            &mut archive,
            &format!("{prefix}/ownership"),
            &[n, (2 * area).div_ceil(8) as u64],
            rows.iter()
                .flat_map(|record| record.ownership_bytes().iter().copied()),
        )?;
    }
    // Drop-based ZIP/NPY finalization ignores errors; finish explicitly instead.
    archive.finish().map_err(io::Error::from)?.flush()?;
    Ok(())
}

fn array<T: AutoSerialize>(
    archive: &mut zip::ZipWriter<impl Write + Seek>,
    name: &str,
    shape: &[u64],
    values: impl IntoIterator<Item = T>,
) -> io::Result<()> {
    // Stored entries preserve packed size and avoid compression CPU overhead.
    archive.start_file(
        npyz::npz::file_name_from_array_name(name),
        FileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .large_file(true),
    )?;
    // Buffer inside the ZIP entry, so small scalar writes also batch CRC work.
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(shape)
        .writer(BufWriter::new(archive))
        .begin_nd()?;
    writer.extend(values)?;
    writer.finish()
}
