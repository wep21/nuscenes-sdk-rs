# nuScenes Rust bindings

This crate provides a PyO3-based drop-in for the most common nuScenes SDK data access helpers. It focuses on fast table loading and indexing while keeping the Python API surface compatible (`NuScenes`, `get`, `getind`, `field2token`, `get_sample_data_path`, `list_scenes`, etc.).

## Build

Use [maturin](https://github.com/PyO3/maturin) to build or install the extension:

```bash
cd rust/nuscenes-rs
maturin develop --release
```

This will produce the `nuscenes_rs` extension module that can be imported from Python.
