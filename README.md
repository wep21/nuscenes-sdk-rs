# nuScenes Rust bindings

This crate provides a PyO3-based drop-in for the most common nuScenes SDK data access helpers. It focuses on fast table loading and indexing while keeping the Python API surface compatible (`NuScenes`, `get`, `getind`, `field2token`, `get_sample_data_path`, `list_scenes`, etc.).

## Build

```bash
uv sync
```

## How to use

```python
from nuscenes_rs import NuScenes
```

This will produce the `nuscenes_rs` extension module that can be imported from Python.
