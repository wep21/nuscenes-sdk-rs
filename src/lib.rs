use pyo3::conversion::IntoPyObject;
use pyo3::exceptions::{PyAttributeError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyModule, PyString, PyTuple};
use pyo3::{Bound, Py};
use rayon::prelude::*;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Minimal Rust-backed nuScenes loader that mirrors the Python SDK's data-access APIs.
#[pyclass(module = "nuscenes_rs")]
pub struct NuScenes {
    version: String,
    dataroot: String,
    verbose: bool,
    map_resolution: f64,
    table_names: Vec<String>,
    colormap: PyOnceLock<Py<PyAny>>,
    tables: HashMap<String, Vec<Value>>,
    py_tables: HashMap<String, Py<PyAny>>,
    token2ind: HashMap<String, HashMap<String, usize>>,
    lidarseg_idx2name_mapping: HashMap<i64, String>,
    lidarseg_name2idx_mapping: HashMap<String, i64>,
}

#[pymethods]
impl NuScenes {
    #[new]
    #[pyo3(
        signature = (
            version = "v1.0-mini",
            dataroot = "/data/sets/nuscenes",
            verbose = true,
            map_resolution = 0.1,
            colormap = None
        )
    )]
    fn new(
        py: Python<'_>,
        version: &str,
        dataroot: &str,
        verbose: bool,
        map_resolution: f64,
        colormap: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let table_root = Path::new(dataroot).join(version);
        if !table_root.exists() {
            return Err(PyValueError::new_err(format!(
                "Database version not found: {}",
                table_root.display()
            )));
        }

        if verbose {
            println!("======\nLoading NuScenes tables for version {}...", version);
        }

        let mut table_names = vec![
            "category".to_string(),
            "attribute".to_string(),
            "visibility".to_string(),
            "instance".to_string(),
            "sensor".to_string(),
            "calibrated_sensor".to_string(),
            "ego_pose".to_string(),
            "log".to_string(),
            "scene".to_string(),
            "sample".to_string(),
            "sample_data".to_string(),
            "sample_annotation".to_string(),
            "map".to_string(),
        ];

        let mut tables = py.detach(|| load_tables(&table_root, &table_names))?;

        let mut lidarseg_idx2name_mapping: HashMap<i64, String> = HashMap::new();
        let mut lidarseg_name2idx_mapping: HashMap<String, i64> = HashMap::new();
        for lidar_task in ["lidarseg", "panoptic"] {
            if table_root.join(format!("{lidar_task}.json")).exists() {
                if verbose {
                    println!("Loading nuScenes-{lidar_task}...");
                }
                tables.insert(lidar_task.to_string(), load_table(&table_root, lidar_task)?);
                table_names.push(lidar_task.to_string());
            }
        }

        if table_root.join("image_annotations.json").exists() {
            tables.insert(
                "image_annotations".to_string(),
                load_table(&table_root, "image_annotations")?,
            );
            table_names.push("image_annotations".to_string());
        }

        let mut token2ind = py.detach(|| build_token_index(&tables));
        py.detach(|| decorate_tables(&mut tables, &token2ind))?;

        // Rebuild the token index to ensure it reflects any new tables that were added.
        token2ind = py.detach(|| build_token_index(&tables));

        if tables.contains_key("lidarseg") {
            let (idx2name, name2idx) = build_lidarseg_mappings(&tables)?;
            lidarseg_idx2name_mapping = idx2name;
            lidarseg_name2idx_mapping = name2idx;
        }

        let py_tables = to_py_tables(py, &tables)?;

        let colormap_cell = PyOnceLock::new();
        let initial_cmap = if let Some(c) = colormap {
            c
        } else {
            build_colormap(py, &tables)?
        };
        let _ = colormap_cell.set(py, initial_cmap.clone_ref(py));

        if verbose {
            for name in &table_names {
                let len = tables.get(name).map(|v| v.len()).unwrap_or_default();
                println!("{len} {name},");
            }
            println!("Done loading.\n======");
        }

        Ok(Self {
            version: version.to_string(),
            dataroot: dataroot.to_string(),
            verbose,
            map_resolution,
            table_names,
            colormap: colormap_cell,
            tables,
            py_tables,
            token2ind,
            lidarseg_idx2name_mapping,
            lidarseg_name2idx_mapping,
        })
    }

    #[getter]
    fn version(&self) -> &str {
        &self.version
    }

    #[getter]
    fn dataroot(&self) -> &str {
        &self.dataroot
    }

    #[getter]
    fn verbose(&self) -> bool {
        self.verbose
    }

    #[getter]
    fn map_resolution(&self) -> f64 {
        self.map_resolution
    }

    #[getter]
    fn table_names(&self) -> Vec<String> {
        self.table_names.clone()
    }

    #[getter]
    fn colormap(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let cmap = self
            .colormap
            .get_or_try_init(py, || build_colormap(py, &self.tables))?;
        Ok(cmap.clone_ref(py))
    }

    #[getter]
    fn table_root(&self) -> String {
        Path::new(&self.dataroot)
            .join(&self.version)
            .to_string_lossy()
            .to_string()
    }

    #[getter]
    fn lidarseg_idx2name_mapping(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (k, v) in &self.lidarseg_idx2name_mapping {
            dict.set_item(k, v)?;
        }
        Ok(dict.unbind().into_any())
    }

    #[getter]
    fn lidarseg_name2idx_mapping(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (k, v) in &self.lidarseg_name2idx_mapping {
            dict.set_item(k, v)?;
        }
        Ok(dict.unbind().into_any())
    }

    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        if let Some(table) = self.py_tables.get(name) {
            return Ok(table.clone_ref(py));
        }
        match name {
            "colormap" => return self.colormap(py),
            "lidarseg_idx2name_mapping" => return self.lidarseg_idx2name_mapping(py),
            "lidarseg_name2idx_mapping" => return self.lidarseg_name2idx_mapping(py),
            _ => Err(PyAttributeError::new_err(format!(
                "{name} is not a known table"
            ))),
        }
    }

    /// Returns a record from a table in constant time using token lookup.
    fn get(&self, py: Python<'_>, table_name: &str, token: &str) -> PyResult<Py<PyAny>> {
        let idx = self.getind(table_name, token)?;
        let record = self
            .tables
            .get(table_name)
            .and_then(|v| v.get(idx))
            .ok_or_else(|| {
                PyKeyError::new_err(format!("Token {token} not found in {table_name}"))
            })?;
        json_to_py(py, record)
    }

    /// Returns the index of a record within a table.
    fn getind(&self, table_name: &str, token: &str) -> PyResult<usize> {
        ensure_table_exists(&self.table_names, table_name)?;
        self.token2ind
            .get(table_name)
            .and_then(|m| m.get(token).copied())
            .ok_or_else(|| PyKeyError::new_err(format!("Token {token} not found in {table_name}")))
    }

    /// Queries all records for a certain field value and returns matching tokens.
    fn field2token(
        &self,
        py: Python<'_>,
        table_name: &str,
        field: &str,
        query: Py<PyAny>,
    ) -> PyResult<Vec<String>> {
        ensure_table_exists(&self.table_names, table_name)?;
        let query_value: Value = py_to_json_value(&query.bind(py))?;
        py.detach(|| {
            let mut matches = Vec::new();
            if let Some(entries) = self.tables.get(table_name) {
                matches.reserve(entries.len() / 4); // heuristic to reduce reallocs
                for entry in entries.iter() {
                    if entry
                        .get(field)
                        .filter(|val| *val == &query_value)
                        .is_some()
                    {
                        if let Some(token) = entry.get("token").and_then(Value::as_str) {
                            matches.push(token.to_string());
                        }
                    }
                }
            }
            Ok(matches)
        })
    }

    /// Returns the path to a sample_data file relative to the dataroot.
    fn get_sample_data_path(&self, sample_data_token: &str) -> PyResult<String> {
        let record = self.get_record("sample_data", sample_data_token)?;
        let filename = expect_str_field(record, "filename")?;
        Ok(Path::new(&self.dataroot)
            .join(filename)
            .to_string_lossy()
            .to_string())
    }

    /// Lists scene names. Mirrors the Python helper but returns the list for easier use.
    fn list_scenes(&self) -> PyResult<Vec<String>> {
        let mut scenes = Vec::new();
        ensure_table_exists(&self.table_names, "scene")?;
        if let Some(entries) = self.tables.get("scene") {
            for entry in entries {
                if let Some(name) = entry.get("name").and_then(Value::as_str) {
                    scenes.push(name.to_string());
                }
            }
        }
        if self.verbose {
            for name in &scenes {
                println!("{name}");
            }
        }
        Ok(scenes)
    }

    /// Estimate velocity for an annotation; mirrors Python SDK signature.
    #[pyo3(signature = (sample_annotation_token, max_time_diff=1.5))]
    fn box_velocity(
        &self,
        py: Python<'_>,
        sample_annotation_token: &str,
        max_time_diff: f64,
    ) -> PyResult<Py<PyAny>> {
        let mut max_dt = max_time_diff;

        let current = self.get_record("sample_annotation", sample_annotation_token)?;
        let prev_token = current
            .get("prev")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let next_token = current
            .get("next")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let has_prev = !prev_token.is_empty();
        let has_next = !next_token.is_empty();

        if !has_prev && !has_next {
            return Ok(numpy_array(py, &[f64::NAN, f64::NAN, f64::NAN])?);
        }

        let first = if has_prev {
            self.get_record("sample_annotation", &prev_token)?
        } else {
            current
        };
        let last = if has_next {
            self.get_record("sample_annotation", &next_token)?
        } else {
            current
        };

        let pos_last = vec3_from_field(last, "translation")?;
        let pos_first = vec3_from_field(first, "translation")?;
        let pos_diff: Vec<f64> = pos_last
            .iter()
            .zip(pos_first.iter())
            .map(|(a, b)| a - b)
            .collect();

        let time_last = 1e-6 * sample_timestamp(self, last)?;
        let time_first = 1e-6 * sample_timestamp(self, first)?;
        let time_diff = time_last - time_first;

        if has_next && has_prev {
            max_dt *= 2.0;
        }

        if time_diff > max_dt {
            Ok(numpy_array(py, &[f64::NAN, f64::NAN, f64::NAN])?)
        } else {
            let vel: Vec<f64> = pos_diff.iter().map(|d| d / time_diff).collect();
            Ok(numpy_array(py, &vel)?)
        }
    }
}

#[pymodule]
fn nuscenes_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NuScenes>()?;
    Ok(())
}

fn load_tables(table_root: &Path, table_names: &[String]) -> PyResult<HashMap<String, Vec<Value>>> {
    table_names
        .par_iter()
        .map(|name| load_table(table_root, name).map(|table| (name.clone(), table)))
        .collect()
}

fn load_table(table_root: &Path, table_name: &str) -> PyResult<Vec<Value>> {
    let path = table_root.join(format!("{table_name}.json"));
    let data = fs::read_to_string(&path).map_err(|err| {
        PyValueError::new_err(format!("Failed to read {}: {err}", path.to_string_lossy()))
    })?;
    serde_json::from_str(&data).map_err(|err| {
        PyValueError::new_err(format!("Failed to parse {}: {err}", path.to_string_lossy()))
    })
}

fn build_token_index(
    tables: &HashMap<String, Vec<Value>>,
) -> HashMap<String, HashMap<String, usize>> {
    tables
        .par_iter()
        .map(|(name, entries)| {
            let indices = entries
                .par_iter()
                .enumerate()
                .filter_map(|(idx, entry)| {
                    entry
                        .get("token")
                        .and_then(Value::as_str)
                        .map(|token| (token.to_string(), idx))
                })
                .collect();
            (name.clone(), indices)
        })
        .collect()
}

fn decorate_tables(
    tables: &mut HashMap<String, Vec<Value>>,
    token2ind: &HashMap<String, HashMap<String, usize>>,
) -> PyResult<()> {
    decorate_sample_annotation(tables, token2ind)?;
    decorate_sample_data(tables, token2ind)?;
    decorate_samples(tables, token2ind)?;
    Ok(())
}

fn decorate_sample_annotation(
    tables: &mut HashMap<String, Vec<Value>>,
    token2ind: &HashMap<String, HashMap<String, usize>>,
) -> PyResult<()> {
    if tables.get("instance").is_none() || tables.get("category").is_none() {
        return Ok(());
    }

    let Some(mut sample_annotations) = tables.remove("sample_annotation") else {
        return Ok(());
    };

    {
        let instances = tables
            .get("instance")
            .expect("instance table presence was checked above");
        let categories = tables
            .get("category")
            .expect("category table presence was checked above");

        for record in sample_annotations.iter_mut() {
            let inst_token = expect_str_field(record, "instance_token")?;
            let inst_idx = *token2ind
                .get("instance")
                .and_then(|m| m.get(inst_token))
                .ok_or_else(|| {
                    PyKeyError::new_err(format!("instance token {inst_token} not found"))
                })?;
            let inst = instances
                .get(inst_idx)
                .ok_or_else(|| PyKeyError::new_err(format!("instance index {inst_idx} missing")))?;
            let cat_token = expect_str_field(inst, "category_token")?;
            let cat_idx = *token2ind
                .get("category")
                .and_then(|m| m.get(cat_token))
                .ok_or_else(|| {
                    PyKeyError::new_err(format!("category token {cat_token} not found"))
                })?;
            let cat_name = expect_str_field(
                categories.get(cat_idx).ok_or_else(|| {
                    PyKeyError::new_err(format!("category index {cat_idx} missing"))
                })?,
                "name",
            )?
            .to_string();
            insert_value(record, "category_name", Value::String(cat_name));
        }
    }

    tables.insert("sample_annotation".to_string(), sample_annotations);
    Ok(())
}

fn decorate_sample_data(
    tables: &mut HashMap<String, Vec<Value>>,
    token2ind: &HashMap<String, HashMap<String, usize>>,
) -> PyResult<()> {
    if tables.get("calibrated_sensor").is_none() || tables.get("sensor").is_none() {
        return Ok(());
    }

    let Some(mut sample_data) = tables.remove("sample_data") else {
        return Ok(());
    };

    {
        let calibrated_sensor = tables
            .get("calibrated_sensor")
            .expect("calibrated_sensor presence was checked above");
        let sensor = tables
            .get("sensor")
            .expect("sensor presence was checked above");

        for record in sample_data.iter_mut() {
            let cs_token = expect_str_field(record, "calibrated_sensor_token")?;
            let cs_idx = *token2ind
                .get("calibrated_sensor")
                .and_then(|m| m.get(cs_token))
                .ok_or_else(|| {
                    PyKeyError::new_err(format!("calibrated_sensor token {cs_token} not found"))
                })?;
            let cs_record = calibrated_sensor.get(cs_idx).ok_or_else(|| {
                PyKeyError::new_err(format!("calibrated_sensor index {cs_idx} missing"))
            })?;
            let sensor_token = expect_str_field(cs_record, "sensor_token")?;
            let sensor_idx = *token2ind
                .get("sensor")
                .and_then(|m| m.get(sensor_token))
                .ok_or_else(|| {
                    PyKeyError::new_err(format!("sensor token {sensor_token} not found"))
                })?;
            let sensor_record = sensor
                .get(sensor_idx)
                .ok_or_else(|| PyKeyError::new_err(format!("sensor index {sensor_idx} missing")))?;
            let modality = expect_str_field(sensor_record, "modality")?.to_string();
            let channel = expect_str_field(sensor_record, "channel")?.to_string();
            insert_value(record, "sensor_modality", Value::String(modality));
            insert_value(record, "channel", Value::String(channel));
        }
    }

    tables.insert("sample_data".to_string(), sample_data);
    Ok(())
}

fn decorate_samples(
    tables: &mut HashMap<String, Vec<Value>>,
    token2ind: &HashMap<String, HashMap<String, usize>>,
) -> PyResult<()> {
    let Some(mut samples) = tables.remove("sample") else {
        return Ok(());
    };

    for record in samples.iter_mut() {
        insert_value(record, "data", Value::Object(Map::new()));
        insert_value(record, "anns", Value::Array(Vec::new()));
    }

    if let Some(sample_data) = tables.get("sample_data") {
        let sample_index = token2ind
            .get("sample")
            .ok_or_else(|| PyKeyError::new_err("missing sample token index"))?;
        for sd in sample_data.iter() {
            let is_key_frame = expect_bool_field(sd, "is_key_frame")?;
            if !is_key_frame {
                continue;
            }
            let sample_token = expect_str_field(sd, "sample_token")?;
            let channel = expect_str_field(sd, "channel")?;
            let sd_token = expect_str_field(sd, "token")?;
            if let Some(idx) = sample_index.get(sample_token) {
                if let Some(sample_record) = samples.get_mut(*idx) {
                    let data_entry = ensure_object_mut(sample_record)?
                        .entry("data")
                        .or_insert_with(|| Value::Object(Map::new()));
                    if let Value::Object(map) = data_entry {
                        map.insert(channel.to_string(), Value::String(sd_token.to_string()));
                    }
                }
            }
        }
    }

    if let Some(sample_annotations) = tables.get("sample_annotation") {
        let sample_index = token2ind
            .get("sample")
            .ok_or_else(|| PyKeyError::new_err("missing sample token index"))?;
        for ann in sample_annotations.iter() {
            let sample_token = expect_str_field(ann, "sample_token")?;
            let ann_token = expect_str_field(ann, "token")?;
            if let Some(idx) = sample_index.get(sample_token) {
                if let Some(sample_record) = samples.get_mut(*idx) {
                    let anns_entry = ensure_object_mut(sample_record)?
                        .entry("anns")
                        .or_insert_with(|| Value::Array(Vec::new()));
                    if let Value::Array(list) = anns_entry {
                        list.push(Value::String(ann_token.to_string()));
                    }
                }
            }
        }
    }

    tables.insert("sample".to_string(), samples);
    Ok(())
}

fn to_py_tables(
    py: Python<'_>,
    tables: &HashMap<String, Vec<Value>>,
) -> PyResult<HashMap<String, Py<PyAny>>> {
    let mut out = HashMap::new();
    for (name, entries) in tables {
        let list = PyList::empty(py);
        for entry in entries {
            let py_obj = json_to_py(py, entry)?;
            list.append(py_obj)?;
        }
        out.insert(name.clone(), list.into_any().unbind());
    }
    Ok(out)
}

fn build_lidarseg_mappings(
    tables: &HashMap<String, Vec<Value>>,
) -> PyResult<(HashMap<i64, String>, HashMap<String, i64>)> {
    let categories = tables
        .get("category")
        .ok_or_else(|| PyValueError::new_err("category table missing"))?;
    let mut idx2name = HashMap::new();
    let mut name2idx = HashMap::new();
    for cat in categories {
        let idx = cat
            .get("index")
            .and_then(Value::as_i64)
            .ok_or_else(|| PyValueError::new_err("category index missing"))?;
        let name = expect_str_field(cat, "name")?.to_string();
        idx2name.insert(idx, name.clone());
        name2idx.insert(name, idx);
    }
    Ok((idx2name, name2idx))
}

fn build_colormap(py: Python<'_>, tables: &HashMap<String, Vec<Value>>) -> PyResult<Py<PyAny>> {
    let mut entries = Vec::new();
    if let Some(categories) = tables.get("category") {
        for cat in categories {
            let idx = cat
                .get("index")
                .and_then(Value::as_i64)
                .ok_or_else(|| PyValueError::new_err("category index missing"))?;
            let name = expect_str_field(cat, "name")?;
            entries.push((idx, name.to_string()));
        }
        entries.sort_by_key(|(idx, _)| *idx);
    } else {
        entries = default_colormap()
            .keys()
            .cloned()
            .enumerate()
            .map(|(i, k)| (i as i64, k))
            .collect();
    }

    let default_map = default_colormap();
    let dict = PyDict::new(py);
    for (_, name) in entries {
        if let Some((r, g, b)) = default_map.get(&name) {
            let tuple = PyTuple::new(py, [*r, *g, *b])?;
            dict.set_item(name, tuple)?;
        }
    }
    Ok(dict.unbind().into_any())
}

fn default_colormap() -> HashMap<String, (i64, i64, i64)> {
    let mut map = HashMap::new();
    map.insert("noise".to_string(), (0, 0, 0));
    map.insert("animal".to_string(), (70, 130, 180));
    map.insert("human.pedestrian.adult".to_string(), (0, 0, 230));
    map.insert("human.pedestrian.child".to_string(), (135, 206, 235));
    map.insert(
        "human.pedestrian.construction_worker".to_string(),
        (100, 149, 237),
    );
    map.insert(
        "human.pedestrian.personal_mobility".to_string(),
        (219, 112, 147),
    );
    map.insert("human.pedestrian.police_officer".to_string(), (0, 0, 128));
    map.insert("human.pedestrian.stroller".to_string(), (240, 128, 128));
    map.insert("human.pedestrian.wheelchair".to_string(), (138, 43, 226));
    map.insert("movable_object.barrier".to_string(), (112, 128, 144));
    map.insert("movable_object.debris".to_string(), (210, 105, 30));
    map.insert(
        "movable_object.pushable_pullable".to_string(),
        (105, 105, 105),
    );
    map.insert("movable_object.trafficcone".to_string(), (47, 79, 79));
    map.insert("static_object.bicycle_rack".to_string(), (188, 143, 143));
    map.insert("vehicle.bicycle".to_string(), (220, 20, 60));
    map.insert("vehicle.bus.bendy".to_string(), (255, 127, 80));
    map.insert("vehicle.bus.rigid".to_string(), (255, 69, 0));
    map.insert("vehicle.car".to_string(), (255, 158, 0));
    map.insert("vehicle.construction".to_string(), (233, 150, 70));
    map.insert("vehicle.emergency.ambulance".to_string(), (255, 83, 0));
    map.insert("vehicle.emergency.police".to_string(), (255, 215, 0));
    map.insert("vehicle.motorcycle".to_string(), (255, 61, 99));
    map.insert("vehicle.trailer".to_string(), (255, 140, 0));
    map.insert("vehicle.truck".to_string(), (255, 99, 71));
    map.insert("flat.driveable_surface".to_string(), (0, 207, 191));
    map.insert("flat.other".to_string(), (175, 0, 75));
    map.insert("flat.sidewalk".to_string(), (75, 0, 75));
    map.insert("flat.terrain".to_string(), (112, 180, 60));
    map.insert("static.manmade".to_string(), (222, 184, 135));
    map.insert("static.other".to_string(), (255, 228, 196));
    map.insert("static.vegetation".to_string(), (0, 175, 0));
    map.insert("vehicle.ego".to_string(), (255, 240, 245));
    map
}

fn expect_str_field<'a>(record: &'a Value, field: &str) -> PyResult<&'a str> {
    record
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| PyValueError::new_err(format!("Field {field} missing or not a string")))
}

fn expect_bool_field(record: &Value, field: &str) -> PyResult<bool> {
    record
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| PyValueError::new_err(format!("Field {field} missing or not a bool")))
}

fn insert_value(record: &mut Value, key: &str, value: Value) {
    if let Some(obj) = record.as_object_mut() {
        obj.insert(key.to_string(), value);
    }
}

fn ensure_object_mut(record: &mut Value) -> PyResult<&mut Map<String, Value>> {
    match record {
        Value::Object(map) => Ok(map),
        _ => Err(PyValueError::new_err(
            "Expected object while decorating sample data",
        )),
    }
}

fn ensure_table_exists(table_names: &[String], table_name: &str) -> PyResult<()> {
    if table_names.iter().any(|name| name == table_name) {
        Ok(())
    } else {
        Err(PyKeyError::new_err(format!("Table {table_name} not found")))
    }
}

fn json_to_py(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    Ok(match value {
        Value::Null => py.None(),
        Value::Bool(b) => Py::from(b.into_pyobject(py)?).into_any(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                PyInt::new(py, i).unbind().into_any()
            } else if let Some(u) = n.as_u64() {
                PyInt::new(py, u).unbind().into_any()
            } else if let Some(f) = n.as_f64() {
                PyFloat::new(py, f).unbind().into_any()
            } else {
                return Err(PyValueError::new_err("Unsupported number value"));
            }
        }
        Value::String(s) => PyString::new(py, s).unbind().into_any(),
        Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                let py_item = json_to_py(py, item)?;
                list.append(py_item)?;
            }
            list.into_any().unbind()
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                let py_val = json_to_py(py, v)?;
                dict.set_item(k, py_val)?;
            }
            dict.into_any().unbind()
        }
    })
}

fn py_to_json_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if obj.is_instance_of::<PyBool>() {
        let b: bool = obj.extract()?;
        return Ok(Value::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::Number(i.into()));
    }
    if let Ok(u) = obj.extract::<u64>() {
        return Ok(Value::Number(u.into()));
    }
    if let Ok(f) = obj.extract::<f64>() {
        let num = serde_json::Number::from_f64(f)
            .ok_or_else(|| PyValueError::new_err("Failed to convert float"))?;
        return Ok(Value::Number(num));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::String(s));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let mut out = Vec::new();
        for item in list {
            out.push(py_to_json_value(&item)?);
        }
        return Ok(Value::Array(out));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = Map::new();
        for (k, v) in dict {
            let key: String = k.extract::<String>()?;
            map.insert(key, py_to_json_value(&v)?);
        }
        return Ok(Value::Object(map));
    }
    Err(PyValueError::new_err("Unsupported query type"))
}

impl NuScenes {
    fn get_record(&self, table_name: &str, token: &str) -> PyResult<&Value> {
        let idx = self.getind(table_name, token)?;
        self.tables
            .get(table_name)
            .and_then(|v| v.get(idx))
            .ok_or_else(|| PyKeyError::new_err(format!("Token {token} not found in {table_name}")))
    }
}

fn numpy_array(py: Python<'_>, values: &[f64]) -> PyResult<Py<PyAny>> {
    let np = PyModule::import(py, "numpy")?;
    let arr = np.getattr("array")?.call1((values.to_vec(),))?;
    Ok(arr.unbind().into_any())
}

fn vec3_from_field(record: &Value, field: &str) -> PyResult<Vec<f64>> {
    let arr = record
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| PyValueError::new_err(format!("Field {field} missing or not an array")))?;
    if arr.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "Field {field} expected 3 elements"
        )));
    }
    let mut out = Vec::with_capacity(3);
    for v in arr {
        out.push(
            v.as_f64()
                .or_else(|| v.as_i64().map(|i| i as f64))
                .ok_or_else(|| PyValueError::new_err("Expected numeric translation component"))?,
        );
    }
    Ok(out)
}

fn sample_timestamp(nusc: &NuScenes, ann_record: &Value) -> PyResult<f64> {
    let sample_token = expect_str_field(ann_record, "sample_token")?;
    let sample = nusc.get_record("sample", sample_token)?;
    let ts = sample
        .get("timestamp")
        .and_then(Value::as_i64)
        .ok_or_else(|| PyValueError::new_err("timestamp missing"))?;
    Ok(ts as f64)
}
