use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const EXPECTED_LICENSE: &str = "MIT OR Apache-2.0";
const EXPECTED_HORIZON_PIN: &str = "f00125f61f794aaa3d4e950d301f353b35b6ee3d";
const EXPECTED_MIXINS_PIN: &str = "e59da9094f0c2fdda6c627fe9eff22dec6a906f0";
const EXPECTED_PACK_GAME_VERSION: &str = "26.3";

const SKIP_DIRS: &[&str] = &[
    ".git",
    ".horizon",
    ".horizon-cache",
    ".local",
    "dist",
    "port_work",
    "target",
];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn strict_full_audit_enabled() -> bool {
    matches!(
        env::var("STRICT_FULL_AUDIT").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn collect_files_with_extensions(root: &Path, exts: &[&str]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir)
            .unwrap_or_else(|err| panic!("failed to read directory {}: {err}", dir.display()));

        for entry in entries {
            let entry = entry.expect("failed to read directory entry");
            let path = entry.path();
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();

            if path.is_dir() {
                if SKIP_DIRS.contains(&name.as_ref()) {
                    continue;
                }
                stack.push(path);
            } else if path.is_file()
                && let Some(ext) = path.extension().and_then(|s| s.to_str())
                && exts.iter().any(|wanted| ext.eq_ignore_ascii_case(wanted))
            {
                out.push(path);
            }
        }
    }

    out.sort();
    out
}

fn strip_toml_comment(line: &str) -> &str {
    let mut in_double_quote = false;
    let mut in_single_quote = false;
    let mut escaped = false;

    for (idx, ch) in line.char_indices() {
        if in_double_quote {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_double_quote = false;
            }
            continue;
        }
        if in_single_quote {
            if ch == '\'' {
                in_single_quote = false;
            }
            continue;
        }
        match ch {
            '"' => in_double_quote = true,
            '\'' => in_single_quote = true,
            '#' => return &line[..idx],
            _ => {}
        }
    }

    line
}

fn unquote_toml_string(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if let Some(inner) = trimmed.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        let mut out = String::with_capacity(inner.len());
        let mut chars = inner.chars();
        while let Some(ch) = chars.next() {
            if ch == '\\' {
                match chars.next() {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('n') => out.push('\n'),
                    Some('r') => out.push('\r'),
                    Some('t') => out.push('\t'),
                    Some(other) => {
                        out.push('\\');
                        out.push(other);
                    }
                    None => return None,
                }
            } else {
                out.push(ch);
            }
        }
        Some(out)
    } else {
        trimmed
            .strip_prefix('\'')
            .and_then(|s| s.strip_suffix('\''))
            .map(ToString::to_string)
    }
}

#[derive(Debug, Default)]
struct ParsedToml {
    tables: BTreeMap<String, BTreeMap<String, String>>,
    array_tables: BTreeMap<String, Vec<BTreeMap<String, String>>>,
}

impl ParsedToml {
    fn parse(content: &str) -> Self {
        let mut parsed = Self::default();
        let mut current_table = String::new();
        let mut current_array_table: Option<(String, usize)> = None;

        for raw_line in content.lines() {
            let line = strip_toml_comment(raw_line).trim();
            if line.is_empty() {
                continue;
            }

            if let Some(inner) = line.strip_prefix("[[").and_then(|s| s.strip_suffix("]]")) {
                let name = inner.trim().to_string();
                if !name.is_empty() {
                    let list = parsed.array_tables.entry(name.clone()).or_default();
                    list.push(BTreeMap::new());
                    current_array_table = Some((name, list.len() - 1));
                    current_table.clear();
                    continue;
                }
            }

            if let Some(inner) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                let name = inner.trim().to_string();
                if !name.is_empty() {
                    parsed.tables.entry(name.clone()).or_default();
                    current_table = name;
                    current_array_table = None;
                    continue;
                }
            }

            if let Some((k, v)) = split_key_value(line) {
                if let Some((array_name, idx)) = &current_array_table {
                    if let Some(list) = parsed.array_tables.get_mut(array_name)
                        && let Some(entry) = list.get_mut(*idx)
                    {
                        entry.insert(k.to_string(), v.to_string());
                    }
                } else {
                    parsed
                        .tables
                        .entry(current_table.clone())
                        .or_default()
                        .insert(k.to_string(), v.to_string());
                }
            }
        }

        parsed
    }

    fn get_str(&self, table: &str, key: &str) -> Option<String> {
        let raw = self.tables.get(table)?.get(key)?;
        unquote_toml_string(raw)
    }

    fn get_raw(&self, table: &str, key: &str) -> Option<&str> {
        self.tables.get(table)?.get(key).map(String::as_str)
    }

    fn find_str_anywhere(&self, key: &str) -> Vec<(String, String)> {
        let mut matches = Vec::new();
        let dotted_suffix = format!(".{key}");
        for (table_name, entries) in &self.tables {
            for (k, raw) in entries {
                if (k == key || k.ends_with(&dotted_suffix))
                    && let Some(val) = unquote_toml_string(raw)
                {
                    matches.push((table_name.clone(), val));
                } else if let Some(inner) = raw.strip_prefix('{').and_then(|s| s.strip_suffix('}'))
                {
                    for part in inner.split(',') {
                        if let Some((ik, iv)) = split_key_value(part.trim())
                            && (ik == key || ik.ends_with(&dotted_suffix))
                            && let Some(val) = unquote_toml_string(iv)
                        {
                            matches.push((format!("{table_name}.{k}"), val));
                        }
                    }
                }
            }
        }
        matches
    }
}

fn split_key_value(line: &str) -> Option<(&str, &str)> {
    let mut in_double = false;
    let mut in_single = false;
    let mut escaped = false;

    for (idx, ch) in line.char_indices() {
        if in_double {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_double = false;
            }
            continue;
        }
        if in_single {
            if ch == '\'' {
                in_single = false;
            }
            continue;
        }
        match ch {
            '"' => in_double = true,
            '\'' => in_single = true,
            '=' => {
                let key = line[..idx].trim();
                let val = line[idx + 1..].trim();
                if !key.is_empty() && !val.is_empty() {
                    return Some((key, val));
                }
                return None;
            }
            _ => {}
        }
    }
    None
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn non_comment_toml_contains_parent_escape(content: &str) -> Vec<(usize, String)> {
    let mut violations = Vec::new();
    for (idx, raw_line) in content.lines().enumerate() {
        let stripped = strip_toml_comment(raw_line).trim();
        if stripped.is_empty() {
            continue;
        }
        if stripped.contains("../")
            || stripped.contains("..\\")
            || stripped.contains("\"..\"")
            || stripped.contains("'..'")
        {
            violations.push((idx + 1, stripped.to_string()));
        }
    }
    violations
}

fn non_comment_toml_contains_workspace_inherit(content: &str) -> Vec<(usize, String)> {
    let mut violations = Vec::new();
    for (idx, raw_line) in content.lines().enumerate() {
        let stripped = strip_toml_comment(raw_line).trim();
        if stripped.is_empty() {
            continue;
        }
        let compact: String = stripped.chars().filter(|c| !c.is_whitespace()).collect();
        if compact.contains(".workspace=true") || compact.contains("workspace=true") {
            violations.push((idx + 1, stripped.to_string()));
        }
    }
    violations
}

fn forbidden_source_tokens() -> Vec<String> {
    vec![
        ["poly", "form"].concat(),
        ["ryan", "hcode"].concat(),
        ["sa", "ble"].concat(),
    ]
}

fn forbidden_unsafe_attr_compact() -> String {
    ["#![forbid(", "unsafe_code", ")]"].concat()
}

fn strip_rust_line_comments_for_attr_check(content: &str) -> String {
    let mut out = String::with_capacity(content.len());
    for line in content.lines() {
        let code = line.split_once("//").map_or(line, |(before, _)| before);
        for ch in code.chars() {
            if !ch.is_whitespace() {
                out.push(ch);
            }
        }
    }
    out
}

#[test]
fn test_licenses_exist_and_valid() {
    let root = repo_root();

    let mit_path = root.join("LICENSE-MIT");
    let mit = fs::read_to_string(&mit_path).expect("LICENSE-MIT must exist at repository root");
    assert!(
        mit.contains("MIT License"),
        "LICENSE-MIT must contain the MIT License header"
    );
    assert!(
        mit.contains("Permission is hereby granted, free of charge"),
        "LICENSE-MIT must contain standard MIT grant clause"
    );
    assert!(
        mit.contains("THE SOFTWARE IS PROVIDED \"AS IS\""),
        "LICENSE-MIT must contain standard MIT warranty disclaimer"
    );

    let apache_path = root.join("LICENSE-APACHE");
    let apache =
        fs::read_to_string(&apache_path).expect("LICENSE-APACHE must exist at repository root");
    assert!(
        apache.contains("Apache License"),
        "LICENSE-APACHE must contain Apache License title"
    );
    assert!(
        apache.contains("Version 2.0, January 2004"),
        "LICENSE-APACHE must specify Version 2.0, January 2004"
    );
    assert!(
        apache.contains("http://www.apache.org/licenses/"),
        "LICENSE-APACHE must contain canonical Apache license URL"
    );
    assert!(
        apache.contains("TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION"),
        "LICENSE-APACHE must contain full Apache 2.0 terms"
    );

    let root_cargo =
        fs::read_to_string(root.join("Cargo.toml")).expect("root Cargo.toml must exist");
    let parsed_root = ParsedToml::parse(&root_cargo);
    assert_eq!(
        parsed_root.get_str("package", "license").as_deref(),
        Some(EXPECTED_LICENSE),
        "root Cargo.toml must declare license = \"{EXPECTED_LICENSE}\""
    );
}

#[test]
fn test_no_forbidden_terms_or_unsafe_forbid_in_sources() {
    let root = repo_root();
    let files = collect_files_with_extensions(&root, &["rs", "toml"]);
    assert!(
        !files.is_empty(),
        "expected at least one .rs or .toml file in workspace"
    );

    let forbidden_tokens = forbidden_source_tokens();
    let forbidden_attr = forbidden_unsafe_attr_compact();
    let mut violations = Vec::new();

    for path in files {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(&path)
            .display()
            .to_string();
        let content = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
        let lower = content.to_ascii_lowercase();

        for token in &forbidden_tokens {
            if lower.contains(token.as_str()) {
                violations.push(format!("{rel}: contains forbidden token '{token}'"));
            }
        }

        if path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("rs"))
        {
            let compact = strip_rust_line_comments_for_attr_check(&content);
            if compact.contains(&forbidden_attr) {
                violations.push(format!(
                    "{rel}: contains forbidden attribute {forbidden_attr}"
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "red-team source compliance violations detected:\n{}",
        violations.join("\n")
    );
}

#[test]
fn test_horizon_physics_core_manifest_compliance() {
    let root = repo_root();
    let core_dir = root.join("mods/horizon_physics/crates/horizon_physics_core");
    let core_cargo_path = core_dir.join("Cargo.toml");

    if !core_cargo_path.exists() {
        assert!(
            !strict_full_audit_enabled(),
            "STRICT_FULL_AUDIT=1 requires {} to exist",
            core_cargo_path.display()
        );
        return;
    }

    let build_rs = core_dir.join("build.rs");
    assert!(
        !build_rs.exists(),
        "horizon_physics_core must not contain build.rs ({})",
        build_rs.display()
    );

    let content = fs::read_to_string(&core_cargo_path).expect("failed to read core Cargo.toml");
    let parsed = ParsedToml::parse(&content);

    assert_eq!(
        parsed.get_str("package", "license").as_deref(),
        Some(EXPECTED_LICENSE),
        "horizon_physics_core must declare license = \"{EXPECTED_LICENSE}\""
    );
    assert!(
        parsed.get_raw("package", "build").is_none(),
        "horizon_physics_core must not declare a custom build script"
    );

    let deps = parsed.tables.get("dependencies");
    assert!(
        deps.is_none_or(BTreeMap::is_empty),
        "horizon_physics_core must have zero [dependencies], found: {deps:?}"
    );

    if let Some(root_entries) = parsed.tables.get("") {
        for key in root_entries.keys() {
            assert!(
                !key.starts_with("dependencies."),
                "horizon_physics_core must not declare dotted dependency `{key}`"
            );
            assert!(
                !key.starts_with("features."),
                "horizon_physics_core must not declare dotted feature `{key}`"
            );
        }
    }

    for (table_name, entries) in &parsed.tables {
        if (table_name.ends_with(".dependencies") || table_name.starts_with("dependencies."))
            && !entries.is_empty()
        {
            panic!(
                "horizon_physics_core must not declare dependencies in [{table_name}]: {entries:?}"
            );
        }
    }

    let features = parsed.tables.get("features");
    assert!(
        features.is_none_or(BTreeMap::is_empty),
        "horizon_physics_core must have zero [features], found: {features:?}"
    );

    let ws_violations = non_comment_toml_contains_workspace_inherit(&content);
    assert!(
        ws_violations.is_empty(),
        "horizon_physics_core must not use `.workspace = true`: {ws_violations:?}"
    );

    let escape_violations = non_comment_toml_contains_parent_escape(&content);
    assert!(
        escape_violations.is_empty(),
        "horizon_physics_core Cargo.toml must not contain `..` path escapes: {escape_violations:?}"
    );
}

#[test]
fn test_horizon_physics_mod_manifest_and_sdk_pins() {
    let root = repo_root();
    let mod_dir = root.join("mods/horizon_physics");
    let cargo_path = mod_dir.join("Cargo.toml");
    let mod_toml_path = mod_dir.join("horizon-mod.toml");

    if !mod_dir.exists() {
        assert!(
            !strict_full_audit_enabled(),
            "STRICT_FULL_AUDIT=1 requires {} to exist",
            mod_dir.display()
        );
        return;
    }

    assert!(
        cargo_path.exists(),
        "mods/horizon_physics/Cargo.toml must exist when mods/horizon_physics is present"
    );
    assert!(
        mod_toml_path.exists(),
        "mods/horizon_physics/horizon-mod.toml must exist when mods/horizon_physics is present"
    );

    let build_rs = mod_dir.join("build.rs");
    assert!(
        !build_rs.exists(),
        "mods/horizon_physics must not contain build.rs ({})",
        build_rs.display()
    );

    let cargo_text = fs::read_to_string(&cargo_path).expect("failed to read mod Cargo.toml");
    let mod_text = fs::read_to_string(&mod_toml_path).expect("failed to read horizon-mod.toml");

    let parsed_cargo = ParsedToml::parse(&cargo_text);
    let parsed_mod = ParsedToml::parse(&mod_text);

    assert!(
        parsed_cargo.get_raw("package", "build").is_none(),
        "mods/horizon_physics/Cargo.toml must not declare a custom build script"
    );

    let features = parsed_cargo.tables.get("features");
    assert!(
        features.is_none_or(BTreeMap::is_empty),
        "mods/horizon_physics/Cargo.toml must not define custom [features]: {features:?}"
    );

    let cargo_escapes = non_comment_toml_contains_parent_escape(&cargo_text);
    assert!(
        cargo_escapes.is_empty(),
        "mods/horizon_physics/Cargo.toml must not escape mod root via `..`: {cargo_escapes:?}"
    );

    let mod_escapes = non_comment_toml_contains_parent_escape(&mod_text);
    assert!(
        mod_escapes.is_empty(),
        "mods/horizon_physics/horizon-mod.toml must not escape mod root via `..`: {mod_escapes:?}"
    );

    let mut horizon_pins = parsed_mod.find_str_anywhere("horizon");
    if horizon_pins.is_empty() {
        horizon_pins = parsed_cargo.find_str_anywhere("horizon");
    }
    assert!(
        horizon_pins
            .iter()
            .any(|(_, val)| val == EXPECTED_HORIZON_PIN),
        "mods/horizon_physics must pin horizon = \"{EXPECTED_HORIZON_PIN}\", found: {horizon_pins:?}"
    );

    let mut mixins_pins = parsed_mod.find_str_anywhere("mixins");
    if mixins_pins.is_empty() {
        mixins_pins = parsed_cargo.find_str_anywhere("mixins");
    }
    assert!(
        mixins_pins
            .iter()
            .any(|(_, val)| val == EXPECTED_MIXINS_PIN),
        "mods/horizon_physics must pin mixins = \"{EXPECTED_MIXINS_PIN}\", found: {mixins_pins:?}"
    );
}

#[test]
fn test_modpack_manifest_compliance() {
    let root = repo_root();
    let manifest_path = root.join("pack/horizon-modpack.toml");
    let content = fs::read_to_string(&manifest_path)
        .expect("pack/horizon-modpack.toml must exist in repository");
    let parsed = ParsedToml::parse(&content);

    assert_eq!(
        parsed.get_raw("", "format"),
        Some("1"),
        "pack/horizon-modpack.toml must declare format = 1"
    );
    assert_eq!(
        parsed.get_str("", "game").as_deref(),
        Some(EXPECTED_PACK_GAME_VERSION),
        "pack/horizon-modpack.toml must declare game = \"{EXPECTED_PACK_GAME_VERSION}\""
    );

    for required in ["id", "name", "version"] {
        let value = parsed
            .get_str("", required)
            .unwrap_or_else(|| panic!("pack/horizon-modpack.toml missing string key `{required}`"));
        assert!(
            !value.trim().is_empty(),
            "pack/horizon-modpack.toml key `{required}` must not be empty"
        );
    }

    if let Some(mod_entries) = parsed.array_tables.get("mods") {
        for (idx, entry) in mod_entries.iter().enumerate() {
            let id = entry
                .get("id")
                .and_then(|v| unquote_toml_string(v))
                .unwrap_or_else(|| panic!("[[mods]][{idx}] missing quoted `id`"));
            assert!(
                !id.trim().is_empty(),
                "[[mods]][{idx}].id must not be empty"
            );

            let version = entry
                .get("version")
                .and_then(|v| unquote_toml_string(v))
                .unwrap_or_else(|| panic!("[[mods]][{idx}] missing quoted `version`"));
            assert!(
                !version.trim().is_empty(),
                "[[mods]][{idx}].version must not be empty"
            );

            let sha256 = entry
                .get("sha256")
                .and_then(|v| unquote_toml_string(v))
                .unwrap_or_else(|| panic!("[[mods]][{idx}] missing quoted `sha256`"));
            assert!(
                is_lower_hex_sha256(&sha256),
                "[[mods]][{idx}].sha256 must be 64 lowercase hex chars, got `{sha256}`"
            );

            if let Some(on_val) = entry.get("on") {
                assert!(
                    on_val == "true" || on_val == "false",
                    "[[mods]][{idx}].on must be boolean, got `{on_val}`"
                );
            }
        }
    }
}

#[test]
fn test_strict_full_audit_presence_when_enabled() {
    if !strict_full_audit_enabled() {
        return;
    }

    let root = repo_root();
    let required_paths = [
        "Cargo.toml",
        "LICENSE-MIT",
        "LICENSE-APACHE",
        "README.md",
        "docs/ARCHITECTURE.md",
        "docs/THIRD_PARTY_MOD_API.md",
        "docs/CLEAN_ROOM_COMPLIANCE.md",
        "pack/horizon-modpack.toml",
        "scripts/build_pack.ps1",
        "scripts/verify_redteam.ps1",
        "mods/workbase_core/horizon-mod.toml",
        "mods/horizon_physics/Cargo.toml",
        "mods/horizon_physics/horizon-mod.toml",
        "mods/horizon_physics/crates/horizon_physics_core/Cargo.toml",
    ];

    let missing: Vec<&str> = required_paths
        .into_iter()
        .filter(|rel| !root.join(rel).exists())
        .collect();

    assert!(
        missing.is_empty(),
        "STRICT_FULL_AUDIT=1 failed; missing required repository files:\n{}",
        missing.join("\n")
    );
}

#[test]
fn test_toml_parser_sanity() {
    let sample = r#"
format = 1 # comment with "quotes"
game = "26.3"

[sdk]
horizon = "f00125f61f794aaa3d4e950d301f353b35b6ee3d"
mixins = "e59da9094f0c2fdda6c627fe9eff22dec6a906f0"

[[mods]]
id = "horizon_physics"
version = "0.1.0"
sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
on = true
"#;

    let parsed = ParsedToml::parse(sample);
    assert_eq!(parsed.get_raw("", "format"), Some("1"));
    assert_eq!(parsed.get_str("", "game").as_deref(), Some("26.3"));
    assert_eq!(
        parsed.get_str("sdk", "horizon").as_deref(),
        Some(EXPECTED_HORIZON_PIN)
    );
    assert_eq!(
        parsed.get_str("sdk", "mixins").as_deref(),
        Some(EXPECTED_MIXINS_PIN)
    );
    assert_eq!(parsed.array_tables.get("mods").map(Vec::len), Some(1));
}
