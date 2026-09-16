use super::*;
use crate::constants::mlp::HAS_EXPERTS;
use std::collections::HashMap;
use tinyjson::JsonValue;

fn checkpoint_json(model: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(model)
        .join("model.json");
    std::fs::read_to_string(path).unwrap()
}

pub(super) fn selected_config() -> HashMap<String, JsonValue> {
    parse_json_object(&checkpoint_json(crate::constants::engine::MODEL)).unwrap()
}

pub(super) fn text(root: &mut HashMap<String, JsonValue>) -> &mut HashMap<String, JsonValue> {
    root.get_mut("text_config").unwrap().get_mut().unwrap()
}

#[test]
fn pinned_checkpoint_matches_selected_build() {
    let config = Qwen36TextConfig::from_config_object(&selected_config()).unwrap();
    config.validate_compiled_model().unwrap();
}

#[test]
fn selected_mlp_fields_and_explicit_head_dim_are_required() {
    let fields: &[&str] = if HAS_EXPERTS {
        &[
            "head_dim",
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
        ]
    } else {
        &["head_dim", "intermediate_size"]
    };
    for &field in fields {
        let mut root = selected_config();
        text(&mut root).remove(field);
        assert!(
            Qwen36TextConfig::from_config_object(&root).is_err(),
            "{field}"
        );
    }
    let mut root = selected_config();
    text(&mut root).insert("num_attention_heads".into(), 0.0.into());
    assert!(Qwen36TextConfig::from_config_object(&root).is_err());
}

#[test]
fn mlp_width_and_expert_fields_cannot_change_the_selected_architecture() {
    let mut root = selected_config();
    let width = if HAS_EXPERTS {
        "moe_intermediate_size"
    } else {
        "intermediate_size"
    };
    let current = *text(&mut root)[width].get::<f64>().unwrap();
    text(&mut root).insert(width.into(), (current + 8.0).into());
    assert!(matches!(
        Qwen36TextConfig::from_config_object(&root),
        Err(WeightLoadError::InvalidConfig(_))
    ));

    if HAS_EXPERTS {
        let mut root = selected_config();
        let width = text(&mut root)["moe_intermediate_size"].clone();
        text(&mut root).insert("intermediate_size".into(), width);
        Qwen36TextConfig::from_config_object(&root).unwrap();
        let wrong = *text(&mut root)["moe_intermediate_size"]
            .get::<f64>()
            .unwrap()
            + 8.0;
        text(&mut root).insert("intermediate_size".into(), wrong.into());
        assert!(Qwen36TextConfig::from_config_object(&root).is_err());
    } else {
        for field in [
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
        ] {
            for value in [JsonValue::Null, 0.0.into(), 256.0.into()] {
                let mut root = selected_config();
                text(&mut root).insert(field.into(), value);
                assert!(
                    matches!(
                        Qwen36TextConfig::from_config_object(&root),
                        Err(WeightLoadError::InvalidConfig(_))
                    ),
                    "{field}"
                );
            }
        }
    }
}

#[test]
fn other_checkpoint_is_rejected_before_reading_weight_index() {
    let directory =
        std::env::temp_dir().join(format!("qs3-other-checkpoint-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(
        directory.join("config.json"),
        checkpoint_json(if HAS_EXPERTS {
            "qwen3.6-27b"
        } else {
            "qwen3.6-35b-a3b"
        }),
    )
    .unwrap();
    // Config rejection must win over the missing index file.
    let result = QwenBf16LoadPlan::read(&directory);
    std::fs::remove_dir_all(directory).unwrap();
    assert!(matches!(
        result,
        Err(WeightLoadError::InvalidConfig(_) | WeightLoadError::Json(_))
    ));
}
