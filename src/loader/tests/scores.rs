// Included as a child of loader::tests; diagnostic output is explicitly requested.
use super::*;
use tinyjson::JsonValue;

fn object(fields: impl IntoIterator<Item = (&'static str, JsonValue)>) -> JsonValue {
    fields
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect::<std::collections::HashMap<_, _>>()
        .into()
}

fn ids(spec: &JsonValue, key: &str) -> Vec<i32> {
    spec[key]
        .get::<Vec<JsonValue>>()
        .unwrap()
        .iter()
        .map(|v| {
            let x = *v.get::<f64>().unwrap();
            assert!(x.is_finite() && x.fract() == 0.0 && (0.0..248070.0).contains(&x));
            x as i32
        })
        .collect()
}

#[test]
#[ignore = "loads real BF16 weights and dumps forced-prefix scores for an external comparison"]
fn real_qwen36_same_prefix_scores() {
    let input = std::env::var("QS3_SCORE_INPUT").expect("QS3_SCORE_INPUT JSON path");
    let output = std::path::PathBuf::from(
        std::env::var("QS3_SCORE_OUTPUT").expect("QS3_SCORE_OUTPUT directory"),
    );
    let spec: JsonValue = std::fs::read_to_string(input).unwrap().parse().unwrap();
    let prompt = ids(&spec, "prompt_ids");
    let forced = ids(&spec, "forced_ids");
    let captures = ids(&spec, "capture_steps");
    assert!(!prompt.is_empty() && !forced.is_empty());
    assert!(captures.iter().all(|&x| x as usize <= forced.len()));
    std::fs::create_dir_all(&output).unwrap();
    let model_dir = require_real_qwen36_model_dir();
    assert_eq!(
        model_dir.file_name().unwrap().to_str().unwrap(),
        spec["model_revision"].get::<String>().unwrap()
    );
    let plan = QwenBf16LoadPlan::read(&model_dir).unwrap();
    let backend = ManagedUmaBackend::new(cuda_device_from_env()).unwrap();
    let loaded = execute_qwen36_bf16_load_plan(&plan, backend, ptr::null_mut()).unwrap();
    let (config, weights) = loaded
        .into_qwen_model(
            ptr::null_mut(),
            u32::try_from(prompt.len() + forced.len() + 1).unwrap(),
        )
        .unwrap();
    assert_eq!(
        config.gdn_recurrent_precision,
        crate::model::GdnRecurrentPrecision::F32
    );
    let mut runner = crate::model::ModelRunner::new(config, weights).unwrap();
    const REQUEST: u64 = 0x53434f5245;
    runner
        .run(crate::model::QwenRequest {
            request_id: REQUEST,
            tokens: &prompt,
            max_new_tokens: 0,
        })
        .unwrap();
    let mut records = Vec::new();
    let mut expected_live = prompt.clone();
    for step in 0..=forced.len() {
        assert_eq!(runner.live_tokens(), expected_live);
        let raw = runner.last_logits_row_for_test().unwrap();
        assert_eq!(raw.len(), 248320);
        assert!(raw.iter().all(|x| x.is_finite()));
        let mut order: Vec<usize> = (0..raw.len()).collect();
        let compare = |&a: &usize, &b: &usize| raw[b].total_cmp(&raw[a]).then(a.cmp(&b));
        order.select_nth_unstable_by(20, compare);
        order.truncate(20);
        order.sort_unstable_by(compare);
        let max = f64::from(raw[order[0]]);
        let lse = max
            + raw
                .iter()
                .map(|&x| (f64::from(x) - max).exp())
                .sum::<f64>()
                .ln();
        let mut row = object([
            ("step", (step as f64).into()),
            ("argmax", (order[0] as f64).into()),
            (
                "top_ids",
                order
                    .iter()
                    .map(|&i| JsonValue::Number(i as f64))
                    .collect::<Vec<_>>()
                    .into(),
            ),
            (
                "top_values",
                order
                    .iter()
                    .map(|&i| JsonValue::Number(f64::from(raw[i])))
                    .collect::<Vec<_>>()
                    .into(),
            ),
            ("logsumexp", lse.into()),
            ("logits_dtype", "float32".to_owned().into()),
        ]);
        let fields = row
            .get_mut::<std::collections::HashMap<String, JsonValue>>()
            .unwrap();
        if step < forced.len() {
            let token = forced[step] as usize;
            fields.insert("forced_id".into(), (token as f64).into());
            fields.insert("forced_logit".into(), f64::from(raw[token]).into());
            fields.insert("forced_nll".into(), (lse - f64::from(raw[token])).into());
        }
        if captures.contains(&(step as i32)) {
            let filename = format!("logits-{step:04}.f32");
            let bytes: Vec<u8> = raw.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(output.join(&filename), bytes).unwrap();
            fields.insert("file".into(), filename.into());
        }
        records.push(row);
        if step < forced.len() {
            runner.decode_forced_token_for_test(forced[step]).unwrap();
            expected_live.push(forced[step]);
        }
    }
    let result = object([
        ("input", spec),
        ("router_logits_dtype", "bf16".to_owned().into()),
        ("records", records.into()),
        (
            "protocol",
            "one prefill then forced token-by-token decode; raw logits before sampling"
                .to_owned()
                .into(),
        ),
        (
            "gdn_recurrent_state_dtype",
            config.gdn_recurrent_precision.as_str().to_owned().into(),
        ),
        (
            "moe_kernel",
            config.moe_bf16_kernel.as_str().to_owned().into(),
        ),
    ]);
    std::fs::write(output.join("scores.json"), result.stringify().unwrap()).unwrap();
}
