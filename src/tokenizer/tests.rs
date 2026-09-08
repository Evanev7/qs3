use crate::test_assets::real_qwen36_model_dir;
use std::collections::HashMap;
use std::time::Instant;

use tinyjson::JsonValue;
use unicode_normalization::UnicodeNormalization;

use super::bpe::{BpeDefinition, BpeModel};
use super::format::TokenizerDefinition;
use super::pretokenize::{byte_pieces, classification_mask, text_pieces};
use super::{QwenTokenizer, TokenizerError};

#[test]
fn qwen_split_matches_words_punctuation_and_leading_spaces() {
    assert_eq!(byte_pieces("Hello, world!"), ["Hello", ",", "Ġworld", "!"]);
    assert_eq!(byte_pieces(" hello"), ["Ġhello"]);
    assert_eq!(byte_pieces("   hello"), ["ĠĠ", "Ġhello"]);
    assert_eq!(byte_pieces("Hello  world"), ["Hello", "Ġ", "Ġworld"]);
}

#[test]
fn qwen_split_matches_line_and_trailing_whitespace_rules() {
    assert_eq!(byte_pieces("hello\nworld"), ["hello", "Ċ", "world"]);
    assert_eq!(
        byte_pieces("hello \n  world"),
        ["hello", "ĠĊ", "Ġ", "Ġworld"]
    );
    assert_eq!(
        byte_pieces("\n\n  \ntext\t\tend  "),
        ["ĊĊĠĠĊ", "text", "ĉ", "ĉend", "ĠĠ"]
    );
    assert_eq!(byte_pieces("          a"), ["ĠĠĠĠĠĠĠĠĠ", "Ġa"]);
}

#[test]
fn qwen_split_preserves_ordered_contraction_alternatives() {
    assert_eq!(
        byte_pieces("'s 'S 're 'RE 'll 'D"),
        [
            "'s", "Ġ'", "S", "Ġ'", "re", "Ġ'", "RE", "Ġ'", "ll", "Ġ'", "D"
        ]
    );
    assert_eq!(
        text_pieces("rock’n’roll isn't I'M"),
        ["rock", "’n", "’roll", " isn", "'t", " I", "'M"]
    );
    assert_eq!(text_pieces("'ſ 'S"), ["'ſ", " '", "S"]);
    assert_eq!(text_pieces("'ſx"), ["'ſ", "x"]);
}

#[test]
fn qwen_split_uses_unicode_general_categories() {
    assert_eq!(text_pieces("مَرْحَبًا दुनिया"), ["مَرْحَبًا", " दुनिया"]);
    assert_eq!(text_pieces("123٤٥Ⅻ"), ["1", "2", "3", "٤", "٥", "Ⅻ"]);
    assert_eq!(text_pieces("你好，世界！"), ["你好", "，世界", "！"]);
    assert_eq!(text_pieces("👩‍💻👍🏽"), ["👩‍💻👍🏽"]);
}

#[test]
fn qwen_split_handles_unicode_whitespace_and_crlf_suffixes() {
    assert_eq!(
        text_pieces("foo... \r\nbar"),
        ["foo", "...", " \r\n", "bar"]
    );
    assert_eq!(
        text_pieces("a\u{a0}\u{2003}\u{85}b"),
        ["a", "\u{a0}\u{2003}", "\u{85}b"]
    );
}

#[test]
fn unicode_classifier_matches_exhaustive_oniguruma_oracle() {
    let mut counts = [0_u32; 4];
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for codepoint in 0..=0x10ffff {
        let Some(ch) = char::from_u32(codepoint) else {
            continue;
        };
        let mask = classification_mask(ch);
        for (index, bit) in [1_u8, 2, 4, 8].into_iter().enumerate() {
            counts[index] += u32::from(mask & bit != 0);
        }
        hash ^= u64::from(mask);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }

    // Generated once through tokenizers 0.22.2's Oniguruma-backed Regex for
    // every Unicode scalar value, using \p{L}, \p{M}, \p{N}, and \s.
    assert_eq!(counts, [141_028, 2_501, 1_911, 25]);
    assert_eq!(hash, 0x1867_f23d_06b9_e3eb);
}

#[test]
fn nfc_matches_exhaustive_hugging_face_oracle() {
    let mut source = String::with_capacity(5 << 20);
    let mut first = true;
    for codepoint in 1..=0x10ffff {
        let Some(ch) = char::from_u32(codepoint) else {
            continue;
        };
        if !first {
            source.push('\0');
        }
        source.push(ch);
        first = false;
    }
    let normalized: String = source.nfc().collect();

    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut changed = 0_u32;
    hash_bytes_with_separator("\0".as_bytes(), &mut hash);
    let mut scalars = (1..=0x10ffff).filter_map(char::from_u32);
    for piece in normalized.split('\0') {
        let original = scalars.next().unwrap();
        if !piece.starts_with(original) || piece.len() != original.len_utf8() {
            changed += 1;
        }
        hash_bytes_with_separator(piece.as_bytes(), &mut hash);
    }
    assert_eq!(scalars.next(), None);

    // Generated once through the pinned tokenizer's NFC normalizer for every
    // Unicode scalar value, with U+0000 separating independent inputs.
    assert_eq!(changed, 1_120);
    assert_eq!(hash, 0x0420_af24_0761_5d53);
}

fn hash_bytes_with_separator(bytes: &[u8], hash: &mut u64) {
    for &byte in bytes {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    *hash ^= 0xff;
    *hash = hash.wrapping_mul(0x100_0000_01b3);
}

fn synthetic_bpe(merges: &[(&str, &str)]) -> BpeModel {
    let mut vocab = HashMap::new();
    for byte in 0_u16..=255 {
        vocab.insert(
            super::pretokenize::byte_char(byte as u8).to_string(),
            u32::from(byte),
        );
    }
    for (left, right) in merges {
        let token = format!("{left}{right}");
        if !vocab.contains_key(&token) {
            vocab.insert(token, vocab.len() as u32);
        }
    }
    BpeModel::new(BpeDefinition::new(
        vocab,
        merges
            .iter()
            .map(|(left, right)| (String::from(*left), String::from(*right)))
            .collect(),
    ))
    .unwrap()
}

fn encode_one_piece(model: &BpeModel, piece: &str) -> Vec<i32> {
    let mut ids = Vec::new();
    model.encode_piece(piece, &mut ids);
    ids
}

#[test]
fn bpe_merges_by_rank_and_then_left_position() {
    let model = synthetic_bpe(&[("b", "c"), ("a", "b"), ("a", "bc")]);
    assert_eq!(
        encode_one_piece(&model, "abc"),
        [model.token_id("abc").unwrap()]
    );

    let model = synthetic_bpe(&[("a", "a"), ("aa", "a")]);
    assert_eq!(
        encode_one_piece(&model, "aaa"),
        [model.token_id("aaa").unwrap()]
    );
}

#[test]
fn bpe_requeues_new_pairs_and_discards_stale_candidates() {
    let model = synthetic_bpe(&[("x", "ab"), ("a", "b")]);
    assert_eq!(
        encode_one_piece(&model, "xab"),
        [model.token_id("xab").unwrap()]
    );

    let model = synthetic_bpe(&[("a", "b"), ("b", "c"), ("ab", "c")]);
    assert_eq!(
        encode_one_piece(&model, "abc"),
        [model.token_id("abc").unwrap()]
    );
}

#[test]
fn bpe_decode_bytes_can_span_token_boundaries() {
    let model = synthetic_bpe(&[]);
    let ids = encode_one_piece(&model, &byte_pieces("é")[0]);
    assert_eq!(ids.len(), 2);

    let mut bytes = Vec::new();
    for id in ids {
        bytes.extend_from_slice(model.base_token_bytes(id).unwrap());
    }
    assert_eq!(String::from_utf8_lossy(&bytes), "é");
    assert_eq!(model.base_token_bytes(-1), None);
    assert_eq!(model.base_token_bytes(256), None);
}

#[test]
fn bpe_rejects_non_contiguous_ids_and_missing_byte_symbols() {
    let mut missing_byte = HashMap::new();
    missing_byte.insert("a".to_owned(), 0);
    assert!(matches!(
        BpeModel::new(BpeDefinition::new(missing_byte, Vec::new())),
        Err(TokenizerError::InvalidVocabulary(_))
    ));

    let mut non_contiguous = HashMap::new();
    for byte in 0_u16..=255 {
        non_contiguous.insert(
            super::pretokenize::byte_char(byte as u8).to_string(),
            u32::from(byte) + 1,
        );
    }
    assert!(matches!(
        BpeModel::new(BpeDefinition::new(non_contiguous, Vec::new())),
        Err(TokenizerError::InvalidVocabulary(_))
    ));
}

#[test]
fn bpe_rejects_merges_with_missing_inputs_or_outputs() {
    let mut vocab = HashMap::new();
    for byte in 0_u16..=255 {
        vocab.insert(
            super::pretokenize::byte_char(byte as u8).to_string(),
            u32::from(byte),
        );
    }
    assert!(matches!(
        BpeModel::new(BpeDefinition::new(
            vocab.clone(),
            vec![("missing".into(), "a".into())]
        )),
        Err(TokenizerError::InvalidVocabulary(_))
    ));
    assert!(matches!(
        BpeModel::new(BpeDefinition::new(vocab, vec![("a".into(), "b".into())])),
        Err(TokenizerError::InvalidVocabulary(_))
    ));
}

fn object(fields: impl IntoIterator<Item = (&'static str, JsonValue)>) -> JsonValue {
    JsonValue::from(
        fields
            .into_iter()
            .map(|(name, value)| (name.to_owned(), value))
            .collect::<HashMap<_, _>>(),
    )
}

fn string(value: &str) -> JsonValue {
    JsonValue::from(value.to_owned())
}

fn byte_level_component() -> JsonValue {
    object([
        ("type", string("ByteLevel")),
        ("add_prefix_space", false.into()),
        ("trim_offsets", false.into()),
        ("use_regex", false.into()),
    ])
}

fn added_token(id: u32, content: &str) -> JsonValue {
    object([
        ("id", (id as f64).into()),
        ("content", string(content)),
        ("single_word", false.into()),
        ("lstrip", false.into()),
        ("rstrip", false.into()),
        ("normalized", false.into()),
        ("special", true.into()),
    ])
}

fn small_tokenizer_json() -> JsonValue {
    let vocab = (0_u16..=255)
        .map(|byte| {
            (
                super::pretokenize::byte_char(byte as u8).to_string(),
                JsonValue::from(byte as f64),
            )
        })
        .collect::<HashMap<_, _>>();
    object([
        ("version", string("1.0")),
        ("truncation", ().into()),
        ("padding", ().into()),
        (
            "added_tokens",
            vec![added_token(256, "<test-token>")].into(),
        ),
        ("normalizer", object([("type", string("NFC"))])),
        (
            "pre_tokenizer",
            object([
                ("type", string("Sequence")),
                (
                    "pretokenizers",
                    vec![
                        object([
                            ("type", string("Split")),
                            (
                                "pattern",
                                object([("Regex", string(super::pretokenize::QWEN_SPLIT_PATTERN))]),
                            ),
                            ("behavior", string("Isolated")),
                            ("invert", false.into()),
                        ]),
                        byte_level_component(),
                    ]
                    .into(),
                ),
            ]),
        ),
        ("post_processor", byte_level_component()),
        ("decoder", byte_level_component()),
        (
            "model",
            object([
                ("type", string("BPE")),
                ("dropout", ().into()),
                ("unk_token", ().into()),
                ("continuing_subword_prefix", string("")),
                ("end_of_word_suffix", string("")),
                ("fuse_unk", false.into()),
                ("byte_fallback", false.into()),
                ("ignore_merges", false.into()),
                ("vocab", vocab.into()),
                ("merges", Vec::<JsonValue>::new().into()),
            ]),
        ),
    ])
}

#[test]
fn format_parses_the_supported_pipeline_before_artifact_identity_validation() {
    let definition =
        TokenizerDefinition::parse_json(&small_tokenizer_json().stringify().unwrap()).unwrap();
    assert_eq!(definition.base_token_count(), 256);
    assert_eq!(definition.added_token_count(), 1);
    assert!(matches!(
        definition.validate_qwen36_identity(),
        Err(TokenizerError::InvalidVocabulary(_))
    ));
}

#[test]
fn format_rejects_pipeline_semantics_it_does_not_implement() {
    let mut wrong_normalizer = small_tokenizer_json();
    wrong_normalizer["normalizer"]["type"] = string("NFKC");
    assert!(matches!(
        TokenizerDefinition::parse_json(&wrong_normalizer.stringify().unwrap()),
        Err(TokenizerError::UnsupportedDefinition(_))
    ));

    let mut wrong_pattern = small_tokenizer_json();
    wrong_pattern["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = string("\\s+");
    assert!(matches!(
        TokenizerDefinition::parse_json(&wrong_pattern.stringify().unwrap()),
        Err(TokenizerError::UnsupportedDefinition(_))
    ));

    let mut byte_fallback = small_tokenizer_json();
    byte_fallback["model"]["byte_fallback"] = true.into();
    assert!(matches!(
        TokenizerDefinition::parse_json(&byte_fallback.stringify().unwrap()),
        Err(TokenizerError::UnsupportedDefinition(_))
    ));
}

#[test]
fn format_rejects_unsupported_or_ambiguous_added_tokens() {
    let mut normalized = small_tokenizer_json();
    normalized["added_tokens"][0]["normalized"] = true.into();
    assert!(matches!(
        TokenizerDefinition::parse_json(&normalized.stringify().unwrap()),
        Err(TokenizerError::UnsupportedDefinition(_))
    ));

    let mut duplicate_id = small_tokenizer_json();
    duplicate_id["added_tokens"]
        .get_mut::<Vec<JsonValue>>()
        .unwrap()
        .push(added_token(256, "<other-token>"));
    assert!(matches!(
        TokenizerDefinition::parse_json(&duplicate_id.stringify().unwrap()),
        Err(TokenizerError::InvalidVocabulary(_))
    ));
}

#[test]
fn format_rejects_unknown_fields_instead_of_silently_ignoring_them() {
    let mut definition = small_tokenizer_json();
    definition
        .get_mut::<HashMap<String, JsonValue>>()
        .unwrap()
        .insert("new_behavior".into(), true.into());
    assert!(matches!(
        TokenizerDefinition::parse_json(&definition.stringify().unwrap()),
        Err(TokenizerError::UnsupportedDefinition(_))
    ));
}

#[test]
fn format_rejects_numeric_literals_that_f64_would_round_into_valid_ids() {
    let source = small_tokenizer_json().stringify().unwrap();
    assert!(source.contains("\"id\":256"));
    for malformed in ["255.99999999999999", "-1e-400", "4294967295.0000001"] {
        let source = source.replacen("\"id\":256", &format!("\"id\":{malformed}"), 1);
        assert!(matches!(
            TokenizerDefinition::parse_json(&source),
            Err(TokenizerError::Json(_))
        ));
    }
}

fn tokenizer_from_json(value: JsonValue) -> QwenTokenizer {
    let definition = TokenizerDefinition::parse_json(&value.stringify().unwrap()).unwrap();
    QwenTokenizer::from_definition(definition).unwrap()
}

#[test]
fn tokenizer_normalizes_ordinary_text_without_inserting_control_tokens() {
    let tokenizer = tokenizer_from_json(small_tokenizer_json());
    assert!(tokenizer.encode("").is_empty());
    assert_eq!(tokenizer.encode("é e\u{301}"), tokenizer.encode("é é"));
    assert_eq!(
        tokenizer.decode(&tokenizer.encode("é e\u{301}")).unwrap(),
        "é é"
    );
    assert!(!tokenizer.encode("ordinary text").contains(&256));
}

#[test]
fn tokenizer_extracts_added_tokens_before_nfc_with_leftmost_longest_matching() {
    let mut definition = small_tokenizer_json();
    definition["added_tokens"] = vec![
        added_token(256, "e\u{301}"),
        added_token(257, "<test>"),
        added_token(258, "<test>long"),
    ]
    .into();
    let tokenizer = tokenizer_from_json(definition);

    assert_eq!(tokenizer.encode("e\u{301}"), [256]);
    assert_ne!(tokenizer.encode("é"), [256]);
    assert_eq!(
        tokenizer.encode("x<test>longy"),
        [u32::from(b'x') as i32, 258, u32::from(b'y') as i32]
    );
}

#[test]
fn tokenizer_exposes_direct_token_ids_and_preserves_added_tokens_on_decode() {
    let tokenizer = tokenizer_from_json(small_tokenizer_json());
    assert_eq!(tokenizer.token_count(), 257);
    assert_eq!(tokenizer.token_id("a"), Some(i32::from(b'a')));
    assert_eq!(tokenizer.token_id("<test-token>"), Some(256));
    assert_eq!(tokenizer.token_id("missing"), None);
    assert_eq!(tokenizer.decode(&[256]).unwrap(), "<test-token>");
}

#[test]
fn tokenizer_decode_accumulates_bytes_and_rejects_unknown_ids() {
    let tokenizer = tokenizer_from_json(small_tokenizer_json());
    let ids = tokenizer.encode("👩‍💻");
    assert!(ids.len() > 1);
    assert_eq!(tokenizer.decode(&ids).unwrap(), "👩‍💻");
    assert_eq!(
        tokenizer.decode(&[-1]),
        Err(TokenizerError::UnknownTokenId(-1))
    );
    assert_eq!(
        tokenizer.decode(&[257]),
        Err(TokenizerError::UnknownTokenId(257))
    );
}

#[test]
fn real_qwen36_identity_rejects_same_count_bpe_mutations_when_available() {
    let tokenizer_path = real_qwen36_model_dir().join("tokenizer.json");
    if !tokenizer_path.is_file() {
        eprintln!(
            "skipping real Qwen3.6 tokenizer identity mutations; {} is unavailable",
            tokenizer_path.display()
        );
        return;
    }
    let source = std::fs::read_to_string(tokenizer_path).unwrap();
    let original: JsonValue = source.parse().unwrap();

    let mut swapped_ids = original.clone();
    let vocab = swapped_ids["model"]["vocab"]
        .get_mut::<HashMap<String, JsonValue>>()
        .unwrap();
    let bang = vocab.get("!").unwrap().clone();
    let quote = vocab.get("\"").unwrap().clone();
    vocab.insert("!".into(), quote);
    vocab.insert("\"".into(), bang);
    let swapped_ids = TokenizerDefinition::parse_json(&swapped_ids.stringify().unwrap()).unwrap();
    assert!(matches!(
        swapped_ids.validate_qwen36_identity(),
        Err(TokenizerError::InvalidVocabulary(_))
    ));

    let mut reordered_merges = original;
    reordered_merges["model"]["merges"]
        .get_mut::<Vec<JsonValue>>()
        .unwrap()
        .swap(0, 1);
    let reordered_merges =
        TokenizerDefinition::parse_json(&reordered_merges.stringify().unwrap()).unwrap();
    assert!(matches!(
        reordered_merges.validate_qwen36_identity(),
        Err(TokenizerError::InvalidVocabulary(_))
    ));
}

#[test]
fn real_qwen36_tokenizer_matches_hugging_face_oracle_when_available() {
    let model_dir = real_qwen36_model_dir();
    if !model_dir.join("tokenizer.json").is_file() {
        eprintln!(
            "skipping real Qwen3.6 tokenizer oracle; {} is unavailable",
            model_dir.display()
        );
        return;
    }

    let started = Instant::now();
    let tokenizer = QwenTokenizer::from_model_dir(&model_dir).unwrap();
    println!(
        "loaded {} tokenizer IDs from {} in {:.3}s",
        tokenizer.token_count(),
        model_dir.display(),
        started.elapsed().as_secs_f64()
    );
    assert_eq!(tokenizer.token_count(), 248_070);
    assert_eq!(tokenizer.token_id("<|endoftext|>"), Some(248_044));
    assert_eq!(tokenizer.token_id("<|im_start|>"), Some(248_045));
    assert_eq!(tokenizer.token_id("<|im_end|>"), Some(248_046));

    let cases: &[(&str, &str, &[i32])] = &[
        ("", "", &[]),
        ("Hello, world!", "Hello, world!", &[9419, 11, 1814, 0]),
        (" hello", " hello", &[23066]),
        ("   hello", "   hello", &[256, 23066]),
        ("Hello  world", "Hello  world", &[9419, 220, 1814]),
        ("hello\nworld", "hello\nworld", &[14556, 198, 14194]),
        (
            "hello \n  world",
            "hello \n  world",
            &[14556, 695, 220, 1814],
        ),
        (
            "'s 'S 're 'RE 'll 'D",
            "'s 'S 're 'RE 'll 'D",
            &[579, 359, 50, 359, 265, 359, 762, 359, 636, 359, 35],
        ),
        ("é e\u{301} Å Å", "é é Å Å", &[933, 3825, 76533, 76533]),
        ("你好，世界！", "你好，世界！", &[109266, 3709, 96748, 6115]),
        (
            "مَرْحَبًا दुनिया",
            "مَرْحَبًا दुनिया",
            &[9873, 164865, 28850, 150765, 150021, 184642, 235886],
        ),
        (
            "👩‍💻👍🏽",
            "👩‍💻👍🏽",
            &[
                9008, 239, 102, 373, 235, 88995, 119, 9008, 239, 235, 9008, 237, 121,
            ],
        ),
        (
            "123٤٥Ⅻ",
            "123٤٥Ⅻ",
            &[16, 17, 18, 149, 97, 149, 98, 68086, 104],
        ),
        (
            "foo... \r\nbar",
            "foo... \r\nbar",
            &[7724, 1076, 2449, 2185],
        ),
        (
            "a\u{a0}\u{2003}\u{85}b",
            "a\u{a0}\u{2003}\u{85}b",
            &[64, 3966, 373, 225, 126, 227, 65],
        ),
        (
            "<|im_start|>user\nHello<|im_end|>\n",
            "<|im_start|>user\nHello<|im_end|>\n",
            &[248045, 846, 198, 9419, 248046, 198],
        ),
        (
            "<think>x</think>",
            "<think>x</think>",
            &[248068, 87, 248069],
        ),
        (
            "rock’n’roll isn't I'M",
            "rock’n’roll isn't I'M",
            &[19896, 58005, 515, 1065, 4290, 914, 353, 26708],
        ),
        ("          a", "          a", &[670, 264]),
        (
            "\n\n  \ntext\t\tend  ",
            "\n\n  \ntext\t\tend  ",
            &[271, 2228, 1272, 197, 6050, 256],
        ),
    ];

    for &(text, decoded, expected_ids) in cases {
        let actual = tokenizer.encode(text);
        assert_eq!(actual, expected_ids, "token IDs differ for {text:?}");
        assert_eq!(
            tokenizer.decode(&actual).unwrap(),
            decoded,
            "decode differs for {text:?}"
        );
    }
    assert_eq!(
        tokenizer.decode(&[248_070]),
        Err(TokenizerError::UnknownTokenId(248_070))
    );
}
