use std::ops::Range;
use std::sync::OnceLock;

use unicode_general_category::{GeneralCategory, get_general_category};

pub(super) const QWEN_SPLIT_PATTERN: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

pub(super) fn byte_pieces(text: &str) -> Vec<String> {
    piece_ranges(text)
        .into_iter()
        .map(|range| {
            text[range]
                .as_bytes()
                .iter()
                .map(|byte| byte_char(*byte))
                .collect()
        })
        .collect()
}

pub(super) fn byte_char(byte: u8) -> char {
    byte_chars()[usize::from(byte)]
}

pub(super) fn char_byte(ch: char) -> Option<u8> {
    let codepoint = ch as usize;
    char_bytes().get(codepoint).copied().flatten()
}

#[cfg(test)]
pub(super) fn text_pieces(text: &str) -> Vec<&str> {
    piece_ranges(text)
        .into_iter()
        .map(|range| &text[range])
        .collect()
}

#[cfg(test)]
pub(super) fn classification_mask(ch: char) -> u8 {
    u8::from(is_letter(ch))
        | (u8::from(is_mark(ch)) << 1)
        | (u8::from(is_number(ch)) << 2)
        | (u8::from(ch.is_whitespace()) << 3)
}

fn piece_ranges(text: &str) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let end = next_piece_end(text, start);
        debug_assert!(end > start && text.is_char_boundary(end));
        ranges.push(start..end);
        start = end;
    }
    ranges
}

fn next_piece_end(text: &str, start: usize) -> usize {
    // These branches deliberately preserve the alternative order in
    // QWEN_SPLIT_PATTERN. In particular, an alternative starting at an
    // earlier space can win before a contraction beginning one byte later.
    if let Some(end) = contraction_end(text, start) {
        return end;
    }
    if let Some(end) = letter_or_mark_end(text, start) {
        return end;
    }

    let (first, first_end) = char_at(text, start);
    if is_number(first) {
        return first_end;
    }
    if let Some(end) = punctuation_end(text, start) {
        return end;
    }
    if first.is_whitespace() {
        let (whitespace_end, last_newline_end, last_char_start, count) =
            whitespace_run(text, start);
        if let Some(end) = last_newline_end {
            return end;
        }
        if whitespace_end == text.len() {
            return whitespace_end;
        }
        if count > 1 {
            return last_char_start;
        }
        return whitespace_end;
    }

    unreachable!("every Unicode scalar is covered by the Qwen split alternatives")
}

fn contraction_end(text: &str, start: usize) -> Option<usize> {
    if text.as_bytes()[start..].first() != Some(&b'\'') {
        return None;
    }
    for contraction in ["s", "t", "re", "ve", "m", "ll", "d"] {
        let mut cursor = start + 1;
        let mut matches = true;
        for expected in contraction.chars() {
            let Some((actual, next)) = char_at_opt(text, cursor) else {
                matches = false;
                break;
            };
            if !contraction_case_eq(actual, expected) {
                matches = false;
                break;
            }
            cursor = next;
        }
        if matches {
            return Some(cursor);
        }
    }
    None
}

fn contraction_case_eq(actual: char, expected: char) -> bool {
    // Exhaustive comparison with the pinned Oniguruma engine finds one
    // non-ASCII fold for these literals: long s joins the S/s equivalence.
    actual.eq_ignore_ascii_case(&expected) || (expected == 's' && actual == 'ſ')
}

fn letter_or_mark_end(text: &str, start: usize) -> Option<usize> {
    let (first, first_end) = char_at(text, start);
    let letters_start = if first != '\r' && first != '\n' && !is_letter(first) && !is_number(first)
    {
        let Some((next, _)) = char_at_opt(text, first_end) else {
            return is_letter_or_mark(first).then_some(first_end);
        };
        if !is_letter_or_mark(next) {
            return is_letter_or_mark(first).then_some(first_end);
        }
        first_end
    } else if is_letter_or_mark(first) {
        start
    } else {
        return None;
    };

    let mut end = letters_start;
    while let Some((ch, next)) = char_at_opt(text, end) {
        if !is_letter_or_mark(ch) {
            break;
        }
        end = next;
    }
    (end > letters_start).then_some(end)
}

fn punctuation_end(text: &str, start: usize) -> Option<usize> {
    let (first, first_end) = char_at(text, start);
    let mut cursor = start;
    if first == ' ' {
        let (next, _) = char_at_opt(text, first_end)?;
        if !is_punctuation_alternative(next) {
            return None;
        }
        cursor = first_end;
    } else if !is_punctuation_alternative(first) {
        return None;
    }

    let punctuation_start = cursor;
    while let Some((ch, next)) = char_at_opt(text, cursor) {
        if !is_punctuation_alternative(ch) {
            break;
        }
        cursor = next;
    }
    if cursor == punctuation_start {
        return None;
    }
    while let Some((ch @ ('\r' | '\n'), next)) = char_at_opt(text, cursor) {
        let _ = ch;
        cursor = next;
    }
    Some(cursor)
}

fn whitespace_run(text: &str, start: usize) -> (usize, Option<usize>, usize, usize) {
    let mut cursor = start;
    let mut last_newline_end = None;
    let mut last_char_start = start;
    let mut count = 0;
    while let Some((ch, next)) = char_at_opt(text, cursor) {
        if !ch.is_whitespace() {
            break;
        }
        last_char_start = cursor;
        if matches!(ch, '\r' | '\n') {
            last_newline_end = Some(next);
        }
        count += 1;
        cursor = next;
    }
    (cursor, last_newline_end, last_char_start, count)
}

fn is_punctuation_alternative(ch: char) -> bool {
    !ch.is_whitespace() && !is_letter(ch) && !is_mark(ch) && !is_number(ch)
}

fn is_letter_or_mark(ch: char) -> bool {
    is_letter(ch) || is_mark(ch)
}

fn is_letter(ch: char) -> bool {
    matches!(
        get_general_category(ch),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}

fn is_mark(ch: char) -> bool {
    matches!(
        get_general_category(ch),
        GeneralCategory::NonspacingMark
            | GeneralCategory::SpacingMark
            | GeneralCategory::EnclosingMark
    )
}

fn is_number(ch: char) -> bool {
    matches!(
        get_general_category(ch),
        GeneralCategory::DecimalNumber
            | GeneralCategory::LetterNumber
            | GeneralCategory::OtherNumber
    )
}

fn char_at(text: &str, at: usize) -> (char, usize) {
    char_at_opt(text, at).expect("character offset must be within input")
}

fn char_at_opt(text: &str, at: usize) -> Option<(char, usize)> {
    let ch = text.get(at..)?.chars().next()?;
    Some((ch, at + ch.len_utf8()))
}

fn byte_chars() -> &'static [char; 256] {
    static BYTE_CHARS: OnceLock<[char; 256]> = OnceLock::new();
    BYTE_CHARS.get_or_init(|| {
        let mut chars = ['\0'; 256];
        let mut direct = [false; 256];
        for byte in b'!'..=b'~' {
            direct[usize::from(byte)] = true;
            chars[usize::from(byte)] = char::from(byte);
        }
        for byte in b'\xa1'..=b'\xac' {
            direct[usize::from(byte)] = true;
            chars[usize::from(byte)] = char::from(byte);
        }
        for byte in b'\xae'..=b'\xff' {
            direct[usize::from(byte)] = true;
            chars[usize::from(byte)] = char::from(byte);
        }

        let mut extra = 0;
        for byte in 0_u16..=255 {
            let index = usize::from(byte);
            if !direct[index] {
                chars[index] = char::from_u32(256 + extra).expect("byte alphabet is valid Unicode");
                extra += 1;
            }
        }
        chars
    })
}

fn char_bytes() -> &'static [Option<u8>; 324] {
    static CHAR_BYTES: OnceLock<[Option<u8>; 324]> = OnceLock::new();
    CHAR_BYTES.get_or_init(|| {
        let mut bytes = [None; 324];
        for byte in 0_u16..=255 {
            bytes[byte_char(byte as u8) as usize] = Some(byte as u8);
        }
        bytes
    })
}
