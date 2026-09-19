use std::{collections::HashMap, fs, time::{Instant, SystemTime}};
use tinyjson::JsonValue;
mod profile;
fn object(fields: impl IntoIterator<Item = (&'static str, JsonValue)>) -> JsonValue {
    fields.into_iter().map(|(k,v)| (k.to_owned(), v)).collect::<HashMap<_,_>>().into()
}
fn optional(value: Option<String>) -> JsonValue { value.map_or(JsonValue::Null, JsonValue::from) }
fn timestamp(_: SystemTime) -> String { unreachable!("offline reduction only") }
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let db = rusqlite::Connection::open_with_flags(&args[1], rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let start = Instant::now();
    let actual = profile::offline_cpu_summary(&db).unwrap();
    let elapsed = start.elapsed();
    let expected: JsonValue = fs::read_to_string(&args[2]).unwrap().parse().unwrap();
    assert_eq!(actual, expected["nsight"]["cpu"]);
    println!("CPU report identical; indexed reduction took {:.3} seconds", elapsed.as_secs_f64());
}
