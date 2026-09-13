use qs3::{QwenConfig, Status};
use std::ptr;

#[test]
fn runtime_capacity_must_fit_its_page_storage_without_overflow() {
    let mut config = QwenConfig::new(-1, ptr::null_mut(), 17).unwrap();
    assert_eq!(config.max_pages, 5);
    config.max_pages = 4;
    assert_eq!(config.validate(), Err(Status::InvalidArgument));
    config.max_pages = u32::MAX;
    assert_eq!(config.validate(), Err(Status::InvalidArgument));
    assert!(matches!(
        QwenConfig::new(-1, ptr::null_mut(), 0),
        Err(Status::InvalidArgument)
    ));
    assert!(matches!(
        QwenConfig::new(-1, ptr::null_mut(), u32::MAX),
        Err(Status::InvalidArgument)
    ));
}

#[test]
fn runtime_controls_reject_unsupported_batches_and_missing_workspace() {
    let base = QwenConfig::new(-1, ptr::null_mut(), 16).unwrap();
    let mut config = base;
    config.max_batch_rows = 2;
    assert_eq!(config.validate(), Err(Status::Unsupported));
    let mut config = base;
    config.max_live_requests = 2;
    assert_eq!(config.validate(), Err(Status::Unsupported));
    let mut config = base;
    config.qscb_workspace_bytes = 0;
    assert_eq!(config.validate(), Err(Status::InvalidArgument));
    assert!(base.validate().is_ok());
}
