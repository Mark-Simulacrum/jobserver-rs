pub fn main() {
    let c = jobserver::Client::new(10);
    std::mem::drop(c);
    unsafe {
        let id = std::env::args().nth(1).unwrap();
        let name = std::ffi::CString::new(format!("jobserver-rust-{}", id)).unwrap();
        let st = libc::sem_open(name.as_ptr(), 0);
        assert_ne!(
            st,
            libc::SEM_FAILED,
            "failed to open jobserver: {:?}",
            std::io::Error::last_os_error()
        );
        loop {
            let mut val = 0;
            libc::sem_getvalue(st, &mut val);
            println!("{}", val);
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    }
}
