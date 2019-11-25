#![feature(backtrace)]

//! An implementation of the GNU make jobserver.
//!
//! This crate is an implementation, in Rust, of the GNU `make` jobserver for
//! CLI tools that are interoperating with make or otherwise require some form
//! of parallelism limiting across process boundaries. This was originally
//! written for usage in Cargo to both (a) work when `cargo` is invoked from
//! `make` (using `make`'s jobserver) and (b) work when `cargo` invokes build
//! scripts, exporting a jobserver implementation for `make` processes to
//! transitively use.
//!
//! The jobserver implementation can be found in [detail online][docs] but
//! basically boils down to a cross-process semaphore. On Unix this is
//! implemented with the `pipe` syscall and read/write ends of a pipe and on
//! Windows this is implemented literally with IPC semaphores.
//!
//! The jobserver protocol in `make` also dictates when tokens are acquired to
//! run child work, and clients using this crate should take care to implement
//! such details to ensure correct interoperation with `make` itself.
//!
//! ## Examples
//!
//! Connect to a jobserver that was set up by `make` or a different process:
//!
//! ```no_run
//! use jobserver::Client;
//!
//! // See API documentation for why this is `unsafe`
//! let client = match unsafe { Client::from_env() } {
//!     Some(client) => client,
//!     None => panic!("client not configured"),
//! };
//! ```
//!
//! Acquire and release token from a jobserver:
//!
//! ```no_run
//! use jobserver::Client;
//!
//! let client = unsafe { Client::from_env().unwrap() };
//! let token = client.acquire().unwrap(); // blocks until it is available
//! drop(token); // releases the token when the work is done
//! ```
//!
//! Create a new jobserver and configure a child process to have access:
//!
//! ```
//! use std::process::Command;
//! use jobserver::Client;
//!
//! let client = Client::new(4).expect("failed to create jobserver");
//! let mut cmd = Command::new("make");
//! client.configure(&mut cmd);
//! ```
//!
//! ## Caveats
//!
//! This crate makes no attempt to release tokens back to a jobserver on
//! abnormal exit of a process. If a process which acquires a token is killed
//! with ctrl-c or some similar signal then tokens will not be released and the
//! jobserver may be in a corrupt state.
//!
//! Note that this is typically ok as ctrl-c means that an entire build process
//! is being torn down, but it's worth being aware of at least!
//!
//! ## Windows caveats
//!
//! There appear to be two implementations of `make` on Windows. On MSYS2 one
//! typically comes as `mingw32-make` and the other as `make` itself. I'm not
//! personally too familiar with what's going on here, but for jobserver-related
//! information the `mingw32-make` implementation uses Windows semaphores
//! whereas the `make` program does not. The `make` program appears to use file
//! descriptors and I'm not really sure how it works, so this crate is not
//! compatible with `make` on Windows. It is, however, compatible with
//! `mingw32-make`.
//!
//! [docs]: http://make.mad-scientist.net/papers/jobserver-implementation/

#![deny(missing_docs, missing_debug_implementations)]
#![doc(html_root_url = "https://docs.rs/jobserver/0.1")]

//use std::env;
use std::io;
use std::process::Command;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

/// A client of a jobserver
///
/// This structure is the main type exposed by this library, and is where
/// interaction to a jobserver is configured through. Clients are either created
/// from scratch in which case the internal semphore is initialied on the spot,
/// or a client is created from the environment to connect to a jobserver
/// already created.
///
/// Some usage examples can be found in the crate documentation for using a
/// client.
///
/// Note that a `Client` implements the `Clone` trait, and all instances of a
/// `Client` refer to the same jobserver instance.
#[derive(Clone, Debug)]
pub struct Client {
    inner: Arc<imp::Client>,
}

/// An acquired token from a jobserver.
///
/// This token will be released back to the jobserver when it is dropped and
/// otherwise represents the ability to spawn off another thread of work.
#[derive(Debug)]
pub struct Acquired {
    client: Arc<imp::Client>,
    data: imp::Acquired,
}

#[derive(Default, Debug)]
struct HelperState {
    lock: Mutex<HelperInner>,
    cvar: Condvar,
}

#[derive(Default, Debug)]
struct HelperInner {
    requests: usize,
    producer_done: bool,
    consumer_done: bool,
}

impl Client {
    /// Creates a new jobserver initialized with the given parallelism limit.
    ///
    /// A client to the jobserver created will be returned. This client will
    /// allow at most `limit` tokens to be acquired from it in parallel. More
    /// calls to `acquire` will cause the calling thread to block.
    ///
    /// Note that the created `Client` is not automatically inherited into
    /// spawned child processes from this program. Manual usage of the
    /// `configure` function is required for a child process to have access to a
    /// job server.
    ///
    /// # Examples
    ///
    /// ```
    /// use jobserver::Client;
    ///
    /// let client = Client::new(4).expect("failed to create jobserver");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if any I/O error happens when attempting to create the
    /// jobserver client.
    pub fn new(limit: usize) -> io::Result<Client> {
        Ok(Client {
            inner: Arc::new(imp::Client::new(limit)?),
        })
    }

    /// Attempts to connect to the jobserver specified in this process's
    /// environment.
    ///
    /// When the a `make` executable calls a child process it will configure the
    /// environment of the child to ensure that it has handles to the jobserver
    /// it's passing down. This function will attempt to look for these details
    /// and connect to the jobserver.
    ///
    /// Note that the created `Client` is not automatically inherited into
    /// spawned child processes from this program. Manual usage of the
    /// `configure` function is required for a child process to have access to a
    /// job server.
    ///
    /// # Return value
    ///
    /// If a jobserver was found in the environment and it looks correct then
    /// `Some` of the connected client will be returned. If no jobserver was
    /// found then `None` will be returned.
    ///
    /// Note that on Unix the `Client` returned **takes ownership of the file
    /// descriptors specified in the environment**. Jobservers on Unix are
    /// implemented with `pipe` file descriptors, and they're inherited from
    /// parent processes. This `Client` returned takes ownership of the file
    /// descriptors for this process and will close the file descriptors after
    /// this value is dropped.
    ///
    /// Additionally on Unix this function will configure the file descriptors
    /// with `CLOEXEC` so they're not automatically inherited by spawned
    /// children.
    ///
    /// # Unsafety
    ///
    /// This function is `unsafe` to call on Unix specifically as it
    /// transitively requires usage of the `from_raw_fd` function, which is
    /// itself unsafe in some circumstances.
    ///
    /// It's recommended to call this function very early in the lifetime of a
    /// program before any other file descriptors are opened. That way you can
    /// make sure to take ownership properly of the file descriptors passed
    /// down, if any.
    ///
    /// It's generally unsafe to call this function twice in a program if the
    /// previous invocation returned `Some`.
    ///
    /// Note, though, that on Windows it should be safe to call this function
    /// any number of times.
    pub unsafe fn from_env() -> Option<Client> {
        imp::Client::open().map(|c| Client { inner: Arc::new(c) })
    }

    /// Acquires a token from this jobserver client.
    ///
    /// This function will block the calling thread until a new token can be
    /// acquired from the jobserver.
    ///
    /// # Return value
    ///
    /// On successful acquisition of a token an instance of `Acquired` is
    /// returned. This structure, when dropped, will release the token back to
    /// the jobserver. It's recommended to avoid leaking this value.
    ///
    /// # Errors
    ///
    /// If an I/O error happens while acquiring a token then this function will
    /// return immediately with the error. If an error is returned then a token
    /// was not acquired.
    pub fn acquire(&self) -> io::Result<Acquired> {
        let data = self.inner.acquire()?;
        Ok(Acquired {
            client: self.inner.clone(),
            data: data,
        })
    }

    /// Configures a child process to have access to this client's jobserver as
    /// well.
    ///
    /// This function is required to be called to ensure that a jobserver is
    /// properly inherited to a child process. If this function is *not* called
    /// then this `Client` will not be accessible in the child process. In other
    /// words, if not called, then `Client::from_env` will return `None` in the
    /// child process (or the equivalent of `Child::from_env` that `make` uses).
    ///
    /// ## Platform-specific behavior
    ///
    /// On Unix and Windows this will clobber the `CARGO_MAKEFLAGS` environment
    /// variables for the child process, and on Unix this will also allow the
    /// two file descriptors for this client to be inherited to the child.
    ///
    /// On platforms other than Unix and Windows this panics.
    pub fn configure(&self, cmd: &mut Command) {
        let arg = self.inner.string_arg();
        cmd.env("RUST_JOBSERVER", &arg);
        self.inner.configure(cmd);
    }

    /// Converts this `Client` into a helper thread to deal with a blocking
    /// `acquire` function a little more easily.
    ///
    /// The fact that the `acquire` function on `Client` blocks isn't always
    /// the easiest to work with. Typically you're using a jobserver to
    /// manage running other events in parallel! This means that you need to
    /// either (a) wait for an existing job to finish or (b) wait for a
    /// new token to become available.
    ///
    /// Unfortunately the blocking in `acquire` happens at the implementation
    /// layer of jobservers. On Unix this requires a blocking call to `read`
    /// and on Windows this requires one of the `WaitFor*` functions. Both
    /// of these situations aren't the easiest to deal with:
    ///
    /// * On Unix there's basically only one way to wake up a `read` early, and
    ///   that's through a signal. This is what the `make` implementation
    ///   itself uses, relying on `SIGCHLD` to wake up a blocking acquisition
    ///   of a new job token. Unfortunately nonblocking I/O is not an option
    ///   here, so it means that "waiting for one of two events" means that
    ///   the latter event must generate a signal! This is not always the case
    ///   on unix for all jobservers.
    ///
    /// * On Windows you'd have to basically use the `WaitForMultipleObjects`
    ///   which means that you've got to canonicalize all your event sources
    ///   into a `HANDLE` which also isn't the easiest thing to do
    ///   unfortunately.
    ///
    /// This function essentially attempts to ease these limitations by
    /// converting this `Client` into a helper thread spawned into this
    /// process. The application can then request that the helper thread
    /// acquires tokens and the provided closure will be invoked for each token
    /// acquired.
    ///
    /// The intention is that this function can be used to translate the event
    /// of a token acquisition into an arbitrary user-defined event.
    ///
    /// # Arguments
    ///
    /// This function will consume the `Client` provided to be transferred to
    /// the helper thread that is spawned. Additionally a closure `f` is
    /// provided to be invoked whenever a token is acquired.
    ///
    /// This closure is only invoked after calls to
    /// `HelperThread::request_token` have been made and a token itself has
    /// been acquired. If an error happens while acquiring the token then
    /// an error will be yielded to the closure as well.
    ///
    /// # Return Value
    ///
    /// This function will return an instance of the `HelperThread` structure
    /// which is used to manage the helper thread associated with this client.
    /// Through the `HelperThread` you'll request that tokens are acquired.
    /// When acquired, the closure provided here is invoked.
    ///
    /// When the `HelperThread` structure is returned it will be gracefully
    /// torn down, and the calling thread will be blocked until the thread is
    /// torn down (which should be prompt).
    ///
    /// # Errors
    ///
    /// This function may fail due to creation of the helper thread or
    /// auxiliary I/O objects to manage the helper thread. In any of these
    /// situations the error is propagated upwards.
    ///
    /// # Platform-specific behavior
    ///
    /// On Windows this function behaves pretty normally as expected, but on
    /// Unix the implementation is... a little heinous. As mentioned above
    /// we're forced into blocking I/O for token acquisition, namely a blocking
    /// call to `read`. We must be able to unblock this, however, to tear down
    /// the helper thread gracefully!
    ///
    /// Essentially what happens is that we'll send a signal to the helper
    /// thread spawned and rely on `EINTR` being returned to wake up the helper
    /// thread. This involves installing a global `SIGUSR1` handler that does
    /// nothing along with sending signals to that thread. This may cause
    /// odd behavior in some applications, so it's recommended to review and
    /// test thoroughly before using this.
    pub fn into_helper_thread<F>(self, f: F) -> io::Result<HelperThread>
    where
        F: FnMut(io::Result<Acquired>) + Send + 'static,
    {
        let state = Arc::new(HelperState::default());
        Ok(HelperThread {
            inner: Some(imp::spawn_helper(self, state.clone(), Box::new(f))?),
            state,
        })
    }

    /// Blocks the current thread until a token is acquired.
    ///
    /// This is the same as `acquire`, except that it doesn't return an RAII
    /// helper. If successful the process will need to guarantee that
    /// `release_raw` is called in the future.
    pub fn acquire_raw(&self) -> io::Result<()> {
        self.inner.acquire()?;
        Ok(())
    }

    /// Releases a jobserver token back to the original jobserver.
    ///
    /// This is intended to be paired with `acquire_raw` if it was called, but
    /// in some situations it could also be called to relinquish a process's
    /// implicit token temporarily which is then re-acquired later.
    pub fn release_raw(&self) -> io::Result<()> {
        self.inner.release(None)?;
        Ok(())
    }
}

impl Drop for Acquired {
    fn drop(&mut self) {
        drop(self.client.release(Some(&self.data)));
    }
}

/// Structure returned from `Client::into_helper_thread` to manage the lifetime
/// of the helper thread returned, see those associated docs for more info.
#[derive(Debug)]
pub struct HelperThread {
    inner: Option<imp::Helper>,
    state: Arc<HelperState>,
}

impl HelperThread {
    /// Request that the helper thread acquires a token, eventually calling the
    /// original closure with a token when it's available.
    ///
    /// For more information, see the docs on that function.
    pub fn request_token(&self) {
        // Indicate that there's one more request for a token and then wake up
        // the helper thread if it's sleeping.
        self.state.lock().requests += 1;
        self.state.cvar.notify_one();
    }
}

impl Drop for HelperThread {
    fn drop(&mut self) {
        // Flag that the producer half is done so the helper thread should exit
        // quickly if it's waiting. Wake it up if it's actually waiting
        self.state.lock().producer_done = true;
        self.state.cvar.notify_one();

        // ... and afterwards perform any thread cleanup logic
        self.inner.take().unwrap().join();
    }
}

impl HelperState {
    fn lock(&self) -> MutexGuard<'_, HelperInner> {
        self.lock.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn for_each_request(&self, mut f: impl FnMut()) {
        let mut lock = self.lock();

        // We only execute while we could receive requests, but as soon as
        // that's `false` we're out of here.
        while !lock.producer_done {
            // If no one's requested a token then we wait for someone to
            // request a token.
            if lock.requests == 0 {
                lock = self.cvar.wait(lock).unwrap_or_else(|e| e.into_inner());
                continue;
            }

            // Consume the request for a token, and then actually acquire a
            // token after unlocking our lock (not that acquisition happens in
            // `f`). This ensures that we don't actually hold the lock if we
            // wait for a long time for a token.
            lock.requests -= 1;
            drop(lock);
            f();
            lock = self.lock();
        }
        lock.consumer_done = true;
        self.cvar.notify_one();
    }
}

#[cfg(unix)]
mod imp {
    extern crate libc;

    use std::io;
    use std::mem;
    use std::os::unix::prelude::*;
    use std::process::Command;
    use std::ptr;
    use std::sync::{Arc, Once};
    use std::thread::{self, Builder, JoinHandle};
    use std::time::Duration;

    use self::libc::c_int;

    pub struct Client {
        id: u32,
        semaphore: *mut self::libc::sem_t,
        is_primary: bool,
    }

    impl std::fmt::Debug for Client {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Client")
                .field("id", &self.id)
                .field("value", &self.value())
                .field("primary", &self.is_primary)
                .field("process", &std::env::args_os())
                .field("location", &std::backtrace::Backtrace::force_capture())
                .finish()
        }
    }

    // semaphores are MT-safe
    unsafe impl Sync for Client {}
    unsafe impl Send for Client {}

    #[derive(Debug)]
    pub struct Acquired;

    impl Client {
        pub fn new(limit: usize) -> io::Result<Client> {
            let client = unsafe { Client::mk(limit as u32)? };
            Ok(client)
        }

        unsafe fn mk(limit: u32) -> io::Result<Client> {
            let mut st;
            let mut i = 0;
            loop {
                let name = std::ffi::CString::new(format!("jobserver-rust-{}", i)).unwrap();
                st = self::libc::sem_open(
                    name.as_ptr(),
                    self::libc::O_CREAT | self::libc::O_EXCL,
                    0o664,
                    limit,
                );

                if st == self::libc::SEM_FAILED {
                    let err = std::io::Error::last_os_error();
                    if err.kind() == std::io::ErrorKind::AlreadyExists {
                        i += 1;
                        continue;
                    }
                    return Err(std::io::Error::last_os_error());
                } else {
                    break;
                }
            }

            Ok(Client {
                semaphore: st,
                id: i,
                is_primary: true,
            })
        }

        pub unsafe fn open() -> Option<Client> {
            let id = std::env::var("RUST_JOBSERVER")
                .ok()?
                .parse::<u32>()
                .unwrap();
            let name = std::ffi::CString::new(format!("jobserver-rust-{}", id)).unwrap();
            let st = self::libc::sem_open(name.as_ptr(), 0);

            if st == self::libc::SEM_FAILED {
                return None;
            }

            Some(Client {
                semaphore: st,
                id,
                is_primary: false,
            })
        }

        pub fn acquire(&self) -> io::Result<Acquired> {
            loop {
                let ret = unsafe { self::libc::sem_wait(self.semaphore) };
                if ret != 0 {
                    let err = std::io::Error::last_os_error();
                    if err.kind() == std::io::ErrorKind::Interrupted {
                        continue;
                    }
                    //eprintln!("acquire failed {:?}: {:?}", self, err);
                    return Err(err);
                }
                //eprintln!("acquire success {:?}", self);
                break;
            }

            Ok(Acquired)
        }

        fn value(&self) -> i32 {
            unsafe {
                let mut val = 0;
                assert_eq!(libc::sem_getvalue(self.semaphore, &mut val), 0);
                val
            }
        }

        pub fn release(&self, _: Option<&Acquired>) -> io::Result<()> {
            let ret = unsafe { self::libc::sem_post(self.semaphore) };
            if ret != 0 {
                let err = std::io::Error::last_os_error();
                //eprintln!("release failed {:?}: {:?}", self, err);
                return Err(err);
            }
            //eprintln!("release success {:?}", self);

            Ok(())
        }

        pub fn string_arg(&self) -> String {
            format!("{}", self.id)
        }

        pub fn configure(&self, _: &mut Command) {
            // nothing to do
        }
    }

    impl Drop for Client {
        fn drop(&mut self) {
            unsafe {
                // close our handle to the semaphore
                self::libc::sem_close(self.semaphore);
                if self.is_primary {
                    // Destroy the (global) name of this semaphore; any
                    // currently open handles to this semaphore can still
                    // function as needed, but new ones with the same name will
                    // refer to a different semaphore
                    let name =
                        std::ffi::CString::new(format!("jobserver-rust-{}", self.id)).unwrap();
                    self::libc::sem_unlink(name.as_ptr());
                }
            }
        }
    }

    #[derive(Debug)]
    pub struct Helper {
        thread: JoinHandle<()>,
        state: Arc<super::HelperState>,
    }

    pub(crate) fn spawn_helper(
        client: ::Client,
        state: Arc<super::HelperState>,
        mut f: Box<dyn FnMut(io::Result<::Acquired>) + Send>,
    ) -> io::Result<Helper> {
        static USR1_INIT: Once = Once::new();
        let mut err = None;
        USR1_INIT.call_once(|| unsafe {
            let mut new: libc::sigaction = mem::zeroed();
            new.sa_sigaction = sigusr1_handler as usize;
            new.sa_flags = libc::SA_SIGINFO as _;
            if libc::sigaction(libc::SIGUSR1, &new, ptr::null_mut()) != 0 {
                err = Some(io::Error::last_os_error());
            }
        });

        if let Some(e) = err.take() {
            return Err(e);
        }

        let state2 = state.clone();
        let thread = Builder::new().spawn(move || {
            state2.for_each_request(|| f(client.acquire()));
        })?;

        Ok(Helper { thread, state })
    }

    impl Helper {
        pub fn join(self) {
            let dur = Duration::from_millis(10);
            let mut state = self.state.lock();
            debug_assert!(state.producer_done);

            // We need to join our helper thread, and it could be blocked in one
            // of two locations. First is the wait for a request, but the
            // initial drop of `HelperState` will take care of that. Otherwise
            // it may be blocked in `client.acquire()`. We actually have no way
            // of interrupting that, so resort to `pthread_kill` as a fallback.
            // This signal should interrupt any blocking `read` call with
            // `io::ErrorKind::Interrupt` and cause the thread to cleanly exit.
            //
            // Note that we don'tdo this forever though since there's a chance
            // of bugs, so only do this opportunistically to make a best effort
            // at clearing ourselves up.
            for _ in 0..100 {
                if state.consumer_done {
                    break;
                }
                unsafe {
                    // Ignore the return value here of `pthread_kill`,
                    // apparently on OSX if you kill a dead thread it will
                    // return an error, but on other platforms it may not. In
                    // that sense we don't actually know if this will succeed or
                    // not!
                    libc::pthread_kill(self.thread.as_pthread_t() as _, libc::SIGUSR1);
                }
                state = self
                    .state
                    .cvar
                    .wait_timeout(state, dur)
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
                thread::yield_now(); // we really want the other thread to run
            }

            // If we managed to actually see the consumer get done, then we can
            // definitely wait for the thread. Otherwise it's... of in the ether
            // I guess?
            if state.consumer_done {
                drop(self.thread.join());
            }
        }
    }

    extern "C" fn sigusr1_handler(
        _signum: c_int,
        _info: *mut libc::siginfo_t,
        _ptr: *mut libc::c_void,
    ) {
        // nothing to do
    }
}

#[cfg(windows)]
mod imp {
    extern crate getrandom;

    use std::ffi::CString;
    use std::io;
    use std::process::Command;
    use std::ptr;
    use std::sync::Arc;
    use std::thread::{Builder, JoinHandle};

    #[derive(Debug)]
    pub struct Client {
        sem: Handle,
        name: String,
    }

    #[derive(Debug)]
    pub struct Acquired;

    type BOOL = i32;
    type DWORD = u32;
    type HANDLE = *mut u8;
    type LONG = i32;

    const ERROR_ALREADY_EXISTS: DWORD = 183;
    const FALSE: BOOL = 0;
    const INFINITE: DWORD = 0xffffffff;
    const SEMAPHORE_MODIFY_STATE: DWORD = 0x2;
    const SYNCHRONIZE: DWORD = 0x00100000;
    const TRUE: BOOL = 1;
    const WAIT_OBJECT_0: DWORD = 0;

    extern "system" {
        fn CloseHandle(handle: HANDLE) -> BOOL;
        fn SetEvent(hEvent: HANDLE) -> BOOL;
        fn WaitForMultipleObjects(
            ncount: DWORD,
            lpHandles: *const HANDLE,
            bWaitAll: BOOL,
            dwMilliseconds: DWORD,
        ) -> DWORD;
        fn CreateEventA(
            lpEventAttributes: *mut u8,
            bManualReset: BOOL,
            bInitialState: BOOL,
            lpName: *const i8,
        ) -> HANDLE;
        fn ReleaseSemaphore(
            hSemaphore: HANDLE,
            lReleaseCount: LONG,
            lpPreviousCount: *mut LONG,
        ) -> BOOL;
        fn CreateSemaphoreA(
            lpEventAttributes: *mut u8,
            lInitialCount: LONG,
            lMaximumCount: LONG,
            lpName: *const i8,
        ) -> HANDLE;
        fn OpenSemaphoreA(
            dwDesiredAccess: DWORD,
            bInheritHandle: BOOL,
            lpName: *const i8,
        ) -> HANDLE;
        fn WaitForSingleObject(hHandle: HANDLE, dwMilliseconds: DWORD) -> DWORD;
    }

    impl Client {
        pub fn new(limit: usize) -> io::Result<Client> {
            // Try a bunch of random semaphore names until we get a unique one,
            // but don't try for too long.
            //
            // Note that `limit == 0` is a valid argument above but Windows
            // won't let us create a semaphore with 0 slots available to it. Get
            // `limit == 0` working by creating a semaphore instead with one
            // slot and then immediately acquire it (without ever releaseing it
            // back).
            for _ in 0..100 {
                let mut bytes = [0; 4];
                getrandom::getrandom(&mut bytes).map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::Other,
                        format!("failed to get random bytes: {}", e),
                    )
                })?;
                let mut name =
                    format!("__rust_jobserver_semaphore_{}\0", u32::from_ne_bytes(bytes));
                unsafe {
                    let create_limit = if limit == 0 { 1 } else { limit };
                    let r = CreateSemaphoreA(
                        ptr::null_mut(),
                        create_limit as LONG,
                        create_limit as LONG,
                        name.as_ptr() as *const _,
                    );
                    if r.is_null() {
                        return Err(io::Error::last_os_error());
                    }
                    let handle = Handle(r);

                    let err = io::Error::last_os_error();
                    if err.raw_os_error() == Some(ERROR_ALREADY_EXISTS as i32) {
                        continue;
                    }
                    name.pop(); // chop off the trailing nul
                    let client = Client {
                        sem: handle,
                        name: name,
                    };
                    if create_limit != limit {
                        client.acquire()?;
                    }
                    return Ok(client);
                }
            }

            Err(io::Error::new(
                io::ErrorKind::Other,
                "failed to find a unique name for a semaphore",
            ))
        }

        pub unsafe fn open(s: &str) -> Option<Client> {
            let name = match CString::new(s) {
                Ok(s) => s,
                Err(_) => return None,
            };

            let sem = OpenSemaphoreA(SYNCHRONIZE | SEMAPHORE_MODIFY_STATE, FALSE, name.as_ptr());
            if sem.is_null() {
                None
            } else {
                Some(Client {
                    sem: Handle(sem),
                    name: s.to_string(),
                })
            }
        }

        pub fn acquire(&self) -> io::Result<Acquired> {
            unsafe {
                let r = WaitForSingleObject(self.sem.0, INFINITE);
                if r == WAIT_OBJECT_0 {
                    Ok(Acquired)
                } else {
                    Err(io::Error::last_os_error())
                }
            }
        }

        pub fn release(&self, _data: Option<&Acquired>) -> io::Result<()> {
            unsafe {
                let r = ReleaseSemaphore(self.sem.0, 1, ptr::null_mut());
                if r != 0 {
                    Ok(())
                } else {
                    Err(io::Error::last_os_error())
                }
            }
        }

        pub fn string_arg(&self) -> String {
            self.name.clone()
        }

        pub fn configure(&self, _cmd: &mut Command) {
            // nothing to do here, we gave the name of our semaphore to the
            // child above
        }
    }

    #[derive(Debug)]
    struct Handle(HANDLE);
    // HANDLE is a raw ptr, but we're send/sync
    unsafe impl Sync for Handle {}
    unsafe impl Send for Handle {}

    impl Drop for Handle {
        fn drop(&mut self) {
            unsafe {
                CloseHandle(self.0);
            }
        }
    }

    #[derive(Debug)]
    pub struct Helper {
        event: Arc<Handle>,
        thread: JoinHandle<()>,
    }

    pub(crate) fn spawn_helper(
        client: ::Client,
        state: Arc<super::HelperState>,
        mut f: Box<dyn FnMut(io::Result<::Acquired>) + Send>,
    ) -> io::Result<Helper> {
        let event = unsafe {
            let r = CreateEventA(ptr::null_mut(), TRUE, FALSE, ptr::null());
            if r.is_null() {
                return Err(io::Error::last_os_error());
            } else {
                Handle(r)
            }
        };
        let event = Arc::new(event);
        let event2 = event.clone();
        let thread = Builder::new().spawn(move || {
            let objects = [event2.0, client.inner.sem.0];
            state.for_each_request(|| {
                const WAIT_OBJECT_1: u32 = WAIT_OBJECT_0 + 1;
                match unsafe { WaitForMultipleObjects(2, objects.as_ptr(), FALSE, INFINITE) } {
                    WAIT_OBJECT_0 => return,
                    WAIT_OBJECT_1 => f(Ok(::Acquired {
                        client: client.inner.clone(),
                        data: Acquired,
                    })),
                    _ => f(Err(io::Error::last_os_error())),
                }
            });
        })?;
        Ok(Helper { thread, event })
    }

    impl Helper {
        pub fn join(self) {
            // Unlike unix this logic is much easier. If our thread was blocked
            // in waiting for requests it should already be woken up and
            // exiting. Otherwise it's waiting for a token, so we wake it up
            // with a different event that it's also waiting on here. After
            // these two we should be guaranteed the thread is on its way out,
            // so we can safely `join`.
            let r = unsafe { SetEvent(self.event.0) };
            if r == 0 {
                panic!("failed to set event: {}", io::Error::last_os_error());
            }
            drop(self.thread.join());
        }
    }
}

#[cfg(not(any(unix, windows)))]
mod imp {
    use std::io;
    use std::process::Command;
    use std::sync::{Arc, Condvar, Mutex};
    use std::thread::{Builder, JoinHandle};

    #[derive(Debug)]
    pub struct Client {
        inner: Arc<Inner>,
    }

    #[derive(Debug)]
    struct Inner {
        count: Mutex<usize>,
        cvar: Condvar,
    }

    #[derive(Debug)]
    pub struct Acquired;

    impl Client {
        pub fn new(limit: usize) -> io::Result<Client> {
            Ok(Client {
                inner: Arc::new(Inner {
                    count: Mutex::new(limit),
                    cvar: Condvar::new(),
                }),
            })
        }

        pub unsafe fn open(_s: &str) -> Option<Client> {
            None
        }

        pub fn acquire(&self) -> io::Result<Acquired> {
            let mut lock = self.inner.count.lock().unwrap_or_else(|e| e.into_inner());
            while *lock == 0 {
                lock = self
                    .inner
                    .cvar
                    .wait(lock)
                    .unwrap_or_else(|e| e.into_inner());
            }
            *lock -= 1;
            Ok(Acquired(()))
        }

        pub fn release(&self, _data: Option<&Acquired>) -> io::Result<()> {
            let mut lock = self.inner.count.lock().unwrap_or_else(|e| e.into_inner());
            *lock += 1;
            drop(lock);
            self.inner.cvar.notify_one();
            Ok(())
        }

        pub fn string_arg(&self) -> String {
            panic!(
                "On this platform there is no cross process jobserver support,
                 so Client::configure is not supported."
            );
        }

        pub fn configure(&self, _cmd: &mut Command) {
            unreachable!();
        }
    }

    #[derive(Debug)]
    pub struct Helper {
        thread: JoinHandle<()>,
    }

    pub(crate) fn spawn_helper(
        client: ::Client,
        state: Arc<super::HelperState>,
        mut f: Box<dyn FnMut(io::Result<::Acquired>) + Send>,
    ) -> io::Result<Helper> {
        let thread = Builder::new().spawn(move || {
            state.for_each_request(|| f(client.acquire()));
        })?;

        Ok(Helper { thread: thread })
    }

    impl Helper {
        pub fn join(self) {
            // TODO: this is not correct if the thread is blocked in
            // `client.acquire()`.
            drop(self.thread.join());
        }
    }
}

#[test]
fn no_helper_deadlock() {
    let x = crate::Client::new(32).unwrap();
    let _y = x.clone();
    std::mem::drop(x.into_helper_thread(|_| {}).unwrap());
}
