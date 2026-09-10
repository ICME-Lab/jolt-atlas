use num_traits::Zero;
use std::sync::atomic::{AtomicUsize, Ordering};

static PENDING_BACKGROUND_DROPS: AtomicUsize = AtomicUsize::new(0);

pub fn drop_in_background_thread<T>(data: T)
where
    T: Send + 'static,
{
    // h/t https://abrams.cc/rust-dropping-things-in-another-thread
    PENDING_BACKGROUND_DROPS.fetch_add(1, Ordering::SeqCst);
    rayon::spawn(move || {
        drop(data);
        PENDING_BACKGROUND_DROPS.fetch_sub(1, Ordering::SeqCst);
    });
}

/// Blocks until every pending `drop_in_background_thread` has actually freed its data — for
/// bench harnesses repeating prove/verify in one process, so iterations don't stack memory.
pub fn wait_for_background_drops() {
    while PENDING_BACKGROUND_DROPS.load(Ordering::SeqCst) != 0 {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

pub fn unsafe_allocate_zero_vec<T: Sized + Zero>(size: usize) -> Vec<T> {
    // https://stackoverflow.com/questions/59314686/how-to-efficiently-create-a-large-vector-of-items-initialized-to-the-same-value

    #[cfg(test)]
    {
        // Check for safety of 0 allocation
        unsafe {
            let value = &T::zero();
            let ptr = value as *const T as *const u8;
            let bytes = std::slice::from_raw_parts(ptr, std::mem::size_of::<T>());
            assert!(bytes.iter().all(|&byte| byte == 0));
        }
    }

    // Bulk allocate zeros, unsafely
    let result: Vec<T>;
    unsafe {
        let layout = std::alloc::Layout::array::<T>(size).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut T;

        if ptr.is_null() {
            panic!("Zero vec allocation failed");
        }

        result = Vec::from_raw_parts(ptr, size, size);
    }
    result
}
