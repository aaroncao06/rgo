// Concept: Send and Sync are why Arc<Mutex<T>> can be shared across
// threads but Rc<RefCell<T>> can't.
//
// Run with: cargo run --example thread_safety

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

fn main() {
    single_threaded_rc_refcell();

    let start = Instant::now();
    multi_threaded_arc_mutex();
    println!("Mutex version took {:?}", start.elapsed());

    let start = Instant::now();
    multi_threaded_atomic();
    println!("Atomic version took {:?}", start.elapsed());
}

fn single_threaded_rc_refcell() {
    let counter = Rc::new(RefCell::new(0));

    let handle_a = Rc::clone(&counter);
    let handle_b = Rc::clone(&counter);

    *handle_a.borrow_mut() += 1;
    *handle_b.borrow_mut() += 1;

    println!(
        "Rc<RefCell<T>> counter (single thread): {}",
        counter.borrow()
    );
}

// TODO(human): implement this.
//
// Spawn 4 threads. Each thread increments a shared counter 100_000
// times. Use Arc<Mutex<u64>> so the counter can move between threads
// (Arc is Send) and be mutated safely once shared (Mutex is Sync).
//
// Steps:
//   1. Wrap the counter: Arc::new(Mutex::new(0_u64))
//   2. For each of 4 threads: clone the Arc, move the clone into
//      thread::spawn(move || { ... })
//   3. Inside the thread, loop 100_000 times, locking the mutex each
//      time and incrementing: `*counter.lock().unwrap() += 1;`
//   4. Collect the JoinHandles, then .join() them all after spawning
//   5. Print the final count. It must equal 400_000 every run.
//
// Try it, then try swapping Arc<Mutex<u64>> for Rc<RefCell<u64>> in
// the signature/body and see what the compiler tells you — that
// error message is the Send/Sync check happening in front of you.
fn multi_threaded_arc_mutex() {
    let counter = Arc::new(Mutex::new(0_u64));
    let mut handles = Vec::new();
    for _i in 0..4 {
        let counter = counter.clone();
        let id = thread::spawn(move || {
            for _j in 0..100000 {
                *counter.lock().unwrap() += 1;
            }
        });
        handles.push(id);
    }
    for id in handles {
        id.join().unwrap();
    }
    println!("{}", *counter.lock().unwrap());
}

// TODO(human): implement this — the atomic version of the function above.
//
// Same shape: 4 threads, each incrementing a shared counter 100_000
// times, joined at the end, final count printed. But this time:
//   1. Wrap the counter: Arc::new(AtomicU64::new(0))
//   2. Inside each thread, instead of locking, call:
//      `counter.fetch_add(1, Ordering::Relaxed);`
//   3. After joining, read the final value with `counter.load(Ordering::Relaxed)`
//
// No lock, no blocking — `fetch_add` is a single atomic hardware
// instruction. Compare the printed timings: this version should
// noticeably beat the Mutex version, especially if you bump thread
// count or the loop count higher.
fn multi_threaded_atomic() {
    let counter = Arc::new(AtomicU64::new(0));
    let mut handles = Vec::new();
    for _i in 0..4 {
        let counter = counter.clone();
        let id = thread::spawn(move || {
            for _j in 0..100000 {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        });
        handles.push(id);
    }
    for id in handles {
        id.join().unwrap();
    }
    println!("{}", counter.load(Ordering::Relaxed));
}
