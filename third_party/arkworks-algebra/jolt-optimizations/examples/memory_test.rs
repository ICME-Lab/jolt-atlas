//! Memory test example for GLV-4 precomputation methods
//!
//! This example measures the memory usage of different precomputation strategies
//! for GLV-4 scalar multiplication on BN254 G2 points.

use ark_ff::UniformRand;
use ark_std::test_rng;
use jolt_optimizations::{
    glv_four_precompute, glv_four_precompute_windowed2_signed, G2Projective,
    PrecomputedShamir4Data, Windowed2Signed4Data,
};
use std::env;
use std::mem;

/// Helper function to calculate deep size of nested Vec structures
fn deep_size_of_shamir(data: &PrecomputedShamir4Data) -> usize {
    let mut total = mem::size_of::<PrecomputedShamir4Data>();
    total += data.shamir_tables.capacity()
        * mem::size_of::<jolt_optimizations::PrecomputedShamir4Table>();
    for table in &data.shamir_tables {
        total += table.table.capacity() * mem::size_of::<G2Projective>();
    }

    total
}

/// Helper function to calculate deep size of windowed data
fn deep_size_of_windowed(data: &Windowed2Signed4Data) -> usize {
    let mut total = mem::size_of::<Windowed2Signed4Data>();
    total += data.windowed2_tables.capacity()
        * mem::size_of::<jolt_optimizations::Windowed2Signed4Table>();

    for table in &data.windowed2_tables {
        total += table.signed_multiples.capacity() * mem::size_of::<G2Projective>();
    }

    total
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let num_points = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(100)
    } else {
        100
    };

    println!("Memory Test for GLV-4 Precomputation Methods");
    println!("Testing with {} G2 points", num_points);
    println!("{}", "=".repeat(60));

    let mut rng = test_rng();
    let points: Vec<G2Projective> = (0..(1 << num_points))
        .map(|_| G2Projective::rand(&mut rng))
        .collect();

    // Measure size of a single G2Projective point
    let g2_size = mem::size_of::<G2Projective>();
    println!("\nBase sizes:");
    println!("  Size of G2Projective: {} bytes", g2_size);
    println!(
        "  Size of Vec<G2Projective> overhead: {} bytes",
        mem::size_of::<Vec<G2Projective>>()
    );

    // Method 1: Online (no precomputation)
    println!("\n1. Online Method (no precomputation):");
    println!("   Memory per point: 0 bytes");
    println!("   Total memory: 0 bytes");

    // Method 2: Full Shamir table (256 entries per point)
    println!("\n2. Full Shamir Table Precomputation:");
    let shamir_data = glv_four_precompute(&points);
    let shamir_actual_size = deep_size_of_shamir(&shamir_data);
    println!(
        "   Actual runtime memory size: {} bytes ({:.2} MB)",
        shamir_actual_size,
        shamir_actual_size as f64 / (1024.0 * 1024.0)
    );
    println!(
        "   Memory per point: {} bytes ({:.2} KB)",
        shamir_actual_size / points.len(),
        (shamir_actual_size / points.len()) as f64 / 1024.0
    );

    // Method 3: 2-bit windowed signed (24 entries per point)
    println!("\n3. 2-bit Windowed Signed Precomputation:");
    let windowed_data = glv_four_precompute_windowed2_signed(&points);
    let windowed_actual_size = deep_size_of_windowed(&windowed_data);
    println!(
        "   Actual runtime memory size: {} bytes ({:.2} MB)",
        windowed_actual_size,
        windowed_actual_size as f64 / (1024.0 * 1024.0)
    );
    println!(
        "   Memory per point: {} bytes ({:.2} KB)",
        windowed_actual_size / points.len(),
        (windowed_actual_size / points.len()) as f64 / 1024.0
    );

    // Memory savings
    let savings_percent = (1.0 - (windowed_actual_size as f64 / shamir_actual_size as f64)) * 100.0;
    println!("   Memory savings vs Full Shamir: {:.1}%", savings_percent);

    // Summary
    println!("\n{}", "=".repeat(60));
    println!("Summary for {} points:", points.len());
    println!("  Online: 0 bytes");
    println!(
        "  Full Shamir: {} bytes ({:.2} MB)",
        shamir_actual_size,
        shamir_actual_size as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  2-bit Windowed: {} bytes ({:.2} MB)",
        windowed_actual_size,
        windowed_actual_size as f64 / (1024.0 * 1024.0)
    );
}
