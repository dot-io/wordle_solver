use std::fs::File;
use std::io::{BufRead, BufReader};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut charmap: HashMap<char, u32> = HashMap::new();
    let file = File::open("wordle-answers-alphabetical.txt")?;
    let reader = BufReader::new(file);

    let lines: Vec<String> = reader
        .lines()
        .collect::<Result<Vec<_>, _>>()?;

    for line in lines {
        for c in line.chars() {
            *charmap.entry(c).or_insert(0) += 1;
        }
    }

    let total: u32 = charmap.values().sum();
    let mut normalized: Vec<_> = charmap
        .into_iter()
        .map(|(c, count)| (c, count as f64 / total as f64))
        .collect();

    normalized.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Top 5 letter frequencies:");
    for (c, freq) in normalized.iter().take(5) {
        println!("'{}': {:.4}", c, freq);
    }

    Ok(())
}
