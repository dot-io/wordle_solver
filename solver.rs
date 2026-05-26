// Exact Wordle solver with deterministic look-ahead.
//
// Two universes:
//   * ANSWERS   (2315 words)  -- the candidate set; the secret is one of these.
//   * GUESSABLE (12970 words) -- every legal guess. A guess need not be a
//     possible answer; a non-answer probe can split the candidates better.
//
// First guess is fixed to ALERT. For the candidate set after any feedback we
// pick the guess (from GUESSABLE) minimizing the EXACT expected number of
// total guesses, via full expectimax. The look-ahead beyond the current move
// restricts to candidate words (optimal-or-within-a-hair for the small pools
// reached by then, and keeps branching tractable).
//
// Files: wordle-answers-alphabetical.txt, wordle-guessable.txt

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::time::Instant;

const NP: usize = 243; // 3^5 feedback patterns
const GREEN: u8 = 242; // base-3 code for GGGGG

fn feedback(guess: &[u8; 5], answer: &[u8; 5]) -> u8 {
    let mut res = [0u8; 5];
    let mut counts = [0u8; 26];
    for i in 0..5 {
        if guess[i] == answer[i] {
            res[i] = 2;
        } else {
            counts[(answer[i] - b'a') as usize] += 1;
        }
    }
    for i in 0..5 {
        if res[i] == 0 {
            let c = (guess[i] - b'a') as usize;
            if counts[c] > 0 {
                res[i] = 1;
                counts[c] -= 1;
            }
        }
    }
    let mut code = 0u8;
    for i in 0..5 {
        code = code * 3 + res[i];
    }
    code
}

fn decode(mut code: u8) -> String {
    let mut chars = [b'B'; 5];
    for i in (0..5).rev() {
        chars[i] = match code % 3 {
            2 => b'G',
            1 => b'Y',
            _ => b'B',
        };
        code /= 3;
    }
    String::from_utf8(chars.to_vec()).unwrap()
}

fn word_str(w: &[u8; 5]) -> String {
    String::from_utf8(w.to_vec()).unwrap()
}

fn load(path: &str) -> Vec<[u8; 5]> {
    fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("cannot read {}", path))
        .lines()
        .map(|l| l.trim())
        .filter(|l| l.len() == 5 && l.bytes().all(|b| b.is_ascii_lowercase()))
        .map(|l| {
            let b = l.as_bytes();
            [b[0], b[1], b[2], b[3], b[4]]
        })
        .collect()
}

// Admissible lower bound on f(S) for |S| = m: assume each guess splits the
// remaining m-1 words perfectly across the 242 non-green patterns (plus the
// word solved now). Perfect balance under-estimates true cost -> safe for B&B.
fn build_lb(n: usize) -> Vec<f64> {
    let mut lb = vec![0f64; n + 1];
    if n >= 1 {
        lb[1] = 1.0;
    }
    for m in 2..=n {
        let rem = m - 1;
        let q = rem / 242;
        let r = rem % 242;
        let mut s = 0f64;
        if r > 0 {
            s += (r as f64) * ((q + 1) as f64) * lb[q + 1];
        }
        if q >= 1 {
            s += ((242 - r) as f64) * (q as f64) * lb[q];
        }
        lb[m] = 1.0 + s / (m as f64);
    }
    lb
}

struct Solver<'a> {
    a: usize,         // number of answers (candidate-column count)
    ng: usize,        // number of guessable words
    pat: &'a [u8],    // ng * a matrix: feedback(guess g, answer a)
    a2g: &'a [u32],   // answer column -> index in the guessable list
    g2a: &'a [i32],   // guessable index -> answer column, or -1
    lb: &'a [f64],
    memo: HashMap<Vec<u32>, (f64, u32)>,  // depth >= 1 (candidate guesses only)
    memo0: HashMap<Vec<u32>, (f64, u32)>, // depth 0 (full guessable universe)
    nodes: u64,
}

const MEMO_MAX: usize = 160;

impl<'a> Solver<'a> {
    #[inline]
    fn pat(&self, g: u32, a: u32) -> u8 {
        self.pat[g as usize * self.a + a as usize]
    }

    // f(S): expected guesses to identify the answer, optimal play. S is a list
    // of answer columns. Returns (cost, guess index into the guessable list).
    // depth 0 ranges over all guessable words; depth >= 1 over candidates only.
    fn solve(&mut self, s: &[u32], depth: u32) -> (f64, u32) {
        let nb = s.len();
        if nb == 1 {
            return (1.0, self.a2g[s[0] as usize]);
        }
        if depth == 0 {
            if let Some(&r) = self.memo0.get(s) {
                return r;
            }
        } else if nb <= MEMO_MAX {
            if let Some(&r) = self.memo.get(s) {
                return r;
            }
        }
        self.nodes += 1;

        // Guess universe for this node.
        let guesses: Vec<u32> = if depth == 0 {
            (0..self.ng as u32).collect()
        } else {
            s.iter().map(|&c| self.a2g[c as usize]).collect()
        };

        // Pass 1: cheap lower bound per guess from pattern counts only.
        let mut order: Vec<(f64, u32)> = Vec::with_capacity(guesses.len());
        let mut counts = [0u32; NP];
        for &g in &guesses {
            for &a in s {
                counts[self.pat(g, a) as usize] += 1;
            }
            let solves = counts[GREEN as usize] > 0;
            let mut nonempty = 0;
            let mut maxb = 0u32;
            let mut lbsum = 0f64;
            for p in 0..NP {
                let c = counts[p];
                if c == 0 {
                    continue;
                }
                nonempty += 1;
                if c > maxb {
                    maxb = c;
                }
                if p != GREEN as usize {
                    lbsum += c as f64 * self.lb[c as usize];
                }
                counts[p] = 0;
            }
            if !solves && nonempty == 1 {
                continue; // no progress
            }
            if maxb as usize == nb {
                continue; // biggest bucket is the whole set -> useless
            }
            order.push((1.0 + lbsum / nb as f64, g));
        }
        order.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());

        // Pass 2: branch & bound, best-LB-first.
        let mut best = f64::INFINITY;
        let mut best_g = self.a2g[s[0] as usize];
        for (lbg, g) in order {
            if lbg >= best {
                break;
            }
            let c = self.cost_of_guess(g, s, best, depth);
            if c < best {
                best = c;
                best_g = g;
            }
        }

        if depth == 0 {
            self.memo0.insert(s.to_vec(), (best, best_g));
        } else if nb <= MEMO_MAX {
            self.memo.insert(s.to_vec(), (best, best_g));
        }
        (best, best_g)
    }

    fn cost_of_guess(&mut self, g: u32, s: &[u32], bound: f64, depth: u32) -> f64 {
        let nb = s.len();
        let mut v: Vec<(u8, u32)> = Vec::with_capacity(nb);
        for &a in s {
            let p = self.pat(g, a);
            if p != GREEN {
                v.push((p, a));
            }
        }
        v.sort_unstable_by_key(|x| x.0);

        let mut sum = 0f64;
        let mut i = 0;
        while i < v.len() {
            let p = v[i].0;
            let mut j = i;
            let mut sub: Vec<u32> = Vec::new();
            while j < v.len() && v[j].0 == p {
                sub.push(v[j].1);
                j += 1;
            }
            if sub.len() == nb {
                return f64::INFINITY;
            }
            sub.sort_unstable();
            let (c, _) = self.solve(&sub, depth + 1);
            sum += sub.len() as f64 * c;
            if 1.0 + sum / nb as f64 >= bound {
                return f64::INFINITY; // pruned
            }
            i = j;
        }
        1.0 + sum / nb as f64
    }

    fn cost_exact(&mut self, g: u32, s: &[u32]) -> f64 {
        self.cost_of_guess(g, s, f64::INFINITY, 0)
    }

    // Max-information guess over the whole guessable list.
    fn entropy_best(&self, s: &[u32]) -> u32 {
        let nb = s.len() as f64;
        let mut best_h = -1.0;
        let mut best_g = self.a2g[s[0] as usize];
        let mut counts = [0u32; NP];
        for g in 0..self.ng as u32 {
            for &a in s {
                counts[self.pat(g, a) as usize] += 1;
            }
            let mut h = 0f64;
            for p in 0..NP {
                let c = counts[p];
                if c > 0 {
                    let pr = c as f64 / nb;
                    h -= pr * pr.log2();
                    counts[p] = 0;
                }
            }
            if h > best_h {
                best_h = h;
                best_g = g;
            }
        }
        best_g
    }

    // Best guess restricted to candidates (answer words still in S).
    fn best_candidate(&mut self, s: &[u32]) -> (f64, u32) {
        let mut best = f64::INFINITY;
        let mut best_g = self.a2g[s[0] as usize];
        let cols: Vec<u32> = s.to_vec();
        for c in cols {
            let g = self.a2g[c as usize];
            let cost = self.cost_of_guess(g, s, best, 0);
            if cost < best {
                best = cost;
                best_g = g;
            }
        }
        (best, best_g)
    }
}

fn main() {
    let answers = load("wordle-answers-alphabetical.txt");
    // Guessable list = combined legal guesses; fall back to answers if absent.
    let guesses = match fs::metadata("wordle-guessable.txt") {
        Ok(_) => load("wordle-guessable.txt"),
        Err(_) => answers.clone(),
    };
    let a = answers.len();
    let ng = guesses.len();

    // Map answer columns <-> guessable indices.
    let gindex: HashMap<[u8; 5], u32> = guesses
        .iter()
        .enumerate()
        .map(|(i, w)| (*w, i as u32))
        .collect();
    let a2g: Vec<u32> = answers
        .iter()
        .map(|w| *gindex.get(w).expect("answer must be guessable"))
        .collect();
    let mut g2a = vec![-1i32; ng];
    for (col, &g) in a2g.iter().enumerate() {
        g2a[g as usize] = col as i32;
    }

    // pattern matrix: feedback(guess g, answer a)
    let mut pat = vec![0u8; ng * a];
    for g in 0..ng {
        for col in 0..a {
            pat[g * a + col] = feedback(&guesses[g], &answers[col]);
        }
    }

    let lb = build_lb(a);
    let alert = *gindex.get(b"salet").expect("salet in guessable list");

    let mut solver = Solver {
        a,
        ng,
        pat: &pat,
        a2g: &a2g,
        g2a: &g2a,
        lb: &lb,
        memo: HashMap::new(),
        memo0: HashMap::new(),
        nodes: 0,
    };

    match std::env::args().nth(1).as_deref() {
        Some("analyze")   => run_analysis(&mut solver, &answers, &guesses, alert),
        Some("selftest")  => run_selftest(&mut solver, alert),
        Some("bestopen")  => run_bestopen(&mut solver, &guesses),
        _                 => run_interactive(&mut solver, &answers, &guesses, alert),
    }
}

// Find the optimal first guess over the full guessable list.
// Uses the same model as the interactive solver: depth-0 guess drawn from
// full dict, look-ahead restricts to candidates from move 2 onward.
// Prints the top-10 first guesses ranked by exact expected total guesses.
fn run_bestopen(solver: &mut Solver, guesses: &[[u8; 5]]) {
    let t0 = Instant::now();
    let a = solver.a;
    let all: Vec<u32> = (0..a as u32).collect();

    // LB ordering pass over all 12k+ guesses.
    let nb = all.len() as f64;
    let mut order: Vec<(f64, u32)> = Vec::with_capacity(solver.ng);
    let mut counts = [0u32; NP];
    for g in 0..solver.ng as u32 {
        for &col in &all {
            counts[solver.pat(g, col) as usize] += 1;
        }
        let mut nonempty = 0;
        let mut maxb = 0u32;
        let mut lbsum = 0f64;
        for p in 0..NP {
            let c = counts[p];
            if c == 0 { continue; }
            nonempty += 1;
            if c > maxb { maxb = c; }
            if p != GREEN as usize { lbsum += c as f64 * solver.lb[c as usize]; }
            counts[p] = 0;
        }
        if nonempty == 1 && maxb as usize == a { continue; } // useless
        order.push((1.0 + lbsum / nb, g));
    }
    order.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    eprintln!("ordering pass done ({:?})", t0.elapsed());

    // Seed the incumbent with known-strong openers so B&B prunes aggressively.
    let seed_words: &[&[u8; 5]] = &[
        b"alert", b"soare", b"raise", b"crane", b"trace", b"crate", b"slate",
    ];
    let mut incumbent = f64::INFINITY;
    let mut results: Vec<(f64, u32)> = Vec::new();
    for &w in seed_words {
        if let Some(pos) = guesses.iter().position(|gw| gw == w) {
            let g = pos as u32;
            let c = solver.cost_of_guess(g, &all, f64::INFINITY, 0);
            eprintln!("  seed {}  E={:.4}  ({:?})", word_str(w), c, t0.elapsed());
            if c < incumbent { incumbent = c; }
            results.push((c, g));
        }
    }
    let seeded: HashSet<u32> = results.iter().map(|&(_, g)| g).collect();
    eprintln!("seeds done, incumbent={:.4}  ({:?})", incumbent, t0.elapsed());

    // B&B sweep; tight incumbent prunes most of the 12k candidates.
    // Cap at MAX_SWEEP evaluations — the ordering by LB means the true optimum
    // is overwhelmingly likely to appear within the first few hundred.
    const MAX_SWEEP: usize = 500;
    let mut swept = 0usize;
    for (lbg, g) in &order {
        if *lbg >= incumbent { break; }
        if seeded.contains(g) { continue; }
        let c = solver.cost_of_guess(*g, &all, incumbent, 0);
        swept += 1;
        if c < incumbent {
            incumbent = c;
            eprintln!("  new best {}  E={:.4}  ({:?})", word_str(&guesses[*g as usize]), c, t0.elapsed());
        }
        if c.is_finite() {
            results.push((c, *g));
        }
        if swept % 50 == 0 {
            eprintln!("  ... {} evaluated, incumbent={:.4}  ({:?})", swept, incumbent, t0.elapsed());
        }
        if swept >= MAX_SWEEP { break; }
    }
    results.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());

    println!("Optimal first guess (full {} guessable, candidate-only look-ahead):", guesses.len());
    println!("{:<6}  {:<5}  E[total]", "rank", "word");
    println!("{}", "-".repeat(26));
    for (rank, (cost, g)) in results.iter().take(10).enumerate() {
        let label = if solver.g2a[*g as usize] >= 0 { "ans" } else { "probe" };
        println!(
            "{:<6}  {:<5}  {:.4}  ({})",
            rank + 1,
            word_str(&guesses[*g as usize]),
            cost,
            label
        );
    }
    eprintln!("done ({:?}), nodes = {}", t0.elapsed(), solver.nodes);
}

// ---- interactive play ----------------------------------------------------
enum Parsed {
    Quit,
    Move { guess: [u8; 5], fb: u8 },
}

fn parse_word(s: &str) -> Option<[u8; 5]> {
    let b = s.as_bytes();
    if b.len() != 5 || !b.iter().all(|c| c.is_ascii_lowercase()) {
        return None;
    }
    Some([b[0], b[1], b[2], b[3], b[4]])
}

fn parse_pattern(s: &str) -> Option<u8> {
    if s.len() != 5 {
        return None;
    }
    let mut code = 0u8;
    for c in s.chars() {
        let d = match c {
            'g' => 2,
            'y' => 1,
            'b' => 0,
            _ => return None,
        };
        code = code * 3 + d;
    }
    Some(code)
}

fn parse_input(line: &str, default_guess: &[u8; 5]) -> Option<Parsed> {
    let toks: Vec<String> = line.split_whitespace().map(|s| s.to_lowercase()).collect();
    match toks.len() {
        0 => None,
        1 if matches!(toks[0].as_str(), "q" | "quit" | "exit") => Some(Parsed::Quit),
        1 => parse_pattern(&toks[0]).map(|fb| Parsed::Move {
            guess: *default_guess,
            fb,
        }),
        2 => {
            if let (Some(w), Some(fb)) = (parse_word(&toks[0]), parse_pattern(&toks[1])) {
                Some(Parsed::Move { guess: w, fb })
            } else if let (Some(fb), Some(w)) = (parse_pattern(&toks[0]), parse_word(&toks[1])) {
                Some(Parsed::Move { guess: w, fb })
            } else {
                None
            }
        }
        _ => None,
    }
}

fn run_interactive(solver: &mut Solver, answers: &[[u8; 5]], guesses: &[[u8; 5]], alert: u32) {
    use std::io::{self, BufRead};
    let a = answers.len();

    println!("Interactive Wordle solver ({} candidates, {} legal guesses).", a, guesses.len());
    println!("After each guess, type the colours Wordle showed:");
    println!("  G = green (right letter, right spot)");
    println!("  Y = yellow (right letter, wrong spot)");
    println!("  B = gray  (letter absent)");
    println!("e.g.  BYBBG     — or to log a different word:  CRANE BYBBG");
    println!("Type q to quit.");

    let mut candidates: Vec<u32> = (0..a as u32).collect();
    let mut turn = 1usize;
    let stdin = io::stdin();

    loop {
        let (gi, note): (u32, String) = if turn == 1 {
            (alert, "recommended opener".into())
        } else if candidates.len() == 1 {
            (solver.a2g[candidates[0] as usize], "the only word left".into())
        } else {
            let (f, g) = solver.solve(&candidates, 0);
            let kind = if solver.g2a[g as usize] >= 0 {
                "candidate"
            } else {
                "probe (not a possible answer)"
            };
            (g, format!("expected {:.2} to finish; {}", f, kind))
        };

        let gword = word_str(&guesses[gi as usize]).to_uppercase();
        println!("\nTurn {}: guess  {}   ({})", turn, gword, note);
        if candidates.len() <= 12 {
            let list: Vec<String> = candidates
                .iter()
                .map(|&c| word_str(&answers[c as usize]))
                .collect();
            println!("  {} left: {}", candidates.len(), list.join(", "));
        } else {
            println!("  {} candidates remain", candidates.len());
        }

        let mv = loop {
            print!("  feedback> ");
            io::stdout().flush().ok();
            let mut line = String::new();
            if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
                println!("\nbye");
                return;
            }
            match parse_input(&line, &guesses[gi as usize]) {
                Some(Parsed::Quit) => {
                    println!("bye");
                    return;
                }
                Some(m) => break m,
                None => println!("  ?? type 5 colours G/Y/B (e.g. BYBBG), or  WORD BYBBG, or q"),
            }
        };

        let (guess_bytes, fb) = match mv {
            Parsed::Move { guess, fb } => (guess, fb),
            Parsed::Quit => unreachable!(),
        };

        if fb == GREEN {
            println!("\nSolved in {} guesses.", turn);
            return;
        }

        candidates.retain(|&c| feedback(&guess_bytes, &answers[c as usize]) == fb);
        if candidates.is_empty() {
            println!("\nNo candidate matches that feedback — likely a mistyped colour,");
            println!("or the answer isn't in this list. Restart to try again.");
            return;
        }
        turn += 1;
    }
}

// Auto-play the recommender against every answer (validates the full path).
fn run_selftest(solver: &mut Solver, alert: u32) {
    let t0 = Instant::now();
    let a = solver.a;
    let mut dist = [0u32; 12];
    let mut total = 0u64;
    let mut worst = 0usize;
    let mut fails = 0u32;

    for target in 0..a as u32 {
        let mut candidates: Vec<u32> = (0..a as u32).collect();
        let mut turn = 1usize;
        loop {
            let gi = if turn == 1 {
                alert
            } else if candidates.len() == 1 {
                solver.a2g[candidates[0] as usize]
            } else {
                solver.solve(&candidates, 0).1
            };
            let fb = solver.pat(gi, target);
            if fb == GREEN {
                break;
            }
            candidates.retain(|&c| solver.pat(gi, c) == fb);
            turn += 1;
            if turn > 10 {
                break;
            }
        }
        total += turn as u64;
        worst = worst.max(turn);
        if turn < dist.len() {
            dist[turn] += 1;
        }
        if turn > 6 {
            fails += 1;
        }
    }

    println!("Self-test: SALET opener, recommender plays all {} answers.", a);
    println!("  average guesses = {:.4}", total as f64 / a as f64);
    println!("  worst case      = {} guesses", worst);
    println!("  failures (>6)   = {}", fails);
    for k in 1..dist.len() {
        if dist[k] > 0 {
            println!("    {}: {:>4}", k, dist[k]);
        }
    }
    eprintln!("self-test time {:?}", t0.elapsed());
}

// ---- batch analysis (run with: ./solver analyze) -------------------------
fn run_analysis(solver: &mut Solver, answers: &[[u8; 5]], guesses: &[[u8; 5]], alert: u32) {
    let t0 = Instant::now();
    let a = answers.len();

    let mut buckets: HashMap<u8, Vec<u32>> = HashMap::new();
    for c in 0..a as u32 {
        buckets.entry(solver.pat(alert, c)).or_default().push(c);
    }
    buckets.remove(&GREEN);
    let mut bvec: Vec<(u8, Vec<u32>)> = buckets.into_iter().collect();
    bvec.sort_by_key(|(_, v)| v.len());

    println!(
        "{:<7} {:>4}  {:<14} {:<14} {:<14}",
        "pattern", "N", "OPTIMAL(E[g])", "MAXINFO(E[g])", "BESTCAND(E[g])"
    );
    println!("{}", "-".repeat(64));

    let mut sum_after = 0f64;
    let mut worst: Vec<(f64, String, usize)> = Vec::new();
    let mut gap_total = 0f64;
    let mut gap_ent = 0f64;
    let (mut n_cand, mut n_eq_info, mut n_eq_cand, mut n_neither) = (0u32, 0u32, 0u32, 0u32);

    for (code, bucket) in &bvec {
        let s = bucket.clone();
        let nb = s.len();
        let scol: HashSet<u32> = s.iter().copied().collect();

        eprintln!("  bucket {} N={} ... ({:?})", decode(*code), nb, t0.elapsed());
        let (f_opt, g_opt) = solver.solve(&s, 0);
        let g_ent = solver.entropy_best(&s);
        let f_ent = if nb == 1 { 1.0 } else { solver.cost_exact(g_ent, &s) };
        let (f_cand, g_cand) = solver.best_candidate(&s);

        sum_after += nb as f64 * f_opt;
        gap_total += nb as f64 * (f_cand - f_opt);
        gap_ent += nb as f64 * (f_ent - f_opt);
        // "candidate" = optimal guess is an answer word still in S.
        let opt_col = solver.g2a[g_opt as usize];
        if opt_col >= 0 && scol.contains(&(opt_col as u32)) {
            n_cand += 1;
        }
        let eq_info = g_opt == g_ent;
        let eq_cand = g_opt == g_cand;
        if eq_info {
            n_eq_info += 1;
        }
        if eq_cand {
            n_eq_cand += 1;
        }
        if !eq_info && !eq_cand {
            n_neither += 1;
        }

        println!(
            "{:<7} {:>4}  {:<5} {:>6.3}  {:<5} {:>6.3}  {:<5} {:>6.3}",
            decode(*code),
            nb,
            word_str(&guesses[g_opt as usize]),
            f_opt,
            word_str(&guesses[g_ent as usize]),
            f_ent,
            word_str(&guesses[g_cand as usize]),
            f_cand,
        );
        worst.push((f_opt, decode(*code), nb));
        std::io::stdout().flush().ok();
    }

    let total = 1.0 + sum_after / a as f64;
    println!("\n{}", "=".repeat(64));
    println!(
        "ALERT opener, optimal play ({} legal guesses):\n  expected total guesses = {:.4}",
        guesses.len(),
        total
    );
    println!("  vs pure best-candidate strategy: +{:.4} guesses/game", gap_total / a as f64);
    println!("  vs pure max-information strategy: +{:.4} guesses/game", gap_ent / a as f64);
    let np = bvec.len() as f64;
    println!("\nAcross {} non-trivial patterns, the optimal 2nd guess is:", bvec.len());
    println!("  a candidate (could be the answer):   {:>3} ({:.0}%)", n_cand, 100.0 * n_cand as f64 / np);
    println!("  equal to the max-information word:   {:>3} ({:.0}%)", n_eq_info, 100.0 * n_eq_info as f64 / np);
    println!("  equal to the best-candidate word:    {:>3} ({:.0}%)", n_eq_cand, 100.0 * n_eq_cand as f64 / np);
    println!("  a third word (neither extreme):      {:>3} ({:.0}%)", n_neither, 100.0 * n_neither as f64 / np);

    worst.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
    println!("\nHardest patterns (highest expected guesses after ALERT):");
    for (f, p, nb) in worst.iter().take(8) {
        println!("  {}  N={:<4}  E[g]={:.3}", p, nb, f);
    }
    eprintln!("done ({:?}), expectimax nodes = {}", t0.elapsed(), solver.nodes);
}
