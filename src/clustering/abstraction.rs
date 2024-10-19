use serde::{Deserialize, Serialize};

use crate::cards::hole::Hole;
use crate::Probability;
use std::hash::Hash;
use std::u64;

/// Abstraction represents a lookup value for a given set of Observations.
///
/// - River: we use a i8 to represent the equity bucket, i.e. Equity(0) is the worst bucket, and Equity(50) is the best bucket.
/// - Pre-Flop: we do not use any abstraction, rather store the 169 strategically-unique hands as u64.
/// - Other Streets: we use a u64 to represent the hash signature of the centroid Histogram over lower layers of abstraction.
#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Abstraction {
    Equity(i8),   // river
    Random(u64),  // flop, turn
    Pocket(Hole), // preflop
}

impl Abstraction {
    pub const fn range() -> &'static [Self] {
        &Self::BUCKETS
    }
    pub fn random() -> Self {
        Self::Random(rand::random::<u64>())
    }
    fn quantize(p: Probability) -> i8 {
        (p * Probability::from(Self::N)).round() as i8
    }
    fn floatize(q: i8) -> Probability {
        Probability::from(q) / Probability::from(Self::N)
    }
    pub fn value(&self) -> i8 {
        match self {
            Self::Equity(n) => *n,
            Self::Random(n) => *n as i8,
            Self::Pocket(_) => unreachable!("no preflop into value"),
        }
    }
    const N: i8 = 50;
    const BUCKETS: [Self; Self::N as usize + 1] = Self::buckets();
    const fn buckets() -> [Self; Self::N as usize + 1] {
        let mut buckets = [Self::Equity(0); Self::N as usize + 1];
        let mut i = 0;
        while i <= Self::N {
            buckets[i as usize] = Self::Equity(i as i8);
            i += 1;
        }
        buckets
    }
}

/// probability isomorphism
///
/// for river, we use a i8 to represent the equity bucket,
/// i.e. Equity(0) is the 0% equity bucket,
/// and Equity(N) is the 100% equity bucket.
impl From<Probability> for Abstraction {
    fn from(p: Probability) -> Self {
        Self::Equity(Abstraction::quantize(p))
    }
}
impl From<Abstraction> for Probability {
    fn from(abstraction: Abstraction) -> Self {
        match abstraction {
            Abstraction::Equity(n) => Abstraction::floatize(n),
            Abstraction::Random(_) => unreachable!("no cluster into probability"),
            Abstraction::Pocket(_) => unreachable!("no preflop into probability"),
        }
    }
}

/// u64 isomorphism
///
/// conversion to u64 for SQL storage.
impl From<Abstraction> for u64 {
    fn from(a: Abstraction) -> Self {
        match a {
            Abstraction::Random(n) => n,
            Abstraction::Equity(_) => unreachable!("no equity into u64"),
            Abstraction::Pocket(_) => unreachable!("no preflop into u64"),
        }
    }
}
impl From<u64> for Abstraction {
    fn from(n: u64) -> Self {
        Self::Random(n)
    }
}

/// i64 isomorphism
///
/// conversion to i64 for SQL storage.
impl From<Abstraction> for i64 {
    fn from(abstraction: Abstraction) -> Self {
        u64::from(abstraction) as i64
    }
}
impl From<i64> for Abstraction {
    fn from(n: i64) -> Self {
        Self::Random(n as u64)
    }
}

/// lossless preflop abstraction
impl From<Hole> for Abstraction {
    fn from(hole: Hole) -> Self {
        Self::Pocket(hole)
    }
}

impl std::fmt::Display for Abstraction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Random(n) => write!(f, "{:016x}", n),
            Self::Equity(n) => write!(f, "unreachable ? Equity({})", n),
            Self::Pocket(h) => write!(f, "unreachable ? Pocket({})", h),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_quantize_inverse_floatize() {
        for p in (0..=100).map(|x| x as Probability / 100.0) {
            let q = Abstraction::quantize(p);
            let f = Abstraction::floatize(q);
            assert!((p - f).abs() < 1.0 / Abstraction::N as Probability);
        }
    }

    #[test]
    fn is_floatize_inverse_quantize() {
        for q in 0..=Abstraction::N {
            let p = Abstraction::floatize(q);
            let i = Abstraction::quantize(p);
            assert!(q == i);
        }
    }
}
