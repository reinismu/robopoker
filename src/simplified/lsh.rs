use rand::Rng;

/// A hash function for LSH.
/// It projects a distribution into a binary vector using random projections.
pub struct HistogramLSH {
    num_dimensions: usize,
    num_bins: usize,
    random_vectors: Vec<Vec<i64>>,
}

impl HistogramLSH {
    /// Create a new LSH hash function.
    /// `dimensions`
    /// `num_bins` is the number of hash functions to use.
    pub fn new(dimensions: usize, num_bins: usize) -> Self {
        let mut rng = rand::thread_rng();
        let random_vectors: Vec<Vec<i64>> = (0..num_bins)
            .map(|_| (0..dimensions).map(|_| rng.gen_range(-1..=1)).collect())
            .collect();

        HistogramLSH {
            num_dimensions: dimensions,
            num_bins,
            random_vectors,
        }
    }
    
    /// Hash a distribution.
    pub fn hash(&self, distribution: &[u64]) -> u64 {
        let mut hash_value: u64 = 0;

        for (i, random_vector) in self.random_vectors.iter().enumerate() {
            let mut dot_product: i64 = 0;
            for (j, &count) in distribution.iter().enumerate().take(self.num_dimensions) {
                // Use the count directly as an integer weight
                dot_product += (count as i64) * random_vector[j];
            }

            // Set the i-th bit if the dot product is positive
            if dot_product > 0 {
                hash_value |= 1 << i;
            }
        }

        hash_value
    }

    pub fn bins(&self) -> usize {
        self.num_bins
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_histogram_lsh_new() {
        let lsh = HistogramLSH::new(50, 32);
        assert_eq!(lsh.num_dimensions, 50);
        assert_eq!(lsh.num_bins, 32);
        assert_eq!(lsh.random_vectors.len(), 32);
        assert_eq!(lsh.random_vectors[0].len(), 50);
    }

    #[test]
    fn test_histogram_lsh_hash_deterministic() {
        let mut rng = StdRng::seed_from_u64(42); // Use a seeded RNG for deterministic results
        let dimensions = 50;
        let num_bins = 32;

        // Create two identical LSH instances
        let lsh1 = HistogramLSH {
            num_dimensions: dimensions,
            num_bins,
            random_vectors: (0..num_bins)
                .map(|_| (0..dimensions).map(|_| rng.gen_range(-1..=1)).collect())
                .collect(),
        };

        let lsh2 = HistogramLSH {
            num_dimensions: dimensions,
            num_bins,
            random_vectors: lsh1.random_vectors.clone(),
        };

        // Create a sample distribution
        let distribution = vec![10, 20, 30, 40, 50];

        // Hash the distribution using both LSH instances
        let hash1 = lsh1.hash(&distribution);
        let hash2 = lsh2.hash(&distribution);

        // The hashes should be identical
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_histogram_lsh_hash_different_distributions() {
        let lsh = HistogramLSH::new(50, 32);

        let distribution1 = vec![10, 20, 30, 40, 50];

        let distribution2 = vec![50, 40, 30, 20, 10];

        let hash1 = lsh.hash(&distribution1);
        let hash2 = lsh.hash(&distribution2);

        // The hashes should be different for different distributions
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_histogram_lsh_hash_similar_distributions() {
        let lsh = HistogramLSH::new(50, 32);

        let distribution1 = vec![10, 20, 30, 40, 50];

        let distribution2 = vec![11, 21, 29, 39, 50]; // Slightly different from distribution1

        let hash1 = lsh.hash(&distribution1);
        let hash2 = lsh.hash(&distribution2);

        // Count the number of different bits
        let diff_bits = (hash1 ^ hash2).count_ones();

        // For similar distributions, we expect fewer different bits
        assert!(diff_bits < 16); // This threshold might need adjustment based on your specific use case
    }

    #[test]
    // TODO: not sure about this
    fn test_abs_from_more_similar_distributions_are_smaller() {
        let lsh = HistogramLSH::new(50, 65);

        let distribution1 = vec![10, 20, 30, 40, 50];

        let distribution2 = vec![15, 15, 30, 35, 50]; // Slightly different from distribution1

        let distribution3 = vec![50, 40, 30, 20, 10]; // Slightly different from distribution1

        let distribution4 = vec![50, 35, 30, 15, 15]; // Slightly different from distribution1

        let hash1 = lsh.hash(&distribution1);
        let hash2 = lsh.hash(&distribution2);
        let hash3 = lsh.hash(&distribution3);
        let hash4 = lsh.hash(&distribution4);

        let abs1 = (hash1 ^ hash2).count_ones();
        let abs2 = (hash1 ^ hash3).count_ones();
        let abs3 = (hash4 ^ hash3).count_ones();

        // For similar distributions, the number of set bits should be smaller
        assert!(abs1 < abs2);
    }
}