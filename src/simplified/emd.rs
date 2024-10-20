use crate::clustering::histogram::Histogram;

#[derive(Debug)]
pub struct EMDContext {
    sorted_distances: Vec<Vec<f64>>,
    ordered_clusters: Vec<Vec<u8>>,
}

impl EMDContext {
    pub fn new() -> Self {
        let n = 51; // 0 to 50 inclusive
        let mut sorted_distances = vec![vec![0.0; n]; n];
        let mut ordered_clusters = vec![vec![0; n]; n];

        for i in 0..n {
            let mut distances: Vec<(u8, f64)> = (0..n as u8)
                .map(|j| (j, (i as f64 - j as f64).abs()))
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for (k, (j, dist)) in distances.into_iter().enumerate() {
                sorted_distances[i][k] = dist;
                ordered_clusters[i][k] = j;
            }
        }

        EMDContext {
            sorted_distances,
            ordered_clusters,
        }
    }

    pub fn approximate_emd(&self, point: &Histogram, mean: &Histogram) -> f64 {
        let mut targets: Vec<f64> = vec![0.0; 51];
        let mut mean_remaining: Vec<f64> = vec![0.0; 51];
        let point_total = point.norm as f64;
        let mean_total = mean.norm as f64;

        for (&k, &v) in &point.weights {
            targets[k.value() as usize] = v as f64 / point_total;
        }
        for (&k, &v) in &mean.weights {
            mean_remaining[k.value() as usize] = v as f64 / mean_total;
        }

        let mut total_cost = 0.0;
        let mut done = [false; 51];

        for i in 0..51 {
            for j in 0..51 {
                if done[j] {
                    continue;
                }

                let point_cluster = j;
                let mean_cluster = self.ordered_clusters[point_cluster][i] as usize;
                let amt_remaining = mean_remaining[mean_cluster];

                if amt_remaining == 0.0 {
                    continue;
                }

                let d = self.sorted_distances[point_cluster][i];

                if amt_remaining < targets[j] {
                    total_cost += amt_remaining * d;
                    targets[j] -= amt_remaining;
                    mean_remaining[mean_cluster] = 0.0;
                } else {
                    total_cost += targets[j] * d;
                    mean_remaining[mean_cluster] -= targets[j];
                    targets[j] = 0.0;
                    done[j] = true;
                }
            }
        }

        total_cost
    }
}

impl Default for EMDContext {
    fn default() -> Self {
        Self::new()
    }
}


#[cfg(test)]
mod tests {
    use crate::{cards::{hand::Hand, isomorphism::Isomorphism, observation::Observation}, clustering::{abstraction::Abstraction, datasets::ObservationSpace}, utils::persist::try_load};

    use super::*;

    #[test]
    #[ignore]
    fn test_emd() {
        let qq = Isomorphism::from(Observation::from((
            Hand::from("Qs Qh"),
            Hand::from(""),
        )));

        let kk = Isomorphism::from(Observation::from((
            Hand::from("Ks Kh"),
            Hand::from(""),
        )));

        let j6: Isomorphism = Isomorphism::from(Observation::from((
            Hand::from("Js 6h"),
            Hand::from(""),
        )));


        let path = std::path::Path::new("/home/detuks/Projects/poker/robopoker/cache/shortdeck-create_pref_observation_space.lz4");

        let space: ObservationSpace = try_load(path).unwrap();
        let h_qq = space.0.get(&qq).unwrap();
        let h_kk = space.0.get(&kk).unwrap();
        let h_j6 = space.0.get(&j6).unwrap();

        let emd_context = EMDContext::new();

        let emd_qq_kk = emd_context.approximate_emd(&h_qq, &h_kk);
        let emd_qq_j6 = emd_context.approximate_emd(&h_qq, &h_j6);

        assert!(emd_qq_j6 >  emd_qq_kk);
    } 
    
    #[test]
    #[ignore]
    fn test_emd2() {
        let emd_context = EMDContext::new();

        let h1 = Histogram::from(vec![Abstraction::Equity(0), Abstraction::Equity(1)]);
        let h2 = Histogram::from(vec![Abstraction::Equity(49), Abstraction::Equity(50)]);

        let emd = emd_context.approximate_emd(&h1, &h2);

        assert!(emd >  0.0);
    }
}