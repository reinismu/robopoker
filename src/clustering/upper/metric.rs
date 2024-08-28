use super::histogram::Histogram;
use super::xor::Pair;
use crate::clustering::abstraction::Abstraction;
use std::collections::HashMap;

/// Trait for defining distance metrics between abstractions and histograms.
///
/// Calculating similarity between abstractions
/// and Earth Mover's Distance (EMD) between histograms. These metrics are
/// essential for clustering algorithms and comparing distributions.
pub trait Metric {
    fn emd(&self, x: &Histogram, y: &Histogram) -> f32;
    fn similarity(&self, x: &Abstraction, y: &Abstraction) -> f32;
}
impl Metric for HashMap<Pair, f32> {
    fn similarity(&self, x: &Abstraction, y: &Abstraction) -> f32 {
        let ref xor = Pair::from((x, y));
        self.get(xor).expect("precalculated distance").clone()
    }
    fn emd(&self, x: &Histogram, y: &Histogram) -> f32 {
        let n = x.domain().len();
        let m = y.domain().len();
        let mut cost = 0.0;
        let mut targets = x
            .domain()
            .iter()
            .map(|&a| (a, 1.0 / n as f32))
            .collect::<HashMap<&Abstraction, f32>>();
        let mut remains = y
            .domain()
            .iter()
            .map(|&a| (a, y.weight(a)))
            .collect::<HashMap<&Abstraction, f32>>();
        let mut removed = x
            .domain()
            .iter()
            .map(|&a| (a, false))
            .collect::<HashMap<&Abstraction, bool>>();
        for _ in 0..m {
            for supplier in x.domain() {
                if *removed
                    .get(supplier)
                    .expect("xabs not found in removed mass")
                {
                    continue;
                }
                let (ref neighbor, nearest) = y
                    .domain()
                    .iter()
                    .map(|candidate| (candidate.to_owned(), self.similarity(supplier, candidate)))
                    .min_by(|&(_, ref a), &(_, ref b)| a.partial_cmp(b).expect("not NaN"))
                    .expect("receiver domain is empty");
                let supply = remains
                    .get(neighbor)
                    .expect("yabs not found in remains mass")
                    .clone();
                if supply == 0.0 {
                    continue;
                }
                let target = targets
                    .get(supplier)
                    .expect("xabs not found in targets mass")
                    .clone();
                if supply < target {
                    cost += supply * nearest;
                    *targets
                        .get_mut(supplier)
                        .expect("xabs not found in targets mass") -= supply;
                    *remains
                        .get_mut(neighbor)
                        .expect("yabs not found in remains mass") = 0.0;
                } else {
                    cost += target * nearest;
                    *remains
                        .get_mut(neighbor)
                        .expect("yabs not found in remains mass") -= target;
                    *targets
                        .get_mut(supplier)
                        .expect("xabs not found in targets mass") = 0.0;
                    *removed
                        .get_mut(supplier)
                        .expect("xabs not found in removed mass") = true;
                }
            }
        }
        cost
    }
}
