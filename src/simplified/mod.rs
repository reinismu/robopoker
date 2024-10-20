#![feature(portable_simd)]
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::iter::Sum;
use std::path::Path;

use kmeans::HistogramDistance;
use kmeans::KMeans;
use kmeans::KMeansConfig;
use lsh_rs::data::Numeric;
use lsh_rs::prelude::LshMem;
use rand::seq::IteratorRandom;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

use crate::cards::hand::Hand;
use crate::utils::persist::persisted_function;
use crate::{
    cards::{isomorphism::Isomorphism, observation::Observation, street::Street},
    clustering::{
        abstraction::Abstraction, datasets::ObservationSpace, histogram::Histogram, progress,
    },
};

pub mod emd;
pub mod lsh;

pub fn kmeans() {
    let turn_observation_space = persisted_function(
        "create_turn_observation_space",
        create_turn_observation_space,
    )
    .unwrap();


    log::info!(
        "Turn space size = {}",
        turn_observation_space.0.len()
    );

    let turn_observation_space_clusters =
        persisted_function("cluster_turn_observation_space_500_300", || {
            cluster_turn_observation_space(&turn_observation_space, 500, 300)
        })
        .unwrap();

    // print clusters
    for cluster in turn_observation_space_clusters.iter() {
        let first = cluster.first().unwrap();
        log::info!(
            "Cluster ({:.2}) {}: {:?}",
            turn_observation_space.0.get(first).unwrap().equity(),
            first,
            cluster.len()
        );
    }

    // Print first cluster random 20 samples
    let first_cluster = turn_observation_space_clusters.first().unwrap();
    let mut rng = thread_rng();
    let samples = first_cluster.choose_multiple(&mut rng, 20);

    samples.for_each(|i| {
        log::info!(
            "{}: {:.2}",
            i,
            turn_observation_space.0.get(i).unwrap().equity()
        );
    });

    // println!("Centroids {}: {:?}", result.centroids.len(), result.centroids);
    // // println!("Cluster-Assignments: {:?}", result.assignments);
    // println!("Error: {}", result.distsum);
}

pub fn calculate() {
    let turn_observation_space = persisted_function(
        "create_turn_observation_space",
        create_turn_observation_space,
    )
    .unwrap();

    let turn_r = estimate_obserbation_space_r(&turn_observation_space, 1000) * 2.;

    log::info!(
        "Turn space size = {}, L2 r = {}",
        turn_observation_space.0.len(),
        turn_r
    );

    let progress = create_progress(turn_observation_space.0.len());

    let mut lsh: lsh_rs::prelude::LSH<
        lsh_rs::prelude::L2<f32, i8>,
        f32,
        lsh_rs::MemoryTable<f32, i8>,
        i8,
    > = LshMem::new(8, 20, 51).seed(1).l2(turn_r).unwrap();

    let lsh_map = turn_observation_space
        .0
        .iter()
        .map(|(i, h)| (i, lsh.store_vec(&h.raw_distribution()).unwrap()))
        .inspect(|_| progress.inc(1))
        .fold(HashMap::new(), |mut acc, (hash, idx)| {
            acc.entry(idx).or_insert_with(Vec::new).push(hash);
            acc
        });

    log::info!(
        "LSH Map size: {} LSH = {}",
        lsh_map.len(),
        lsh.describe().unwrap()
    );

    let qq_turn = Isomorphism::from(Observation::from((
        Hand::from("As Ah"),
        Hand::from("Ad Ac Td"),
    )));

    let qq_histogram = turn_observation_space.0.get(&qq_turn).unwrap();

    let buckets = lsh
        .query_bucket_ids(&qq_histogram.raw_distribution())
        .unwrap();

    log::info!("Found {} similar hands for {}", buckets.len(), qq_turn);
    // print similar hands
    for bucket in buckets.iter().take(10) {
        let isomorphisms = lsh_map.get(&bucket).unwrap();
        for isomorphism in isomorphisms {
            log::info!(
                "Isomorphism: {} equity: {}",
                isomorphism,
                isomorphism.0.equity()
            );
        }
    }

    // lsh.store_vecs(&vecs)

    // lsh.store_vec(vec![1.0, 2.0, 3.0]).unwrap());

    // let flop_observation_space = persisted_function(
    //     "create_flop_observation_space",
    //     || create_flop_observation_space(&turn_observation_space),
    // ).unwrap();

    // log::info!("Flop space size {}", flop_observation_space.0.len());

    // let pref_observation_space = persisted_function(
    //     "create_pref_observation_space",
    //     || create_pref_observation_space(&flop_observation_space),
    // ).unwrap();

    // log::info!("Preflop space size {}", pref_observation_space.0.len());
}

fn cluster_turn_observation_space(
    turn_observation_space: &ObservationSpace,
    k: usize,
    max_iter: usize,
) -> Vec<Vec<Isomorphism>> {
    let samples = turn_observation_space.0.iter().collect::<Vec<_>>();
    let dim_samples = samples
        .iter()
        .flat_map(|(_, h)| h.raw_distribution())
        .collect::<Vec<_>>();

    // Calculate kmeans, using kmean++ as initialization-method
    // KMeans<_, 8> specifies to use f64 SIMD vectors with 8 lanes (e.g. AVX512)
    let kmean: KMeans<_, 8, _> = KMeans::new(dim_samples, samples.len(), 51, HistogramDistance);

    // config with callback
    let conf = KMeansConfig::build()
        .init_done(&|_| log::info!("Turn Kmeans Initialization completed."))
        .iteration_done(&|s, nr, new_distsum| {
            log::info!(
                "Iteration {} - Error: {:.2} -> {:.2} | Improvement: {:.2}",
                nr,
                s.distsum,
                new_distsum,
                s.distsum - new_distsum
            )
        })
        .abort_strategy(kmeans::AbortStrategy::NoImprovementForXIterations {
            x: 40,
            threshold: 10.,
            abort_on_negative: false,
        })
        .build();

    log::info!("Starting KMeans clustering");
    let result = kmean.kmeans_minibatch(samples.len() / 10, k, max_iter, KMeans::init_kmeanplusplus, &conf);

    let assignements = result.assignments;

    let clusters =
        assignements
            .iter()
            .enumerate()
            .fold(HashMap::new(), |mut acc, (idx, cluster)| {
                acc.entry(*cluster)
                    .or_insert_with(Vec::new)
                    .push(samples[idx].0.clone());
                acc
            });

    clusters.into_iter().map(|(_, v)| v).collect()
}

fn create_turn_observation_space() -> ObservationSpace {
    log::info!("creating turn observation space");
    let isomorphisms = Observation::exhaust(Street::Turn)
        .filter(Isomorphism::is_canonical)
        .map(Isomorphism::from)
        .collect::<Vec<Isomorphism>>();
    let progress = create_progress(isomorphisms.len());

    let space = isomorphisms
        .into_par_iter()
        .map(|isomorphism| (isomorphism, create_turn_histogram(&isomorphism)))
        .inspect(|_| progress.inc(1))
        .collect::<BTreeMap<Isomorphism, Histogram>>();

    ObservationSpace(space)
}

fn create_turn_histogram(turn_isomorphism: &Isomorphism) -> Histogram {
    let obs = turn_isomorphism.0;

    obs.children()
        .map(|river| Abstraction::from(river.equity()))
        .collect::<Vec<Abstraction>>()
        .into()
}

fn create_pref_observation_space(flop_observation_space: &ObservationSpace) -> ObservationSpace {
    log::info!("creating flop observation space");
    let isomorphisms = Observation::exhaust(Street::Pref)
        .filter(Isomorphism::is_canonical)
        .map(Isomorphism::from)
        .collect::<Vec<Isomorphism>>();
    let progress = create_progress(isomorphisms.len());

    let space = isomorphisms
        .into_par_iter()
        .map(|isomorphism| {
            (
                isomorphism,
                create_pref_histogram(&isomorphism, flop_observation_space),
            )
        })
        .inspect(|_| progress.inc(1))
        .collect::<BTreeMap<Isomorphism, Histogram>>();

    ObservationSpace(space)
}

fn create_flop_observation_space(turn_observation_space: &ObservationSpace) -> ObservationSpace {
    log::info!("creating flop observation space");
    let isomorphisms = Observation::exhaust(Street::Flop)
        .filter(Isomorphism::is_canonical)
        .map(Isomorphism::from)
        .collect::<Vec<Isomorphism>>();
    let progress = create_progress(isomorphisms.len());

    let space = isomorphisms
        .into_par_iter()
        .map(|isomorphism| {
            (
                isomorphism,
                create_flop_histogram(&isomorphism, turn_observation_space),
            )
        })
        .inspect(|_| progress.inc(1))
        .collect::<BTreeMap<Isomorphism, Histogram>>();

    ObservationSpace(space)
}

fn create_flop_histogram(
    flop_isomorphism: &Isomorphism,
    turn_observation_space: &ObservationSpace,
) -> Histogram {
    let obs = flop_isomorphism.0;

    obs.children()
        .map(|turn| {
            turn_observation_space
                .0
                .get(&Isomorphism::from(turn))
                .unwrap()
                .clone()
        })
        .collect::<Vec<Histogram>>()
        .into()
}

fn create_pref_histogram(
    pref_isomorphism: &Isomorphism,
    flop_observation_space: &ObservationSpace,
) -> Histogram {
    let obs = pref_isomorphism.0;

    obs.children()
        .map(|flop| {
            flop_observation_space
                .0
                .get(&Isomorphism::from(flop))
                .unwrap()
                .clone()
        })
        .collect::<Vec<Histogram>>()
        .into()
}

fn create_progress(n: usize) -> indicatif::ProgressBar {
    let tick = std::time::Duration::from_secs(1);
    let style = "[{elapsed}] {spinner} {wide_bar} ETA {eta}";
    let style = indicatif::ProgressStyle::with_template(style).unwrap();
    let progress = indicatif::ProgressBar::new(n as u64);
    progress.set_style(style);
    progress.enable_steady_tick(tick);
    progress
}

fn estimate_obserbation_space_r(obserbation_space: &ObservationSpace, sample_size: usize) -> f32 {
    let mut rng = thread_rng();
    let histograms = obserbation_space
        .0
        .values()
        .choose_multiple(&mut rng, sample_size)
        .iter()
        .map(|h| h.raw_distribution())
        .collect();
    let r = estimate_r(&histograms);
    r as f32
}

fn estimate_r<T>(sample: &Vec<Vec<T>>) -> T
where
    T: Numeric + Sum + PartialOrd,
{
    let mut total_distance = T::zero();
    let mut count = T::zero();

    for i in 0..sample.len() {
        for j in (i + 1)..sample.len() {
            let distance = l2_distance(&sample[i], &sample[j]);
            total_distance = total_distance + distance;
            count = count + T::one();
        }
    }

    let avg_distance = total_distance / count;
    avg_distance / (T::one() + T::one() + T::one() + T::one()) // Dividing by 4
}

fn l2_distance<T>(v1: &[T], v2: &[T]) -> T
where
    T: Numeric + Sum + PartialOrd,
{
    approximate_sqrt(
        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| {
                let diff = if a > b { *a - *b } else { *b - *a };
                diff * diff
            })
            .sum::<T>(),
    )
}

fn approximate_sqrt<T>(x: T) -> T
where
    T: Numeric,
{
    if x == T::zero() {
        return T::zero();
    }

    let mut z = x;
    let one = T::one();
    let two = one + one;

    for _ in 0..10 {
        // 10 iterations should be sufficient for most cases
        z = (z + x / z) / two;
    }

    z
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_r() {
        let histograms = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let r = estimate_r(&histograms);
        assert!(((r - 1.0) as f64).abs() < 1.0);
    }

    #[test]
    fn test_l2_distance() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![4.0, 5.0, 6.0];
        let distance = l2_distance(&v1, &v2);
        assert_eq!(distance, 5.196152422706632);
    }

    #[test]
    fn test_approximate_sqrt() {
        let x = 9.0;
        let sqrt = approximate_sqrt(x);
        assert_eq!(sqrt, 3.0);
    }
}
