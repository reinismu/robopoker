use std::collections::BTreeMap;
use std::path::Path;

use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

use crate::utils::persist::persisted_function;
use crate::{
    cards::{isomorphism::Isomorphism, observation::Observation, street::Street},
    clustering::{
        abstraction::Abstraction, datasets::ObservationSpace, histogram::Histogram, progress,
    },
};

pub mod emd;
pub mod lsh;

pub fn calculate() {
    let turn_observation_space = persisted_function(
        "create_turn_observation_space",
        create_turn_observation_space,
    )
    .unwrap();

    log::info!("Turn space size {}", turn_observation_space.0.len());

    let flop_observation_space = persisted_function(
        "create_flop_observation_space",
        || create_flop_observation_space(&turn_observation_space),
    ).unwrap();

    log::info!("Flop space size {}", flop_observation_space.0.len());

    let pref_observation_space = persisted_function(
        "create_pref_observation_space",
        || create_pref_observation_space(&flop_observation_space),
    ).unwrap();

    log::info!("Preflop space size {}", pref_observation_space.0.len());
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
        .map(|isomorphism| (isomorphism, create_pref_histogram(&isomorphism, flop_observation_space)))
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
        .map(|isomorphism| (isomorphism, create_flop_histogram(&isomorphism, turn_observation_space)))
        .inspect(|_| progress.inc(1))
        .collect::<BTreeMap<Isomorphism, Histogram>>();

    ObservationSpace(space)
}

fn create_flop_histogram(flop_isomorphism: &Isomorphism, turn_observation_space: &ObservationSpace) -> Histogram {
    let obs = flop_isomorphism.0;

    obs.children()
        .map(|turn| turn_observation_space.0.get(&Isomorphism::from(turn)).unwrap().clone())
        .collect::<Vec<Histogram>>()
        .into()
}

fn create_pref_histogram(pref_isomorphism: &Isomorphism, flop_observation_space: &ObservationSpace) -> Histogram {
    let obs = pref_isomorphism.0;

    obs.children()
        .map(|flop| flop_observation_space.0.get(&Isomorphism::from(flop)).unwrap().clone())
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
