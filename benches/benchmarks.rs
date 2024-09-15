use criterion::measurement::WallTime;
use criterion::Throughput;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use robopoker::cards::deck::Deck;
use robopoker::cards::evaluator::Evaluator;
use robopoker::cards::hand::Hand;
use robopoker::cards::observation::Observation;
use robopoker::cards::street::Street;
use robopoker::cards::strength::Strength;
use robopoker::cfr::trainer::Trainer;

fn custom_criterion() -> Criterion<WallTime> {
    Criterion::default()
        .without_plots()
        .noise_threshold(0.5)
        .significance_level(0.01)
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(1))
}

fn benchmark_rps_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("RPS Training");
    group.throughput(Throughput::Elements(10_000));

    let mut trainer = Trainer::empty();
    group.bench_function(BenchmarkId::new("MCCFR", 10_000), |b| {
        b.iter(|| trainer.train(10_000))
    });
    group.finish();
}

fn benchmark_exhaustive_flops(c: &mut Criterion) {
    let mut group = c.benchmark_group("Exhaustive Flops");
    group.bench_function("Enumeration", |b| b.iter(|| Observation::all(Street::Flop)));
    group.finish();
}

fn benchmark_exhaustive_equity_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Equity Calculation");
    group.bench_function("RNG 7 card hand", |b| {
        b.iter(|| {
            let mut deck = Deck::new();
            let secret = Hand::from((0..2).map(|_| deck.draw()).collect::<Vec<_>>());
            let public = Hand::from((0..5).map(|_| deck.draw()).collect::<Vec<_>>());
            let observation = Observation::from((secret, public));
            observation.equity()
        })
    });
    group.finish();
}

fn benchmark_evaluator_7_card(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hand Evaluation");
    group.bench_function("RNG 7 card hand", |b| {
        b.iter(|| {
            let mut deck = Deck::new();
            let hand = Hand::from((0..7).map(|_| deck.draw()).collect::<Vec<_>>());
            let evaluator = Evaluator::from(hand);
            Strength::from(evaluator)
        })
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = custom_criterion();
    targets = benchmark_exhaustive_equity_calculation, benchmark_exhaustive_flops, benchmark_evaluator_7_card, benchmark_rps_training
}
criterion_main!(benches);
