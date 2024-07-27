// use tokio::sync::Mutex;

use super::abstraction::Abstraction;
use super::histogram::Centroid;
use super::histogram::Histogram;
use super::observation::Observation;
use super::persistence::postgres::PostgresLookup;
use super::persistence::storage::Storage;
use super::xor::Pair;
use crate::cards::street::Street;
use std::collections::HashMap;
use std::time::Instant;
use std::vec;

pub struct Abstractor {
    storage: PostgresLookup,
    progress: Progress, /* s>> */
}

impl Abstractor {
    pub async fn new() -> Self {
        Self {
            progress: /* Arc::new(Mutex::new( */Progress::new()/* )) */,
            storage: PostgresLookup::new().await,
        }
    }

    pub async fn learn() {
        todo!("the whole thing across all layers streets")
    }

    async fn guesses(&self) -> Vec<Centroid> {
        todo!("implement k-means++ initialization")
    }

    /// Save the river
    ///
    pub async fn river(&mut self) -> &mut Self {
        // let mut handles = vec![];
        let rivers = Observation::all(Street::Rive);
        self.progress /* .lock().await */
            .reset();
        let chunk = rivers.len() / 4;
        for chunk in rivers.chunks(chunk) {
            // let mut storage = self.storage/* .clone() */;
            // let progress = self.progress/* .clone() */;
            // let future = async move {
            for river in chunk {
                let equity = river.equity();
                let bucket = equity * Abstraction::BUCKETS as f32;
                let abstraction = Abstraction::from(bucket as u64);
                self.storage.set_obs(river.clone(), abstraction).await;
                self.progress /* .lock().await */
                    .update();
            }
            // };
            // handles.push(tokio::task::spawn(future));
        }
        self
    }

    pub async fn cluster(&mut self, street: Street) -> &mut Self {
        assert!(street != Street::Rive);
        // maybe predecessors moves to Abstractor
        // this becomes wrapped in a loop over streets
        // for street in Street::iter() { match street { => Obs::preds(s) } }
        let ref possibilities = Observation::all(street);
        let ref mut neighbors = HashMap::<Observation, usize>::with_capacity(possibilities.len());
        let ref mut centroids = self.guesses().await;
        self.kmeans(centroids, neighbors, possibilities).await;
        self.upsert(centroids, neighbors).await;
        self.insert(centroids).await;
        self
    }

    async fn kmeans(
        &self,
        centroids: &mut Vec<Centroid>,
        neighbors: &mut HashMap<Observation, usize>,
        observations: &Vec<Observation>,
    ) {
        const ITERATIONS: usize = 100;
        for _ in 0..ITERATIONS {
            for obs in observations.iter() {
                let histogram = self.storage.get_histogram(obs.clone()).await;
                let ref x = histogram;
                let mut position = 0usize;
                let mut minimium = f32::MAX;
                for (i, centroid) in centroids.iter().enumerate() {
                    let y = centroid.histogram();
                    let emd = self.emd(x, y).await;
                    if emd < minimium {
                        position = i;
                        minimium = emd;
                    }
                }
                neighbors.insert(obs.clone(), position);
                centroids
                    .get_mut(position)
                    .expect("position in range")
                    .expand(histogram);
            }
        }
    }

    async fn upsert(&mut self, centroids: &[Centroid], neighbors: &HashMap<Observation, usize>) {
        for (observation, index) in neighbors.iter() {
            let centroid = centroids.get(*index).expect("index in range");
            let abs = centroid.signature();
            let obs = observation.clone();
            self.storage.set_obs(obs, abs).await;
        }
    }

    async fn insert(&mut self, centroids: &mut Vec<Centroid>) {
        for centroid in centroids.iter_mut() {
            centroid.shrink();
        }
        for (i, a) in centroids.iter().enumerate() {
            for (j, b) in centroids.iter().enumerate() {
                if i > j {
                    let x = a.signature();
                    let y = b.signature();
                    let xor = Pair::from((x, y));
                    let x = a.histogram();
                    let y = b.histogram();
                    let distance = self.emd(x, y).await;
                    self.storage.set_xor(xor, distance).await;
                }
            }
        }
    }

    /// Earth mover's distance using our precomputed distance metric.
    ///
    ///
    async fn emd(&self, this: &Histogram, that: &Histogram) -> f32 {
        let n = this.size();
        let m = that.size();
        let mut cost = 0.0;
        let mut extra = HashMap::new();
        let mut goals = vec![1.0 / n as f32; n];
        let mut empty = vec![false; n];
        for i in 0..m {
            for j in 0..n {
                if empty[j] {
                    continue;
                }
                let this_key = this.domain()[j];
                let that_key = that.domain()[i];
                let spill = extra
                    .get(that_key)
                    .cloned()
                    .or_else(|| Some(that.weight(that_key)))
                    .expect("key is somewhere");
                if spill == 0f32 {
                    continue;
                }
                let xor = Pair::from((*this_key, *that_key));
                let d = self.storage.get_xor(xor).await;
                let bonus = spill - goals[j];
                if (bonus) < 0f32 {
                    extra.insert(*that_key, 0f32);
                    cost += d * bonus as f32;
                    goals[j] -= bonus as f32;
                } else {
                    extra.insert(*that_key, bonus);
                    cost += d * goals[j];
                    goals[j] = 0.0;
                    empty[j] = true;
                }
            }
        }
        cost
    }
}

struct Progress {
    begin: Instant,
    check: Instant,
    complete: usize,
}

impl Progress {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            complete: 0,
            begin: now,
            check: now,
        }
    }

    fn update(&mut self) {
        self.complete += 1;
        if self.complete % 1_000 == 0 {
            let now = Instant::now();
            self.display();
            self.check = now;
        }
    }

    fn display(&self) {
        use std::io::Write;
        let now = Instant::now();
        let total_t = now.duration_since(self.begin);
        let check_t = now.duration_since(self.check);
        println!("\x1B[6F\x1B[2K{:10} Observations", self.complete);
        println!("\x1B[2K Elapsed: {:.0?}", total_t);
        println!("\x1B[2K Last 1k: {:.0?}", check_t);
        println!(
            "\x1B[2K Mean 1k: {:.0?}",
            (total_t / (self.complete / 1_000) as u32)
        );
        println!("\x1B[2K");
        std::io::stdout().flush().unwrap();
    }

    fn reset(&mut self) {
        let now = Instant::now();
        self.complete = 0;
        self.begin = now;
        self.check = now;
    }
}
