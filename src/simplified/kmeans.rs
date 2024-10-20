

pub fn kmeans<T>(values: &[T], custer_num: usize, distance_func: F) -> Vec<T>
where
    F: Fn(&T, &T) -> f32,
{
    let mut rng = thread_rng();
    let mut centroids = values
        .choose_multiple(&mut rng, 3)
        .iter()
        .map(|v| v.clone())
        .collect::<Vec<T>>();

    let mut clusters = vec![Vec::new(); centroids.len()];

    loop {
        clusters.iter_mut().for_each(|c| c.clear());

        for v in values {
            let mut min_distance = f32::INFINITY;
            let mut min_centroid = 0;

            for (i, c) in centroids.iter().enumerate() {
                let distance = distance_func(v, c);
                if distance < min_distance {
                    min_distance = distance;
                    min_centroid = i;
                }
            }

            clusters[min_centroid].push(v.clone());
        }

        let new_centroids = clusters
            .iter()
            .map(|c| {
                let mut centroid = c[0].clone();
                for v in c.iter().skip(1) {
                    centroid = centroid + v;
                }
                centroid / c.len()
            })
            .collect::<Vec<T>>();

        if new_centroids == centroids {
            break;
        }

        centroids = new_centroids;
    }

    centroids
}