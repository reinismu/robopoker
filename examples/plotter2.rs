use plotters::prelude::*;
use robopoker::{cards::{hand::Hand, isomorphism::Isomorphism, observation::Observation}, clustering::{abstraction::Abstraction, abstractor::Abstractor, datasets::ObservationSpace, histogram::Histogram as CardsHistogram}, utils::persist::try_load};

fn main() {
    logging();
    let a = Isomorphism::from(Observation::from((
        Hand::from("Qs Qh"),
        Hand::from(""),
    )));

    log::info!("Finding histogram for {}", a);

    let path = std::path::Path::new("/home/detuks/Projects/poker/robopoker/cache/shortdeck-create_pref_observation_space.lz4");

    let space: ObservationSpace = try_load(path).unwrap();
    let histogram = space.0.get(&a).unwrap();

    plot_histogram(&a, &histogram);
}

fn logging() {
    use std::io::Write;
    use std::time::Instant;
    let start = Instant::now();
    env_logger::Builder::new()
        .filter(None, log::LevelFilter::Info)
        .format(move |buffer, record| {
            let elapsed = start.elapsed();
            writeln!(
                buffer,
                "{:02}:{:02}:{:02} - {}",
                (elapsed.as_secs() / 3600),
                (elapsed.as_secs() % 3600) / 60,
                (elapsed.as_secs() % 60),
                record.args()
            )
        })
        .init();
}

fn plot_histogram(isomorphism: &Isomorphism, histogram: &CardsHistogram) {
    //Create images folder if it doesn't exist
    std::fs::create_dir_all("images").unwrap();
    
    let path = format!("images/{}_equity.png", isomorphism);

    let root_drawing_area = BitMapBackend::new(path.as_str(), (1024, 768))
        .into_drawing_area();

    root_drawing_area.fill(&WHITE).unwrap();

    let equity_data: Vec<(i8, usize)> = histogram.weights().iter()
        .filter_map(|(abstraction, &count)| {
            if let Abstraction::Equity(value) = abstraction {
                Some((*value, count))
            } else {
                None
            }
        })
        .collect();

    let max_count = equity_data.iter().map(|&(_, count)| count).max().unwrap_or(1);

    let mut chart = ChartBuilder::on(&root_drawing_area)
        .caption(format!("Equity Histogram for {} equity {:.2}", isomorphism, histogram.equity()), ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0..51, 0..(max_count as u32))
        .unwrap();

    chart.configure_mesh()
        .x_desc("Equity")
        .y_desc("Count")
        .x_labels(10)
        .y_labels(10)
        .draw()
        .unwrap();

    chart
        .draw_series(
            Histogram::vertical(&chart)
                .style(BLUE.filled())
                .margin(0)
                .data(equity_data.iter().map(|&(value, count)| (value, count as u32)))
        )
        .unwrap()
        .label("Equity Distribution")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .unwrap();

    root_drawing_area.present().unwrap();

    println!("Equity histogram saved to {}", path);
}