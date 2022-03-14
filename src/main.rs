use std::io::Write;

use dotenv::dotenv;
use log::{info, warn};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const WIDTH: usize = 20;
const HEIGHT: usize = 20;
const PPM_SCALER: usize = 25;
const PPM_RANGE: f64 = 10f64;
const PPM_COLOR_INTENSITY: f64 = 255f64;
const SAMPLE_SIZE: usize = 75;
const BIAS: f64 = 20f64;
const DATA_FOLDER: &str = "data";
const TRAIN_PASSES: usize = 2000;
const TRAIN_SEED: u64 = 69;
const CHECK_SEED: u64 = 420;

type Layer = [[f64; HEIGHT]; WIDTH];

#[inline]
fn clampi(x: i32, low: i32, hight: i32) -> i32 {
    if x < low {
        low
    } else if x > hight {
        hight
    } else {
        x
    }
}

fn layer_fill_rect(
    layer: &mut Layer,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    value: f64,
) {
    assert!(w > 0);
    assert!(h > 0);
    let x0 = clampi(x, 0, (WIDTH - 1) as i32);
    let y0 = clampi(y, 0, (HEIGHT - 1) as i32);
    let x1 = clampi(x0 + w - 1, 0, (WIDTH - 1) as i32);
    let y1 = clampi(y0 + w - 1, 0, (HEIGHT - 1) as i32);

    for y in y0..=y1 {
        for x in x0..=x1 {
            layer[y as usize][x as usize] = value;
        }
    }
}

fn layer_fill_circle(layer: &mut Layer, cx: i32, cy: i32, r: i32, value: f64) {
    assert!(r > 0);
    let x0 = clampi(cx - r, 0, (WIDTH - 1) as i32);
    let y0 = clampi(cy - r, 0, (HEIGHT - 1) as i32);
    let x1 = clampi(cx + r, 0, (WIDTH - 1) as i32);
    let y1 = clampi(cy + r, 0, (HEIGHT - 1) as i32);

    for y in y0..=y1 {
        for x in x0..=x1 {
            let dx = x - cx;
            let dy = y - cy;
            if dx * dx + dy * dy <= r * r {
                layer[y as usize][x as usize] = value;
            }
        }
    }
}

fn layer_save_as_ppm(layer: &Layer, file_path: &str) -> std::io::Result<()> {
    let mut f = std::fs::File::create(file_path)?;

    write!(
        &mut f,
        "P6\n{} {} 255\n",
        WIDTH * PPM_SCALER,
        HEIGHT * PPM_SCALER
    )?;

    for y in 0..HEIGHT * PPM_SCALER {
        for x in 0..WIDTH * PPM_SCALER {
            let s: f64 = (layer[y / PPM_SCALER][x / PPM_SCALER] + PPM_RANGE)
                / (2f64 * PPM_RANGE);
            let pixel: [u8; 3] = [
                (PPM_COLOR_INTENSITY * (1f64 - s)).floor() as u8,
                (PPM_COLOR_INTENSITY * (1f64 - s)).floor() as u8,
                (PPM_COLOR_INTENSITY * s).floor() as u8,
            ];
            f.write_all(&pixel)?;
        }
    }

    Ok(())
}

fn feed_forward(inputs: &Layer, weights: &Layer) -> f64 {
    let mut ouput: f64 = 0f64;

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            ouput += inputs[y][x] * weights[y][x];
        }
    }

    ouput
}

fn add_inputs_from_weights(inputs: &Layer, weights: &mut Layer) {
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            weights[y][x] += inputs[y][x]
        }
    }
}

fn sub_inputs_from_weights(inputs: &Layer, weights: &mut Layer) {
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            weights[y][x] -= inputs[y][x]
        }
    }
}

fn layer_random_rect(layer: &mut Layer, rng: &mut ChaCha8Rng) {
    layer_fill_rect(layer, 0, 0, WIDTH as i32, HEIGHT as i32, 0f64);
    let x = rng.gen_range(0..WIDTH);
    let y = rng.gen_range(0..HEIGHT);

    let mut w = WIDTH - x;
    if w < 2 {
        w = 2
    }
    w = rng.gen_range(1..w);

    let mut h = HEIGHT - x;
    if h < 2 {
        h = 2
    }
    h = rng.gen_range(1..h);

    layer_fill_rect(layer, x as i32, y as i32, w as i32, h as i32, 1f64);
}

fn layer_random_circle(layer: &mut Layer, rng: &mut ChaCha8Rng) {
    layer_fill_rect(layer, 0, 0, WIDTH as i32, HEIGHT as i32, 0f64);
    let cx: i32 = rng.gen_range(0..WIDTH).try_into().unwrap();
    let cy = rng.gen_range(0..HEIGHT).try_into().unwrap();
    let mut r = i32::MAX;
    if r > cx {
        r = cx as i32;
    }
    if r > cy {
        r = cy as i32;
    }
    if r > WIDTH as i32 - cx {
        r = WIDTH as i32 - cx;
    }
    if r > HEIGHT as i32 - cy {
        r = HEIGHT as i32 - cy;
    }
    if r < 2 {
        r = 2;
    }
    r = rng.gen_range(1..r);
    layer_fill_circle(layer, cx as i32, cy as i32, r, 1f64);
}

fn train_pass(
    inputs: &mut Layer,
    weights: &mut Layer,
    rng: &mut ChaCha8Rng,
) -> std::io::Result<i32> {
    let mut count: usize = 0;
    let mut adjusted: i32 = 0;

    for _ in 0..SAMPLE_SIZE {
        layer_random_rect(inputs, rng);
        if feed_forward(inputs, weights) > BIAS {
            sub_inputs_from_weights(inputs, weights);
            let file_path =
                format!("{}/weights-{:0>3}.ppm", DATA_FOLDER, count);
            count += 1;
            info!("saving: {}", &file_path);
            layer_save_as_ppm(weights, &file_path)?;
            adjusted += 1;
        }

        layer_random_circle(inputs, rng);
        if feed_forward(inputs, weights) < BIAS {
            add_inputs_from_weights(inputs, weights);
            let file_path =
                format!("{}/weights-{:0>3}.ppm", DATA_FOLDER, count);
            count += 1;
            info!("saving: {}", &file_path);
            layer_save_as_ppm(weights, &file_path)?;
            adjusted += 1;
        }
    }

    Ok(adjusted)
}

fn check_pass(
    inputs: &mut Layer,
    weights: &mut Layer,
    rng: &mut ChaCha8Rng,
) -> i32 {
    let mut adjusted: i32 = 0;

    for _ in 0..SAMPLE_SIZE {
        layer_random_rect(inputs, rng);
        if feed_forward(inputs, weights) > BIAS {
            adjusted += 1;
        }

        layer_random_circle(inputs, rng);
        if feed_forward(inputs, weights) < BIAS {
            adjusted -= 1;
        }
    }
    adjusted
}

fn main() -> std::io::Result<()> {
    dotenv().ok();
    pretty_env_logger::init();

    let mut inputs: Layer = [[0f64; HEIGHT]; WIDTH];
    let mut weights: Layer = [[0f64; HEIGHT]; WIDTH];

    std::fs::create_dir_all(DATA_FOLDER)?;

    let mut rng = ChaCha8Rng::seed_from_u64(CHECK_SEED);
    let adj = check_pass(&mut inputs, &mut weights, &mut rng);
    warn!(
        "fail rate of untrained model is {}",
        adj as f64 / (SAMPLE_SIZE as f64 * 2f64),
    );

    for i in 0..TRAIN_PASSES {
        let mut rng = ChaCha8Rng::seed_from_u64(TRAIN_SEED);
        let adj = train_pass(&mut inputs, &mut weights, &mut rng)?;
        info!("Pass: {}: adjusted {} times", i, adj);
        if adj <= 0 {
            break;
        }
    }

    let mut rng = ChaCha8Rng::seed_from_u64(CHECK_SEED);
    let adj = check_pass(&mut inputs, &mut weights, &mut rng);
    warn!(
        "fail rate trained model is {}",
        adj as f64 / (SAMPLE_SIZE as f64 * 2f64),
    );

    Ok(())
}
