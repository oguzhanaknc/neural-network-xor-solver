use rand::Rng;
use std::env;

#[derive(Clone)]
struct NeuralNetwork {
    w1: f32,
    w2: f32,
    b: f32,
}
#[derive(Clone)]
struct Xor {
    or: NeuralNetwork,
    and: NeuralNetwork,
    nand: NeuralNetwork,
}

// xor-gate training data
const XOR_GATE: [[i32; 3]; 4] = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];
const TRAIN_COUNT: usize = XOR_GATE.len();
const EPS: f32 = 1e-1;
const LEARNING_RATE: f32 = 1e-1;

fn main() {
    let mut verbose: bool = false;
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        if args[1] == "--verbose" {
            println!("verbose mode activated");
            verbose = true;
        }
    }
    // create xor
    let mut xor: Xor = rand_xor();
    // for loop for training
    for _i in 0..1000000 {
        let g: Xor = finite_difference_method(&mut xor);
        xor = train(xor, g);
        if verbose {
            // print xor weights
            println!(
                "or: w1 = {}, w2 = {}, b = {}",
                xor.or.w1, xor.or.w2, xor.or.b
            );
            println!(
                "and: w1 = {}, w2 = {}, b = {}",
                xor.and.w1, xor.and.w2, xor.and.b
            );
            println!(
                "nand: w1 = {}, w2 = {}, b = {}",
                xor.nand.w1, xor.nand.w2, xor.nand.b
            );
            println!("cost = {}", calculate_cost(xor.clone()));
        }
    }
    let inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    for input in inputs.iter() {
        let x1 = input[0];
        let x2 = input[1];
        let y = forward(xor.clone(), x1, x2);
        println!("XOR({}, {}) = {}", x1, x2, y.round());
    }
}

fn calculate_cost(model: Xor) -> f32 {
    let mut result: f32 = 0.0;

    for i in 0..TRAIN_COUNT {
        let x1: f32 = XOR_GATE[i][0] as f32;
        let x2: f32 = XOR_GATE[i][1] as f32;
        let y: f32 = forward(model.clone(), x1, x2);
        let d: f32 = y - XOR_GATE[i][2] as f32;
        result += d * d;
    }
    result /= TRAIN_COUNT as f32;
    result
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn forward(xor: Xor, x1: f32, x2: f32) -> f32 {
    // first layer first input of neural network (or)
    let a: f32 = sigmoid(xor.or.w1 * x1 + xor.or.w2 * x2 + xor.or.b);
    // first layer second input of neural network (nand)
    let b: f32 = sigmoid(xor.nand.w1 * x1 + xor.nand.w2 * x2 + xor.nand.b);
    // second layer input of neural network (and)
    sigmoid(a * xor.and.w1 + b * xor.and.w2 + xor.and.b)
}

fn rand_xor() -> Xor {
    // create xor
    let test = Xor {
        or: NeuralNetwork {
            w1: rand::thread_rng().sample::<f32, _>(rand::distributions::Standard),
            w2: rand::thread_rng().sample::<f32, _>(rand::distributions::Standard),
            b: rand::thread_rng().sample::<f32, _>(rand::distributions::Standard),
        },
        and: NeuralNetwork {
            w1: rand::thread_rng().sample::<f32, _>(rand::distributions::Standard),
            w2: rand::thread_rng().sample::<f32, _>(rand::distributions::Standard),
            b: rand::thread_rng().sample::<f32, _>(rand::distributions::Standard),
        },
        nand: NeuralNetwork {
            w1: rand::thread_rng().sample::<f32, _>(rand::distributions::Standard),
            w2: rand::thread_rng().sample::<f32, _>(rand::distributions::Standard),
            b: rand::thread_rng().sample::<f32, _>(rand::distributions::Standard),
        },
    };
    test
}

fn finite_difference_method(m: &mut Xor) -> Xor {
    let mut g: Xor = Xor {
        or: NeuralNetwork {
            w1: 0.0,
            w2: 0.0,
            b: 0.0,
        },
        and: NeuralNetwork {
            w1: 0.0,
            w2: 0.0,
            b: 0.0,
        },
        nand: NeuralNetwork {
            w1: 0.0,
            w2: 0.0,
            b: 0.0,
        },
    };
    let cost: f32 = calculate_cost(m.clone());
    let mut saved: f32;
    saved = m.or.w1;
    m.or.w1 += EPS;
    g.or.w1 = (calculate_cost(m.clone()) - cost) / EPS;
    m.or.w1 = saved;

    saved = m.or.w2;
    m.or.w2 += EPS;
    g.or.w2 = (calculate_cost(m.clone()) - cost) / EPS;
    m.or.w2 = saved;

    saved = m.or.b;
    m.or.b += EPS;
    g.or.b = (calculate_cost(m.clone()) - cost) / EPS;
    m.or.b = saved;

    saved = m.and.w1;
    m.and.w1 += EPS;
    g.and.w1 = (calculate_cost(m.clone()) - cost) / EPS;
    m.and.w1 = saved;

    saved = m.and.w2;
    m.and.w2 += EPS;
    g.and.w2 = (calculate_cost(m.clone()) - cost) / EPS;
    m.and.w2 = saved;

    saved = m.and.b;
    m.and.b += EPS;
    g.and.b = (calculate_cost(m.clone()) - cost) / EPS;
    m.and.b = saved;

    saved = m.nand.w1;
    m.nand.w1 += EPS;
    g.nand.w1 = (calculate_cost(m.clone()) - cost) / EPS;
    m.nand.w1 = saved;

    saved = m.nand.w2;
    m.nand.w2 += EPS;
    g.nand.w2 = (calculate_cost(m.clone()) - cost) / EPS;
    m.nand.w2 = saved;

    saved = m.nand.b;
    m.nand.b += EPS;
    g.nand.b = (calculate_cost(m.clone()) - cost) / EPS;
    m.nand.b = saved;

    g
}

// apply gradient descent
fn train(mut xor: Xor, g: Xor) -> Xor {
    xor.or.w1 -= LEARNING_RATE * g.or.w1;
    xor.or.w2 -= LEARNING_RATE * g.or.w2;
    xor.or.b -= LEARNING_RATE * g.or.b;

    xor.and.w1 -= LEARNING_RATE * g.and.w1;
    xor.and.w2 -= LEARNING_RATE * g.and.w2;
    xor.and.b -= LEARNING_RATE * g.and.b;

    xor.nand.w1 -= LEARNING_RATE * g.nand.w1;
    xor.nand.w2 -= LEARNING_RATE * g.nand.w2;
    xor.nand.b -= LEARNING_RATE * g.nand.b;

    xor
}
