#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub inputs: Vec<f64>,
    pub expected: Vec<f64>
}

impl TrainingExample {
    pub fn new(inputs: &Vec<f64>, expected: &Vec<f64>) -> TrainingExample {
        TrainingExample {inputs: inputs.clone(), expected: expected.clone()}
    }
}
