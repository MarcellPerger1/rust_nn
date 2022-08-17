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

#[derive(Debug, Clone)]
pub struct TrainingData (Vec<TrainingExample>);

impl TrainingData {
    pub fn chunks(&self, size: usize) -> Vec<TrainingData>{
        self.0.chunks(size).map(|c| TrainingData(c.to_vec())).collect()
    }
}

