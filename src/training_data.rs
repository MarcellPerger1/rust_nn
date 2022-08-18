#[derive(Debug, Clone, PartialEq)]
pub struct TrainingExample {
    pub inputs: Vec<f64>,
    pub expected: Vec<f64>
}

impl TrainingExample {
    pub fn new(inputs: &Vec<f64>, expected: &Vec<f64>) -> TrainingExample {
        TrainingExample {inputs: inputs.clone(), expected: expected.clone()}
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingData (pub Vec<TrainingExample>);

impl TrainingData {
    pub fn chunks(&self, size: usize) -> Vec<TrainingData>{
        self.iter_chunks(size).collect()
    }

    pub fn iter_chunks(&self, size: usize) -> impl Iterator<Item = TrainingData> + '_ {
        self.0.chunks(size).map(|c| TrainingData(c.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn get_tr_data(len: usize) -> TrainingData {
        TrainingData((0..len).map(|i| {
            let i = i as f64;
            let len = len as f64;
            // just some pseudo-random garbage
            TrainingExample::new(
                &vec![i, len], 
                &vec![i + 0.5])
        }).collect())
    }
    #[test]
    fn test_iter_chunks() {
        let data = get_tr_data(24);
        let chunks_iter = data.iter_chunks(10);
        let chunks: Vec<_> = chunks_iter.collect();
        assert_eq!(chunks, vec![TrainingData(data.0[0..10].to_vec()), TrainingData(data.0[10..20].to_vec()), TrainingData(data.0[20..].to_vec())]);
    }
}

