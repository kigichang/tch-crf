use std::fmt::Display;

use candle_core::{DType, Error, IndexOp, Result, Tensor, D};
use candle_nn::{Init, VarBuilder};

#[derive(Debug)]
pub struct CRF {
    num_tags: usize,
    batch_first: bool,

    start_transitions: Tensor,
    end_transitions: Tensor,
    transitions: Tensor,
}

impl Display for CRF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CRF(num_tags: {})", self.num_tags)
    }
}

impl CRF {
    pub fn new(num_tags: usize, batch_first: bool, vb: VarBuilder) -> Result<Self> {
        if num_tags == 0 {
            return Err(Error::Msg(format!("invalid number of tags: {}", num_tags)));
        }

        let start_transitions = vb.get_with_hints(
            num_tags,
            "start_transitions",
            Init::Uniform { lo: -0.1, up: 1.0 },
        )?;
        let end_transitions = vb.get_with_hints(
            num_tags,
            "end_transitions",
            Init::Uniform { lo: -0.1, up: 1.0 },
        )?;

        let transitions = vb.get_with_hints(
            (num_tags, num_tags),
            "transitions",
            Init::Uniform { lo: -0.1, up: 1.0 },
        )?;

        Ok(Self {
            num_tags,
            batch_first,
            start_transitions,
            end_transitions,
            transitions,
        })
    }

    pub(crate) fn set_transitions(&mut self, starts: Tensor, ends: Tensor, transitions: Tensor) {
        self.start_transitions = starts;
        self.end_transitions = ends;
        self.transitions = transitions;
    }

    fn validate(
        &self,
        emissions: &Tensor,
        tags: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<()> {
        let (d1, d2, d3) = emissions.dims3()?; // check if the tensor has 3 dimensions

        if d3 != self.num_tags {
            return Err(Error::Msg(format!(
                "expected last dimension of emissions is {}, got {}",
                self.num_tags, d3
            ))); // check if the last dimension of the tensor is equal to the number of tags
        }

        if let Some(tags) = tags {
            if tags.dtype() != DType::I64 {
                return Err(Error::Msg("tags must be of type i64".to_string()));
            }

            let (tag_d1, tag_d2) = tags.dims2()?; // check if the tensor has 2 dimensions
            if (d1, d2) != (tag_d1, tag_d2) {
                return Err(Error::Msg(format!(
                    "the first two dimensions of emissions and tags must match, got ({}, {}) and ({}, {})",
                    d1, d2, d1, d2
                )));
            }
        }

        if let Some(mask) = mask {
            if mask.dtype() != DType::U8 {
                return Err(Error::Msg("mask must be of type u8".to_string()));
            }
            let (mask_d1, mask_d2) = mask.dims2()?; // check if the tensor has 2 dimensions
            if (d1, d2) != (mask_d1, mask_d2) {
                return Err(Error::Msg(format!(
                    "the first two dimensions of emissions and mask must match, got ({}, {}) and ({}, {})",
                    d1, d2, mask_d1, mask_d2
                )));
            }

            let no_empty_seq = !self.batch_first && all(&mask.i(0)?)?;
            let no_empty_seq_bf = self.batch_first && all(&mask.i((.., 0))?)?;

            if !no_empty_seq && !no_empty_seq_bf {
                return Err(Error::Msg(
                    "mask of the first timestep must all be on".to_string(),
                ));
            }
        }

        Ok(())
    }

    fn compute_score(&self, emissions: &Tensor, tags: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (d1, d2, d3) = emissions.dims3()?;
        let (seq_length, batch_size) = tags.dims2()?;
        assert_eq!(d1, seq_length);
        assert_eq!(d2, batch_size);
        assert_eq!(d3, self.num_tags);
        assert_eq!(mask.shape(), tags.shape());
        assert!(all(&mask.i(0)?)?);

        println!("tags: {:?}", tags.to_vec2::<i64>()?);

        let mask = mask.to_dtype(emissions.dtype())?;

        println!("mask: {:?}", mask.to_vec2::<f32>()?);

        println!(
            "start_transitions: {:?}",
            self.start_transitions.to_vec1::<f32>()?
        );

        let mut score = self.start_transitions.i(&tags.i(0)?)?;
        println!("score: {:?}", score.to_vec1::<f32>()?);

        println!("emissions: {:?}", emissions.to_vec3::<f32>()?);

        let z = multi_index(&emissions.i((0, 0..batch_size))?, &tags.i(0)?)?;
        println!("z: {:?}", z.to_vec1::<f32>()?);

        score = score.broadcast_add(&z)?;
        println!("score: {:?}", score.to_vec1::<f32>()?);

        for i in 1..seq_length {
            let z = multi_index(&self.transitions.i(&tags.i(i - 1)?)?, &tags.i(i)?)?;
            println!("{i}, z: {:?}", z.to_vec1::<f32>()?);
            score = score.broadcast_add(&z.broadcast_mul(&mask.i(i)?)?)?;

            let z = multi_index(&emissions.i((i, 0..batch_size))?, &tags.i(i)?)?;
            score = score.broadcast_add(&z.broadcast_mul(&mask.i(i)?)?)?;
        }

        let seq_ends = mask
            .to_dtype(DType::I64)?
            .sum(0)?
            .broadcast_sub(&Tensor::ones(1, DType::I64, mask.device())?)?;

        let last_tags = multi_index(
            &tags.i(&seq_ends)?,
            &Tensor::arange(0, batch_size as i64, mask.device())?,
        )?;

        score.broadcast_add(&self.end_transitions.i(&last_tags)?)
    }

    fn compute_normalizer(&self, emissions: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (d1, d2, d3) = emissions.dims3()?;
        let (seq_length, batch_size) = mask.dims2()?;
        assert_eq!(d1, seq_length);
        assert_eq!(d2, batch_size);
        assert_eq!(d3, self.num_tags);
        assert!(all(&mask.i(0)?)?);

        println!("starts: {:?}", self.start_transitions.to_vec1::<f32>()?);
        println!("transitions: {:?}", self.transitions.to_vec2::<f32>()?);
        println!("ends: {:?}", self.end_transitions.to_vec1::<f32>()?);
        println!("emissions: {:?}", emissions.to_vec3::<f32>()?);
        println!("mask: {:?}", mask.to_vec2::<u8>()?);

        let mut score = self.start_transitions.broadcast_add(&emissions.i(0)?)?;
        println!("score: {:?}", score.to_vec2::<f32>()?);

        for i in 1..seq_length {
            let broadcast_score = score.unsqueeze(2)?;
            println!("broadcast_score: {:?}", broadcast_score.to_vec3::<f32>()?);
            let broadcast_emissions = emissions.i(i)?.unsqueeze(1)?;
            println!(
                "broadcast_emissions: {:?}",
                broadcast_emissions.to_vec3::<f32>()?
            );
            let next_score = broadcast_score
                .broadcast_add(&self.transitions)?
                .broadcast_add(&broadcast_emissions)?;

            println!("next_score: {:?}", next_score.to_vec3::<f32>()?);

            let next_score = next_score.log_sum_exp(1)?;
            println!("next_score: {:?}", next_score.to_vec2::<f32>()?);
            println!(
                "mask[i].unsqueeze(1): {:?}",
                mask.i(i)?.unsqueeze(1)?.to_vec2::<u8>()
            );
            let z = mask.i(i)?.unsqueeze(1)?.broadcast_as(next_score.shape())?;
            score = z.where_cond(&next_score, &score)?;
            println!("score: {:?}", score.to_vec2::<f32>()?);
        }

        score = score.broadcast_add(&self.end_transitions)?;
        println!("score: {:?}", score.to_vec2::<f32>()?);
        println!("result: {:?}", score.log_sum_exp(1)?.to_vec1::<f32>()?);
        score.log_sum_exp(1)
    }

    fn viterbi_decode() -> Vec<Vec<isize>> {
        unimplemented!("viterbi_decode")
    }

    fn decode(&self, emissions: &Tensor) -> Result<()> {
        unimplemented!("decode")
    }
}

// -----------------------------------------------------------------------------

fn all(x: &Tensor) -> Result<bool> {
    let zero = Tensor::zeros(1, x.dtype(), x.device())?;
    Ok(x.broadcast_ne(&zero)?
        .flatten_all()?
        .min(0)?
        .to_scalar::<u8>()?
        != 0)
}

// -----------------------------------------------------------------------------

fn multi_index(src: &Tensor, idx: &Tensor) -> Result<Tensor> {
    let index = idx.reshape((idx.dim(0)?, 1))?;
    src.gather(&index, D::Minus1)?.squeeze(D::Minus1)
}

// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {

    use super::*;
    use candle_core::{Device, Result};
    use candle_nn::VarMap;

    fn init_data() -> Result<(CRF, Tensor, Tensor, Tensor)> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let num_tags = 5;
        let m = {
            let mut m = CRF::new(num_tags, false, vb.clone())?;

            let starts = Tensor::new(&[-0.0579_f32, -0.0496, -0.0710, -0.0734, 0.0341], &device)?;

            let transitions = Tensor::new(
                &[
                    [0.0657_f32, 0.0745, 0.0158, 0.0118, 0.0006],
                    [0.0388, 0.0191, -0.0075, -0.0740, 0.0204],
                    [0.0116, -0.0147, -0.0462, 0.0784, -0.0037],
                    [-0.0215, 0.0935, 0.0649, 0.0946, 0.0573],
                    [-0.0048, -0.0320, 0.0051, 0.0460, 0.0511],
                ],
                &device,
            )?;

            let ends = Tensor::new(&[-0.0785_f32, 0.0676, 0.0347, 0.0093, 0.0864], &device)?;
            m.set_transitions(starts, ends, transitions);
            m
        };

        let emissions = Tensor::new(
            &[
                [
                    [-0.6094_f32, 0.4727, -0.2072, 0.8678, -0.3666],
                    [-0.5847, 0.8231, 2.2543, -1.8335, -1.5335],
                ],
                [
                    [0.6712, -0.6021, -0.0490, 0.8836, -0.8548],
                    [1.1375, -0.0518, -2.1887, -0.9732, -0.0702],
                ],
                [
                    [-0.2326, 2.0694, -1.0252, 1.3830, 0.5637],
                    [2.7534, -0.3950, 1.4539, -0.7752, -0.7997],
                ],
            ],
            &device,
        )?;
        let tags = Tensor::new(&[[0_i64, 1], [2, 4], [3, 1]], &device)?;
        let mask = Tensor::new(&[[1_u8, 1], [1, 1], [1, 1]], &device)?;

        Ok((m, emissions, tags, mask))
    }

    #[test]
    fn test_compute_score() -> Result<()> {
        let (m, emissions, tags, mask) = init_data()?;

        let score = m.compute_score(&emissions, &tags, &mask)?;
        println!("score: {:?}", score);
        Ok(())
    }

    #[test]
    fn test_compute_normalizer() -> Result<()> {
        let (m, emissions, _tags, mask) = init_data()?;
        let normalizer = m.compute_normalizer(&emissions, &mask)?;
        println!("normalizer: {:?}", normalizer.to_vec1::<f32>()?);
        Ok(())
    }

    #[test]
    fn test_as() {
        let a = Tensor::new(&[[1_u32], [1]], &Device::Cpu).unwrap();
        let z = a.broadcast_as((2, 5)).unwrap();
        println!("{:?}", z.to_vec2::<u32>().unwrap());
    }
}

// #[cfg(test)]
// mod tests {
//     use std::fmt::Result;

//     use super::*;
//     use candle_core::op::CmpOp;
//     use candle_core::{backend, Device, Error, IndexOp, Tensor, D};

//     #[test]
//     fn test1() {
//         let a = [[0.9808_f32, 0.6285], [0.5903, 0.9723], [0.6518, 0.0818]];
//         let a = Tensor::new(&a, &Device::Cpu).unwrap();
//         let b = a.i((.., 0)).unwrap();
//         println!("{:?}", b);
//     }

//     #[test]
//     fn test2() {
//         let a = Tensor::new(&[[1_i64, 2, 3], [4, 5, 6]], &Device::Cpu).unwrap();
//         // let m = a.flatten_all().unwrap().min(0).unwrap();
//         // let z = m.zeros_like().unwrap();
//         // let m = m.ne(&z).unwrap();

//         let all = all(&a).unwrap();
//         println!("{:?}", all);
//     }

//     #[test]
//     fn test_compute_score() {
//         let data = [
//             [
//                 [0.8632_f32, 0.8215, 2.3053, 0.4782, -0.3433],
//                 [1.1366, 0.6457, -0.1799, -0.8175, -0.3634],
//             ],
//             [
//                 [-1.3191, -1.0503, 1.3044, -0.1696, 0.3270],
//                 [0.7163, -0.6747, -0.3489, -0.0389, 0.7747],
//             ],
//             [
//                 [-0.7034, -1.1089, -1.5936, 0.4257, -0.6306],
//                 [-1.5459, -1.2779, -1.7721, 1.6539, -1.5292],
//             ],
//         ];

//         let dev = &Device::Cpu;
//         let seq_length: usize = 3;
//         let batch_size: usize = 2;
//         let tags = Tensor::new(&[[0_i64, 1], [2, 4], [3, 1]], dev).unwrap();
//         let starts = Tensor::new(&[-0.0644_f32, -0.0809, -0.0657, 0.0744, 0.0357], &dev).unwrap();
//         let score = starts.i(&tags.i(0).unwrap()).unwrap();
//         println!("{:?}", score.to_vec1::<f32>().unwrap());

//         // let idx0 = Tensor::zeros((batch_size, 1), DType::I64, &dev).unwrap();
//         // let idx1 = Tensor::arange(0 as i64, batch_size as i64, &dev)
//         //     .unwrap()
//         //     .unsqueeze(0)
//         //     .unwrap()
//         //     .t()
//         //     .unwrap();
//         // let idx2 = tags.i(0).unwrap().unsqueeze(0).unwrap().t().unwrap();
//         // println!("{:?}", idx0.to_vec2::<i64>().unwrap());
//         // println!("{:?}", idx1.to_vec2::<i64>().unwrap());
//         // println!("{:?}", idx2.to_vec2::<i64>().unwrap());
//         // let idx = Tensor::cat(&[&idx0, &idx1, &idx2], 1).unwrap();
//         // println!("{:?}", idx.to_vec2::<i64>().unwrap());

//         let idx0 = Tensor::zeros(batch_size, DType::I64, &dev).unwrap();
//         let idx1 = Tensor::arange(0 as i64, batch_size as i64, &dev).unwrap();
//         let idx2 = tags.i(0).unwrap();
//         let idx = Tensor::stack(&[&idx0, &idx1, &idx2], 1).unwrap();
//         println!("{:?}", idx.to_vec2::<i64>().unwrap());

//         let emissions = Tensor::new(&data, &Device::Cpu).unwrap();
//         let z = emissions.i((0, 0..batch_size)).unwrap();
//         println!("{:?}", z.to_vec2::<f32>().unwrap());
//         let z = z.index_select(&tags.i(0).unwrap(), 1).unwrap();
//         println!("{:?}", z.to_vec2::<f32>().unwrap());
//         let z = z
//             .gather(&tags.i(0).unwrap().reshape((batch_size, 1)).unwrap(), 1)
//             .unwrap();
//         println!("{:?}", z.to_vec2::<f32>().unwrap());

//         let z = emissions
//             .i((0, 0..batch_size))
//             .unwrap()
//             .gather(&tags.i(0).unwrap().reshape((batch_size, 1)).unwrap(), 1)
//             .unwrap()
//             .squeeze(1)
//             .unwrap();
//         println!("{:?}", z.to_vec1::<f32>().unwrap());

//         let z = multi_index(
//             &emissions.i((0, 0..batch_size)).unwrap(),
//             &tags.i(0).unwrap(),
//         )
//         .unwrap();

//         println!("{:?}", z.to_vec1::<f32>().unwrap());

//         //let x = Tensor::arange(0, batch_size, &dev).unwrap();
//         //let z = emissions.i((0, &x, &tags.i(0).unwrap())).unwrap();
//         //println!("{:?}", z);
//         // let tag = Tensor::new(&[0_i64, 1], &Device::Cpu).unwrap();
//         // let batch = Tensor::new(&[0_i64, 1], &Device::Cpu).unwrap();

//         // let z = emissions.i((0, &batch, &tag)).unwrap();
//         // println!("{:?}", z.to_vec2::<f32>().unwrap());

//         // println!("idx: {:?}", idx.to_vec2::<i64>().unwrap());
//         // let idx = Tensor::cat(&[&idx.t().unwrap(), &tag.t().unwrap()], 0).unwrap();
//         // println!("idx: {:?}", idx.to_vec2::<i64>().unwrap());
//         // let idx= Tensor::cat(, dim)
//     }

//     #[test]
//     fn test_cat() {
//         let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap();
//         let b = Tensor::ones((2, 3), DType::F32, &Device::Cpu).unwrap();

//         let c = Tensor::cat(&[&a, &b], 0).unwrap();
//         assert_eq!(c.shape().dims(), &[4, 3]);
//         println!("{:?}", c.to_vec2::<f32>().unwrap());

//         let c = Tensor::cat(&[&a, &b], 1).unwrap();
//         assert_eq!(c.shape().dims(), &[2, 6]);
//         println!("{:?}", c.to_vec2::<f32>().unwrap());
//     }

//     #[test]
//     fn test_stack() {
//         //let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap();
//         //let b = Tensor::ones((2, 3), DType::F32, &Device::Cpu).unwrap();
//         let a = Tensor::arange(1.0_f32, 7., &Device::Cpu)
//             .unwrap()
//             .reshape((2, 3))
//             .unwrap();
//         let b = Tensor::arange(7.0_f32, 13., &Device::Cpu)
//             .unwrap()
//             .reshape((2, 3))
//             .unwrap();

//         let z = Tensor::stack(&[&a, &b], 0).unwrap();
//         println!("0:{:?}, {:?}", z.shape(), z.to_vec3::<f32>().unwrap());

//         let z = Tensor::stack(&[&a, &b], 1).unwrap();
//         println!("1:{:?}, {:?}", z.shape(), z.to_vec3::<f32>().unwrap());

//         let z = Tensor::stack(&[&a, &b], 2).unwrap();
//         println!("2:{:?}, {:?}", z.shape(), z.to_vec3::<f32>().unwrap());

//         // let a = Tensor::zeros(2, DType::I64, &Device::Cpu).unwrap();
//         // let b = Tensor::new(&[0_i64, 1], &Device::Cpu).unwrap();
//         // let c = Tensor::new(&[0_i64, 1], &Device::Cpu).unwrap();

//         // let z = Tensor::stack(&[&a, &b], 0).unwrap();
//         // let z = Tensor::stack(&[&z, &c], 1).unwrap();
//         // println!("{:?}", z.to_vec2::<i64>().unwrap());
//     }

//     #[test]
//     fn test_stack2() {
//         let a = Tensor::new(&[1_i64, 2], &Device::Cpu).unwrap();
//         let b = Tensor::new(&[3_i64, 4], &Device::Cpu).unwrap();
//         let c = Tensor::new(&[5_i64, 6], &Device::Cpu).unwrap();

//         let z = Tensor::cat(&[&a, &b], 1).unwrap();
//         println!("{:?}", z.to_vec1::<i64>().unwrap());

//         // let c = c.unsqueeze(0).unwrap();
//         // println!("{:?}", z);
//         // println!("{:?}", c);
//         // let z = Tensor::cat(&[&z, &c], 0).unwrap();
//         // println!("{:?}", z.to_vec2::<i64>().unwrap());
//     }

//     #[test]
//     fn test_1() {
//         let tags = Tensor::new(&[[0_i64, 1], [2, 4], [3, 1]], &Device::Cpu).unwrap();

//         let start =
//             Tensor::new(&[0.0665_f32, 0.0114, 0.0563, 0.0625, -0.0981], &Device::Cpu).unwrap();
//         let z = start.i(&tags.i(0).unwrap()).unwrap();
//         println!("{:?}", z.to_vec1::<f32>().unwrap());
//     }

//     #[test]
//     fn test_2() {
//         let starts = Tensor::new(
//             &[-0.0644_f32, -0.0809, -0.0657, 0.0744, 0.0357],
//             &Device::Cpu,
//         )
//         .unwrap();

//         let tags = Tensor::new(&[[0_i64, 1], [2, 4], [3, 1]], &Device::Cpu).unwrap();

//         let score = starts.i(&tags.i(0).unwrap()).unwrap();
//         println!("{:?}", score.to_vec1::<f32>().unwrap());

//         // let z = starts
//         //     .i((&tags.i(0).unwrap(), &tags.i(1).unwrap()))
//         //     .unwrap();
//         // println!("{:?}", z.to_vec1::<f32>().unwrap());

//         println!("{}", tags.i(0).unwrap().dim(0).unwrap());
//     }
// }
