use std::fmt::Display;

use candle_core::{shape::Dim, DType, Error, IndexOp, Result, Tensor, D};
use candle_nn::{Init, VarBuilder};

/// Reduction type
#[derive(Debug)]
pub enum Reduction {
    None,
    Sum,
    Meam,
    TokenMean,
}

impl Default for Reduction {
    fn default() -> Self {
        Self::Sum
    }
}

// -----------------------------------------------------------------------------

///Conditional Random Field ported from [PyTorch-CRF](https://pytorch-crf.readthedocs.io/en/stable/)
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
        write!(
            f,
            "CRF(num_tags: {}, batch_first: {})",
            self.num_tags, self.batch_first
        )
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

    // pub(crate) fn set_transitions(&mut self, starts: Tensor, ends: Tensor, transitions: Tensor) {
    //     self.start_transitions = starts;
    //     self.end_transitions = ends;
    //     self.transitions = transitions;
    // }

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

        // println!("tags: {:?}", tags.to_vec2::<i64>()?);

        let mask = mask.to_dtype(emissions.dtype())?;

        // println!("mask: {:?}", mask.to_vec2::<f32>()?);

        // println!(
        //     "start_transitions: {:?}",
        //     self.start_transitions.to_vec1::<f32>()?
        // );

        let mut score = self.start_transitions.i(&tags.i(0)?)?;
        // println!("score: {:?}", score.to_vec1::<f32>()?);

        // println!("emissions: {:?}", emissions.to_vec3::<f32>()?);

        let z = multi_index(&emissions.i((0, 0..batch_size))?, &tags.i(0)?)?;
        // println!("z: {:?}", z.to_vec1::<f32>()?);

        score = score.broadcast_add(&z)?;
        // println!("score: {:?}", score.to_vec1::<f32>()?);

        for i in 1..seq_length {
            let z = multi_index(&self.transitions.i(&tags.i(i - 1)?)?, &tags.i(i)?)?;
            // println!("{i}, z: {:?}", z.to_vec1::<f32>()?);
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

        // println!("starts: {:?}", self.start_transitions.to_vec1::<f32>()?);
        // println!("transitions: {:?}", self.transitions.to_vec2::<f32>()?);
        // println!("ends: {:?}", self.end_transitions.to_vec1::<f32>()?);
        // println!("emissions: {:?}", emissions.to_vec3::<f32>()?);
        // println!("mask: {:?}", mask.to_vec2::<u8>()?);

        let mut score = self.start_transitions.broadcast_add(&emissions.i(0)?)?;
        // println!("score: {:?}", score.to_vec2::<f32>()?);

        for i in 1..seq_length {
            let broadcast_score = score.unsqueeze(2)?;
            // println!("broadcast_score: {:?}", broadcast_score.to_vec3::<f32>()?);
            let broadcast_emissions = emissions.i(i)?.unsqueeze(1)?;
            // println!(
            //     "broadcast_emissions: {:?}",
            //     broadcast_emissions.to_vec3::<f32>()?
            // );
            let next_score = broadcast_score
                .broadcast_add(&self.transitions)?
                .broadcast_add(&broadcast_emissions)?;

            // println!("next_score: {:?}", next_score.to_vec3::<f32>()?);

            let next_score = next_score.log_sum_exp(1)?;
            // println!("next_score: {:?}", next_score.to_vec2::<f32>()?);
            // println!(
            //     "mask[i].unsqueeze(1): {:?}",
            //     mask.i(i)?.unsqueeze(1)?.to_vec2::<u8>()
            // );
            let z = mask.i(i)?.unsqueeze(1)?.broadcast_as(next_score.shape())?;
            score = z.where_cond(&next_score, &score)?;
            // println!("score: {:?}", score.to_vec2::<f32>()?);
        }

        score = score.broadcast_add(&self.end_transitions)?;
        // println!("score: {:?}", score.to_vec2::<f32>()?);
        // println!("result: {:?}", score.log_sum_exp(1)?.to_vec1::<f32>()?);
        score.log_sum_exp(1)
    }

    fn viterbi_decode(&self, emissions: &Tensor, mask: &Tensor) -> Result<Vec<Vec<u32>>> {
        let (d1, d2, d3) = emissions.dims3()?;
        let (seq_length, batch_size) = mask.dims2()?;
        assert_eq!(d1, seq_length);
        assert_eq!(d2, batch_size);
        assert_eq!(d3, self.num_tags);
        assert!(all(&mask.i(0)?)?);

        let mut score = self.start_transitions.broadcast_add(&emissions.i(0)?)?;
        // println!("score: {:?}", score.to_vec2::<f32>()?);

        let mut history = Vec::with_capacity(seq_length);
        for i in 1..seq_length {
            let broadcast_sore = score.unsqueeze(2)?;
            // println!("broadcast_score: {:?}", broadcast_sore.to_vec3::<f32>()?);

            let broadcast_emission = emissions.i(i)?.unsqueeze(1)?;
            // println!(
            //     "broadcast_emission: {:?}",
            //     broadcast_emission.to_vec3::<f32>()?
            // );

            let next_score = broadcast_sore
                .broadcast_add(&self.transitions)?
                .broadcast_add(&broadcast_emission)?;

            // println!("next_score: {:?}", next_score.to_vec3::<f32>()?);

            let (next_score, indices) = max_indices(&next_score, 1)?;
            // println!("next_score: {:?}", next_score.to_vec2::<f32>()?);
            // println!("indices: {:?}", indices.to_vec2::<u32>()?);

            let z = mask.i(i)?.unsqueeze(1)?.broadcast_as(next_score.shape())?;
            score = z.where_cond(&next_score, &score)?;
            // println!("score: {:?}", score.to_vec2::<f32>()?);
            history.push(indices);
        }

        score = score.broadcast_add(&self.end_transitions)?;
        // println!("score: {:?}", score.to_vec2::<f32>()?);

        let seq_ends = mask
            .to_dtype(DType::I64)?
            .sum(0)?
            .broadcast_sub(&Tensor::ones(1, DType::I64, mask.device())?)?;
        // println!("seq_ends: {:?}", seq_ends.to_vec1::<i64>()?);

        let mut best_tags_list = vec![];

        for idx in 0..batch_size {
            let best_last_tag = score.i(idx)?.argmax(0)?;
            // println!(
            //     "{idx}:best_last_tag: {:?}",
            //     best_last_tag.to_scalar::<u32>()?
            // );

            let mut best_tags = vec![best_last_tag.to_scalar::<u32>()?];
            // println!("{idx}:best_tags: {:?}", best_tags);

            let z = seq_ends.i(idx)?.to_scalar::<i64>()? as usize;
            let mut a = history[..z].to_vec();
            a.reverse();
            for hist in a.iter() {
                // println!("hist: {:?}", hist.to_vec2::<u32>()?);
                let last_idx = *best_tags.last().unwrap() as usize;
                let best_last_tag = hist.i(idx)?.i(last_idx)?;
                // println!("best_last_tag: {:?}", best_last_tag.to_scalar::<u32>()?);
                best_tags.push(best_last_tag.to_scalar::<u32>()?);
            }

            best_tags.reverse();
            // println!("best_tags: {:?}", best_tags);
            best_tags_list.push(best_tags);
        }

        Ok(best_tags_list)
    }

    pub fn decode(&self, emissions: &Tensor, mask: Option<&Tensor>) -> Result<Vec<Vec<u32>>> {
        self.validate(emissions, None, mask)?;
        let mask = if let Some(mask) = mask {
            mask.clone()
        } else {
            let (d1, d2, _) = emissions.dims3()?;
            Tensor::ones((d1, d2), DType::U8, emissions.device())?
        };

        let (emissions, mask) = if self.batch_first {
            (emissions.transpose(0, 1)?, mask.transpose(0, 1)?)
        } else {
            (emissions.clone(), mask.clone())
        };
        self.viterbi_decode(&emissions, &mask)
    }

    pub fn forward(
        &self,
        emissions: &Tensor,
        tags: &Tensor,
        mask: Option<&Tensor>,
        reduction: Reduction,
    ) -> Result<Tensor> {
        self.validate(emissions, Some(tags), mask)?;
        let mask = if let Some(mask) = mask {
            mask.clone()
        } else {
            Tensor::ones_like(tags)?.to_dtype(DType::U8)?
        };

        let (emissions, tags, mask) = if self.batch_first {
            (
                emissions.transpose(0, 1)?,
                tags.transpose(0, 1)?,
                mask.transpose(0, 1)?,
            )
        } else {
            (emissions.clone(), tags.clone(), mask.clone())
        };

        let numerator = self.compute_score(&emissions, &tags, &mask)?;
        let denominator = self.compute_normalizer(&emissions, &mask)?;

        let llh = numerator.broadcast_sub(&denominator)?;

        match reduction {
            Reduction::Sum => llh.sum_all(),
            Reduction::Meam => llh.mean_all(),
            Reduction::TokenMean => {
                let mask = mask.to_dtype(llh.dtype())?;
                let z = mask.sum_all()?;
                llh.sum_all()?.broadcast_div(&z)
            }
            Reduction::None => Ok(llh),
        }
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

fn max_indices<D: Dim + Copy>(x: &Tensor, dim: D) -> Result<(Tensor, Tensor)> {
    let max = x.max(dim)?;
    let idx = x.argmax(dim)?;
    Ok((max, idx))
}

// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {

    use super::*;
    use candle_core::{Device, Result};
    use candle_nn::VarMap;

    fn init_data(num_tags: usize) -> Result<(CRF, Tensor, Tensor, Tensor)> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let m = {
            let mut m = CRF::new(num_tags, false, vb.clone())?;

            m.start_transitions =
                Tensor::new(&[-0.0958_f32, -0.0838, -0.0998, -0.0567, 0.0658], &device)?;

            m.transitions = Tensor::new(
                &[
                    [0.0186_f32, 0.0086, 0.0116, 0.0276, -0.0708],
                    [0.0453, 0.0228, 0.0292, 0.0237, -0.0622],
                    [0.0478, -0.0518, -0.0099, -0.0932, -0.0855],
                    [-0.0479, 0.0702, -0.0069, 0.0862, 0.0412],
                    [0.0668, 0.0480, 0.0736, -0.0466, 0.0364],
                ],
                &device,
            )?;

            m.end_transitions =
                Tensor::new(&[0.0051_f32, -0.0636, -0.0887, -0.0541, -0.0946], &device)?;
            m
        };

        let emissions = Tensor::new(
            &[
                [
                    [-1.0815_f32, -1.0964, 0.4902, -0.1495, 1.0123],
                    [0.4571, 0.3830, 0.0305, -1.8651, -1.3217],
                ],
                [
                    [-0.4570, -0.3757, -1.2726, -0.3914, 1.0987],
                    [-2.0027, 0.8384, -1.2015, -1.9222, 1.1583],
                ],
                [
                    [1.1398, -1.4247, 1.4649, 0.5058, 0.9214],
                    [0.1536, -1.2564, -0.6121, 0.2406, -1.0176],
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
        let (m, emissions, tags, mask) = init_data(5)?;

        let score = m.compute_score(&emissions, &tags, &mask)?;
        println!("score: {:?}", score);
        Ok(())
    }

    #[test]
    fn test_compute_normalizer() -> Result<()> {
        let (m, emissions, _tags, mask) = init_data(5)?;
        let normalizer = m.compute_normalizer(&emissions, &mask)?;
        println!("normalizer: {:?}", normalizer.to_vec1::<f32>()?);
        Ok(())
    }

    #[test]
    fn test_forward() -> Result<()> {
        let (m, emissions, tags, _mask) = init_data(5)?;
        let llh = m.forward(&emissions, &tags, None, Reduction::default())?;
        println!("llh: {:?}", llh.to_scalar::<f32>()?);
        Ok(())
    }

    #[test]
    fn test_decode() -> Result<()> {
        let (m, emissions, _tags, _mask) = init_data(5)?;
        let tags = m.decode(&emissions, None)?;
        println!("tags: {:?}", tags);
        Ok(())
    }

    #[test]
    fn test_as() {
        let a = Tensor::new(&[[1_u32], [1]], &Device::Cpu).unwrap();
        let z = a.broadcast_as((2, 5)).unwrap();
        println!("{:?}", z.to_vec2::<u32>().unwrap());
    }

    #[test]
    fn test_sum() {
        let a = Tensor::new(&[-6.8511, -6.5189], &Device::Cpu).unwrap();
        let z = a.sum_all().unwrap();
        println!("{:?}", z.to_scalar::<f64>().unwrap());
    }

    #[test]
    fn test_max_indices() {
        let a = Tensor::new(
            &[
                [
                    [-1.6157_f32, -1.5444, -2.4382, -1.5410, -0.1493],
                    [-1.5919, -1.5331, -2.4236, -1.5479, -0.1437],
                    [-0.0189, -0.0371, -0.8921, -0.0942, 1.4036],
                    [-0.7111, -0.5118, -1.4856, -0.5114, 0.9337],
                    [0.6879, 0.7503, -0.1210, 0.6400, 2.2131],
                ],
                [
                    [-1.6227, 1.2083, -0.8286, -1.5332, 1.4490],
                    [-1.6582, 1.1603, -0.8732, -1.5994, 1.3953],
                    [-2.0242, 0.7173, -1.2808, -2.0847, 1.0035],
                    [-3.9724, -1.0133, -3.1302, -3.7578, -0.7223],
                    [-3.1918, -0.3696, -2.3839, -3.2247, -0.0612],
                ],
            ],
            &Device::Cpu,
        )
        .unwrap();

        let max = a.max(1).unwrap();
        let max_idx = a.argmax(1).unwrap();
        println!("{:?}", max.to_vec2::<f32>().unwrap());
        println!("{:?}", max_idx.to_vec2::<u32>());
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
