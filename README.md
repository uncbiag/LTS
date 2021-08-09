# Local Temperature Scaling (LTS)
This is the official repository for 

[Zhipeng Ding](https://biag.cs.unc.edu/author/zhipeng-ding/), [Xu Han](https://biag.cs.unc.edu/author/xu-han/), [Peirong Liu](https://biag.cs.unc.edu/author/peirong-liu/), and [Marc Niethammer](https://biag.cs.unc.edu/author/marc-niethammer/)  
**Local Temperature Scaling for Probability Calibration**  
*ICCV 2021*  
[paper](https://arxiv.org/abs/2008.05105)

## Key Feasures
Different from previous probability calibration methods, **LTS** is a spatially localized probability calibration approach for semantic segmentation.

### Spatially Localized Feature
In Figure

### Theoretical Justification
Using KKT conditions, we can prove that 
```
When the to-be-calibrated segmentation network is overconfident, 
minimizing NLL w.r.t. TS, IBTS, and LTS results in solutions that are also the solutions of 
maximizing entropy of the calibrated probability w.r.t. TS, IBTS and LTS under the condition of overconfidence.
```
Similarly, there is another theorem to validate the effectiveness of TS, IBTS and LTS under the condition of underconfidence in Appendix. 

### Implementation
In this paper, we use a simple tree-like convolutional network as in [(Lee et al.)](https://pages.ucsd.edu/~ztu/publication/pami_gpooling.pdf). However other neural network architectures could also work as illustrated by [(Bai et al.)](https://openreview.net/pdf?id=jsM6yvqiT0W)

## Instructions


## Paper
If you use LTS or some part of the code, please cite:
```
@article{ding2020local,
  title={Local temperature scaling for probability calibration},
  author={Ding, Zhipeng and Han, Xu and Liu, Peirong and Niethammer, Marc},
  journal={arXiv preprint arXiv:2008.05105},
  year={2020}
}
```
