## Distillers

This is a PyTorch implementation of the [DCD paper](https://arxiv.org/abs/2407.11802):

```
@misc{giakoumoglou2024discriminative,
      title={DCD: Discriminative and Consistent Representation Distillation}, 
      author={Nikolaos Giakoumoglou and Tania Stathaki},
      year={2024},
      eprint={2407.11802},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.11802}, 
}
```

It also includes the implementation of and [RRD paper](https://arxiv.org/abs/2407.12073):
```
@misc{giakoumoglou2024relational,
      title={Relational Representation Distillation}, 
      author={Nikolaos Giakoumoglou and Tania Stathaki},
      year={2024},
      eprint={2407.12073},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.12073}, 
}
```

This repo is based on [RepDistiller implementation](https://github.com/HobbitLong/RepDistiller): [Paper](http://arxiv.org/abs/1910.10699)


### Benchmarks 13 state-of-the-art knowledge distillation methods in PyTorch

1. (KD) - Distilling the Knowledge in a Neural Network
2. (FitNet) - Fitnets: Hints for Thin Deep Nets
3. (AT) - Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer
4. (SP) - Similarity-Preserving Knowledge Distillation
5. (CC) - Correlation Congruence for Knowledge Distillation
6. (VID) - Variational Information Distillation for Knowledge Transfer
7. (RKD) - Relational Knowledge Distillation
8. (PKT) - Probabilistic Knowledge Transfer for Deep Representation Learning
9. (AB) - Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons
10. (FT) - Paraphrasing Complex Network: Network Compression via Factor Transfer
11. (FSP) - A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning
12. (NST) - Like What You Like: Knowledge Distill via Neuron Selectivity Transfer
13. (CRD) - Contrastive Representation Distillation
14. (RRD) - Relational Representation Distillation
15. (DCD) - Discriminative and Consistent Distillation

### Installation

1. Open your terminal and run the following command to clone the repository:
   ```
   git clone https://github.com/giakoumoglou/distillers.git
   ```

2. Change into the directory of the cloned repository and nstall the necessary dependencies using `pip`:

   ```
   cd distillers
   pip install -r requirements.txt
   ```

3. This repo was tested with Ubuntu 16.04.5 LTS, Python 3.5, PyTorch 0.4.0, and CUDA 9.0. But it should be runnable with recent PyTorch versions >=0.4.0

4. Fetch the pretrained teacher models by:

    ```
    sh scripts/fetch_pretrained_teachers.sh
    ```
    
   which will download and save the models to `save/models`

### Train Teacher Models

(Optional) Train teacher networks from scratch. Example commands are in `scripts/run_cifar_vanilla.sh`

### Train Student Models
  
1. Run distillation by following commands in `scripts/run_cifar_distill.sh`. An example of running Geoffrey's original Knowledge Distillation (KD) is given by:

    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --trial 1
    ```
    where the flags are explained as:
    - `--path_t`: specify the path of the teacher model
    - `--model_s`: specify the student model, see 'models/\_\_init\_\_.py' to check the available model types.
    - `--distill`: specify the distillation method
    - `-r`: the weight of the cross-entropy loss between logit and ground truth, default: `1`
    - `-a`: the weight of the KD loss, default: `None`
    - `-b`: the weight of other distillation losses, default: `None`
    - `--trial`: specify the experimental id to differentiate between multiple runs.
    
    Therefore, the command for running **DCD** is:
    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill dcd --model_s resnet8x4 -a 0 -b 1 --trial 1
    ```

    While the command for running **RRD+KD** is:
      ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill rrd --model_s resnet8x4 -a 0 -b 1 --trial 1
    ```
    
2. Combining a distillation objective with KD is simply done by setting `-a` as a non-zero value.

   The command for running **DCD+KD** is:
    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill dcd --model_s resnet8x4 -a 1 -b 1 --trial 1     
    ```

    While the command for running **RRD+KD** is:
    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill rrd --model_s resnet8x4 -a 1 -b 1 --trial 1     
    ```

### Transfer Learning

Run transfer learning on STL-10 and TinyImageNet-200 using the pretrained student model with frozen backbone is given by:

```
python transfer_student.py --path_s <PATH_TO_WRN_16_2> --model_s wrn_16_2 --dataset stl10 --trial 1     
```

To download TinyImageNet-200, run the following script:
```
sh data/get_tinyimagenet.sh
```

   The default directory to save datasets is `./data/`.


### Benchmark Results on CIFAR-100

Performance is measured by classification accuracy (%)

1. Teacher and student are of the **same** architectural type.

| Teacher <br> Student | wrn-40-2 <br> wrn-16-2 | wrn-40-2 <br> wrn-40-1 | resnet56 <br> resnet20 | resnet110 <br> resnet20 | resnet110 <br> resnet32 | resnet32x4 <br> resnet8x4 |  vgg13 <br> vgg8 |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:------------------:|:------------------:|:--------------------:|:-----------:|
| Teacher <br> Student |    75.61 <br> 73.26    |    75.61 <br> 71.98    |    72.34 <br> 69.06    |     74.31 <br> 69.06    |     74.31 <br> 71.14    |      79.42 <br> 72.50     | 74.64 <br> 70.36 |
| KD | 74.92 | 73.54 | 70.66 | 70.67 | 73.08 | 73.33 | 72.98 |
| FitNet | 73.58 | 72.24 | 69.21 | 68.99 | 71.06 | 73.50 | 71.02 |
| AT | 74.08 | 72.77 | 70.55 | 70.22 | 72.31 | 73.44 | 71.43 |
| SP | 73.83 | 72.43 | 69.67 | 70.04 | 72.69 | 72.94 | 72.68 |
| CC | 73.56 | 72.21 | 69.63 | 69.48 | 71.48 | 72.97 | 70.71 |
| VID | 74.11 | 73.30 | 70.38 | 70.16 | 72.61 | 73.09 | 71.23 |
| RKD | 73.35 | 72.22 | 69.61 | 69.25 | 71.82 | 71.90 | 71.48 |
| PKT | 74.54 | 73.45 | 70.34 | 70.25 | 72.61 | 73.64 | 72.88 |
| AB | 72.50 | 72.38 | 69.47 | 69.53 | 70.98 | 73.17 | 70.94 |
| FT | 73.25 | 71.59 | 69.84 | 70.22 | 72.37 | 72.86 | 70.58 |
| FSP | 72.91 | N/A | 69.95 | 70.11 | 71.89 | 72.62 | 70.23 |
| NST | 73.68 | 72.24 | 69.60 | 69.53 | 71.96 | 73.30 | 71.53 |
| CRD | 75.48 | 74.14 | 71.16 | 71.46 | 73.48 | 75.51 | 73.94 |
| CRD+KD | 75.64 | 74.38 | 71.63 | 71.56 | 73.75 | 75.46 | 74.29 |
| DCD (ours) | 74.99 | 73.69 | 71.18 | 71.00 | 73.12 | 74.23 | 73.22 |
| DCD+KD (ours) | 76.06 | 74.76 | 71.81 | 72.03 | 73.62 | 75.09 | 73.95 |
| RRD (ours) | 75.33 | 73.55 | 70.71 | 70.72 | 73.10 | 74.48 | 73.99 |
| RRD+KD (ours) | 75.66 | 74.67 | 72.19 | 71.74 | 73.54 | 75.08 | 74.32 |


2. Teacher and student are of **different** architectural type.

| Teacher <br> Student | vgg13 <br> MobileNetV2 | ResNet50 <br> MobileNetV2 | ResNet50 <br> vgg8 | resnet32x4 <br> ShuffleNetV1 | resnet32x4 <br> ShuffleNetV2 | wrn-40-2 <br> ShuffleNetV1 |
|:---------------:|:-----------------:|:--------------------:|:-------------:|:-----------------------:|:-----------------------:|:---------------------:|
| Teacher <br> Student |    74.64 <br> 64.60    |      79.34 <br> 64.60     |  79.34 <br> 70.36  |       79.42 <br> 70.50       |       79.42 <br> 71.82       |      75.61 <br> 70.50      |
| KD | 67.37 | 67.35 | 73.81 | 74.07 | 74.45 | 74.83 |
| FitNet | 64.14 | 63.16 | 70.69 | 73.59 | 73.54 | 73.73 |
| AT | 59.40 | 58.58 | 71.84 | 71.73 | 72.73 | 73.32 |
| SP | 66.30 | 68.08 | 73.34 | 73.48 | 74.56 | 74.52 |
| CC | 64.86 | 65.43 | 70.25 | 71.14 | 71.29 | 71.38 |
| VID | 65.56 | 67.57 | 70.30 | 73.38 | 73.40 | 73.61 |
| RKD | 64.52 | 64.43 | 71.50 | 72.28 | 73.21 | 72.21 |
| PKT | 67.13 | 66.52 | 73.01 | 74.10 | 74.69 | 73.89 |
| AB | 66.06 | 67.20 | 70.65 | 73.55 | 74.31 | 73.34 |
| FT | 61.78 | 60.99 | 70.29 | 71.75 | 72.50 | 72.03 |
| NST | 58.16 | 64.96 | 71.28 | 74.12 | 74.68 | 74.89 |
| CRD | 69.73 | 69.11 | 74.30 | 75.11 | 75.65 | 76.05 |
| CRD+KD | 69.94 | 69.54 | 74.58 | 75.12 | 76.05 | 76.27 |
| DCD (ours) | 68.35 | 67.39 | 73.85 | 74.26 | 75.26 | 74.98 |
| DCD+KD (ours) | 69.77 | 70.03 | 74.08 | 76.01 | 76.95 | 76.51 |
| RRD (ours) | 67.93 | 68.84 | 74.01 | 74.11 | 74.80 | 74.98 |
| RRD+KD (ours) | 69.98 | 69.13 | 74.26 | 75.18 | 76.83 | 76.31 |


### Transferability of Representations

Performance is measured by classification accuracy (%)

|  | **CIFAR-100 → STL-10** | **CIFAR-100 → Tiny ImageNet** |
|:---------------------:|:-------------------:|:---------------------:|
| Teacher <br> Student | 68.6 <br> 69.7 | 31.5 <br> 33.7 |
| KD | 70.9 | 33.9 |
| AT | 70.7 | 34.2 |
| FitNet | 70.3 | 33.5 |
| CRD | 71.6 | 35.6 |
| CRD+KD | 72.2 | 35.5 |
| DCD | 71.2 | 35.0 |
| DCD+KD | 72.5 | 36.2 |
| RRD | 71.2 | 34.6 |
| RRD+KD | 71.4 | 34.5 |

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
