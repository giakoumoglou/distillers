## Representation Distillation

This is a PyTorch implementation of the [RRD paper](https://arxiv.org/abs/2407.12073):
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

It also includes the implementation of and [DCD paper](https://arxiv.org/abs/2407.11802):

```
@misc{giakoumoglou2024discriminative,
      title={Discriminative and Consistent Representation Distillation}, 
      author={Nikolaos Giakoumoglou and Tania Stathaki},
      year={2024},
      eprint={2407.11802},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.11802}, 
}
```

This repo is based on [RepDistiller implementation](https://github.com/HobbitLong/RepDistiller).

### State-of-the-art Knowledge Distillation Methods

This repository benchmarks the following knowledge distillation methods in PyTorch:

1. `KD` - Distilling the Knowledge in a Neural Network  
2. `FitNet` - Fitnets: Hints for Thin Deep Nets  
3. `AT` - Paying More Attention to Attention: Improving the Performance of CNNs via Attention Transfer  
4. `SP` - Similarity-Preserving Knowledge Distillation  
5. `CC` - Correlation Congruence for Knowledge Distillation  
6. `VID` - Variational Information Distillation for Knowledge Transfer  
7. `RKD` - Relational Knowledge Distillation  
8. `PKT` - Probabilistic Knowledge Transfer for Deep Representation Learning  
9. `AB` - Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons  
10. `FT` - Paraphrasing Complex Network: Network Compression via Factor Transfer  
11. `FSP` - A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning  
12. `NST` - Like What You Like: Knowledge Distill via Neuron Selectivity Transfer  
13. `CRD` - Contrastive Representation Distillation  
14. `RRD` - Relational Representation Distillation
15. `DCD` - Discriminative and Consistent Distillation

### Installation

Open your terminal and run the following command to clone the repository:

```bash
git clone https://github.com/giakoumoglou/distillers.git
cd distillers
cd cifar
pip install -r requirements.txt
```

Fetch the pretrained teacher models by:

```bash
sh scripts/fetch_pretrained_teachers.sh
```

This will save the models to `save/models`

Download TinyImageNet-200:
```bash
sh data/get_tinyimagenet.sh
```

Datasets are saved in `./data/` by default. CIFAR-100 and STL-10 are downloaded automatically.

### Knowledge Transfer


Run distillation by following commands in `scripts/run_cifar_distill.sh`. An example of running Geoffrey's original Knowledge Distillation (KD) is given by:

```bash
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

While the command for running **RRD** is:
```
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill rrd --model_s resnet8x4 -a 0 -b 1 --trial 1
```

Combining a distillation objective with KD is simply done by setting `-a` as a non-zero value.

The command for running **DCD+KD** is:

```
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill dcd --model_s resnet8x4 -a 1 -b 1 --trial 1     
```

While the command for running **RRD+KD** is:
```
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill rrd --model_s resnet8x4 -a 1 -b 1 --trial 1     
```

More scripts are provided in `./scripts`

### Transfer Learning

Run transfer learning on STL-10 and TinyImageNet-200:

```bash
python transfer_student.py --path_s <PATH_TO_WRN_16_2> --model_s wrn_16_2 --dataset stl10 --trial 1
```
