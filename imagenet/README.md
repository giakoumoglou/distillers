## Distillers

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

This repo is based on [SimKD implementation](https://github.com/DefangChen/SimKD) and [SemCKD implementation](https://github.com/DefangChen/SemCKD).

## State-of-the-Art Knowledge Distillation Methods

1. `KD` - Distilling the Knowledge in a Neural Network  
2. `FitNet` - Fitnets: Hints for Thin Deep Nets  
3. `AT` - Paying More Attention to Attention: Improving the Performance of CNNs via Attention Transfer  
4. `SP` - Similarity-Preserving Knowledge Distillation  
5. `VID` - Variational Information Distillation for Knowledge Transfer 
6. `CRD` - Contrastive Representation Distillation  
7. `SRRL` - Knowledge distillation via softmax regression representation learning
8. `SemCKD` - Cross-Layer Distillation with Semantic Calibration
9. `SimKD` - Knowledge Distillation with the Reused Teacher Classifier
10. `RRD` - Relational Representation Distillation
11. `DCD` - Discriminative and Consistent Distillation


### Installation

```bash
git clone https://github.com/giakoumoglou/distillers.git
cd distillers
cd imagenet
pip install -r requirements.txt
```

We use [DALI](https://github.com/NVIDIA/DALI) for data loading and pre-processing.

Fetch the pretrained teacher models by:

```bash
# CIFAR-100
python train_teacher.py --batch_size 64 --epochs 240 --dataset cifar100 --model resnet32x4 --learning_rate 0.05 --lr_decay_epochs 150,180,210 --weight_decay 5e-4 --trial 0 --gpu_id 0

# ImageNet
python train_teacher.py --batch_size 256 --epochs 120 --dataset imagenet --model ResNet18 --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 --num_workers 32 --gpu_id 0,1,2,3 --dist-url tcp://127.0.0.1:23333 --multiprocessing-distributed --dali gpu --trial 0 
```

The pretrained teacher models used in our paper are provided in this [link](https://drive.google.com/drive/folders/1j7b8TmftKIRC7ChUwAqVWPIocSiacvP4?usp=sharing). 

This will save the models to `save/models`.


### Knowledge Transfer

```bash
# CIFAR-100
python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill simkd --model_s resnet8x4 -c 0 -d 0 -b 1 --trial 0

# ImageNet
python train_student.py --path-t './save/teachers/models/ResNet50_vanilla/ResNet50_best.pth' --batch_size 256 --epochs 120 --dataset imagenet --model_s ResNet18 --distill simkd -c 0 -d 0 -b 1 --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 --num_workers 32 --gpu_id 0,1,2,3 --dist-url tcp://127.0.0.1:23444 --multiprocessing-distributed --dali gpu --trial 0 
```
More scripts are provided in `./scripts`
