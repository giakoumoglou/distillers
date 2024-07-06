## Distillers

This is a PyTorch implementation of the [XXX paper](https://arxiv.org/abs/XXXX.XXXXX):

```
@Article{giakoumoglou2024invariant,
  author  = {Giakoumoglou Nikolaos and Stathaki Tania},
  title   = {Invariant Consistency for Knowledge Distillation},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2024},
}
```

It also includes the implementation of and [XXX paper](https://arxiv.org/abs/XXXX.XXXXX):

```
@Article{giakoumoglou2024relational,
  author  = {Giakoumoglou Nikolaos and Stathaki Tania},
  title   = {Relational Consistency for Knowledge Distillation},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2024},
}
```

This repo is based on [RepDistiller implementation](https://github.com/HobbitLong/RepDistiller): [Paper](http://arxiv.org/abs/1910.10699)


### Benchmarks 13 state-of-the-art knowledge distillation methods in PyTorch

(KD) - Distilling the Knowledge in a Neural Network  
(FitNet) - Fitnets: hints for thin deep nets  
(AT) - Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer  
(SP) - Similarity-Preserving Knowledge Distillation  
(CC) - Correlation Congruence for Knowledge Distillation  
(VID) - Variational Information Distillation for Knowledge Transfer  
(RKD) - Relational Knowledge Distillation  
(PKT) - Probabilistic Knowledge Transfer for deep representation learning  
(AB) - Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons  
(FT) - Paraphrasing Complex Network: Network Compression via Factor Transfer  
(FSP) - A Gift from Knowledge Distillation:
    Fast Optimization, Network Minimization and Transfer Learning  
(NST) - Like what you like: knowledge distill via neuron selectivity transfer  
(CRD) - Contrastive Representation Distillation  

### Installation

1. Open your terminal and run the following command to clone the repository:
   ```sh
   git clone https://github.com/giakoumoglou/distillers.git
   ```

2. Change into the directory of the cloned repository and nstall the necessary dependencies using `pip`:

   ```sh
   cd distillers
   pip install -r requirements.txt
   ```

3. This repo was tested with Ubuntu 16.04.5 LTS, Python 3.5, PyTorch 0.4.0, and CUDA 9.0. But it should be runnable with recent PyTorch versions >=0.4.0

### Running

1. Fetch the pretrained teacher models by:

    ```
    sh scripts/fetch_pretrained_teachers.sh
    ```
   which will download and save the models to `save/models`
   
2. Run distillation by following commands in `scripts/run_cifar_distill.sh`. An example of running Geoffrey's original Knowledge Distillation (KD) is given by:

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
    
    Therefore, the command for running ICD is:
    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill icd --model_s resnet8x4 -a 0 -b 1 --trial 1
    ```
    
   And the command for running RCD is:
   
   ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill rcd --model_s resnet8x4 -a 0 -b 1 --trial 1
    ```
    
3. Combining a distillation objective with KD is simply done by setting `-a` as a non-zero value.

   The command for running ICD+KD is:
    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill icd --model_s resnet8x4 -a 1 -b 1 --trial 1     
    ```

    And the command for running RCD+KD is:
    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill rcd --model_s resnet8x4 -a 1 -b 1 --trial 1     
    ```

4. Run transfer learning on STL-10 and TinyImageNet-200 using the pretrained student model with frozen backbone is given by:

    ```
    python transfer_student.py --path_s <PATH_TO_WRN_16_2> --model_s wrn_16_2 --dataset stl10 --trial 1     
    ```

5. (optional) Train teacher networks from scratch. Example commands are in `scripts/run_cifar_vanilla.sh`

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
