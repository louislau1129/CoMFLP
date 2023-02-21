## CoMFLP: Correlation Measure based Fast Search on Layer Pruning
This repository contains code for the implementation of CoMFLP, a fast search layer pruning algorithm.

### Background:
The recently proposed DeepNet [^1] model scaled the transformer up to 1,000 layers. Despite the significant performance improvement, the model is inefficient and difficult to apply on the 
edge device. Layer pruning (LP) is a reasonable method considering the layer redundancy. However, previous LP methods mostly rely on an task-specific evaluation metric to search, which is
quite time-consuming especially when layers are very deep. In contrast to previous methods, the proposed CoMFLP has a very fast search speed with a consant time complexity. Also the searched
pruning strategy is shown to be high-quality and can serve as a good strating point for later fine-tuning.

### Code Structure and Usage:
*We use clip12to6 as an example for illustration:*
- Correlation Matrix Computation: Two correlation measure methods are adopted here, namely SVCCA [^2] and DC [^3]. 

  1. Compute correlation matrix among layers using SVCCA:
  ```bash
  python svcca_analysis.py --batch_size 32 --iter_num 300 --thre 0.99 --mode U
  ```
  2. Compute correlation matrix among layers using DC:
  ```bash
  python dc_analysis.py --batch_size 4 --iter_num 10
  ```
- Perform Coarse Search on the correlation matrix:
```bash
cd search_strategy
python search_prune_then_verify_mix.py --num_layers 12 \
                                       --num_clip_layers 6 \
                                       --search_mode beam \
                                       --select_measure dc \
                                       --coarse_search_only true
```
where `select_measure` can switch to `svcca`. Note that the hyperparamteres for correlation measure computation should be consitent with the above setting: For SVCCA, you should provide arguments `bs=32`, `iter=300`, `svcca_mode=U` and `thre=0.99`; For DC, `bs=4` and `iter 10`.

- Perform Fine-grained Search on top of the candidates provided by Coarse Search:
```bash
cd search_strategy
python search_prune_then_verify_mix.py --num_layers 12 \
                                       --num_clip_layers 6 \
                                       --search_mode beam \
                                       --select_measure dc 
```
Usually, you just need to run this one step to perform coarse search and fine-grained search sequentially.

- (Optional) Fine-tuning based on a pruned model (select the desired pruning strategy from the previous step)
```bash
python clip_finetune_train.py --output_dir clip12to6_exp1 \
                              --resume_epoch 0 \
                              --num_layers 12 \
                              --num_clip_layers 6
```
By default, it will fiine-tune the pruned model for 50 epochs, you can specify the `resume_epoch`. The fine_tuned model and the corresponding training log file `train.log` will be stored in the specified `output_dir` under `exp` dir.


## Referenced repositories
1. SVCCA measure computation: https://github.com/google/svcca
2. DC measure computation: https://github.com/zhenxingjian/Partial_Distance_Correlation


### References
[^1]: H. Wang, S. Ma, L. Dong, S. Huang, D. Zhang, and F. Wei,
“Deepnet: Scaling transformers to 1,000 layers,” arXiv preprint
arXiv:2203.00555, 2022

[^2]: M. Raghu, J. Gilmer, J. Yosinski, and J. Sohl-Dickstein, “Svcca:
Singular vector canonical correlation analysis for deep learning
dynamics and interpretability,” Advances in neural information
processing systems, vol. 30, 2017

[^3]: X. Zhen, Z. Meng, R. Chakraborty, and V. Singh, “On the versatile
uses of partial distance correlation in deep learning,” in Computer
Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel,
October 23–27, 2022, Proceedings, Part XXVI. Springer, 2022,
pp. 327–346.

