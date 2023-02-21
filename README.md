## CoMFLP: Correlation Measure based Fast Search on Layer Pruning
This repository contains code for the implementation of CoMFLP, a fast search layer pruning algorithm.

### Background:
The recently proposed DeepNet [^1] model scaled the transformer up to 1,000 layers. Despite the significant performance improvement, the model is inefficient and difficult to apply on the 
edge device. Layer pruning (LP) is a reasonable method considering the layer redundancy. However, previous LP methods mostly rely on an task-specific evaluation metric to search, which is
quite time-consuming especially when layers are very deep. In contrast to previous methods, the proposed CoMFLP has a very fast search speed with a consant time complexity. Also the searched
pruning strategy is shown to be high-quality and can serve as a good strating point for later fine-tuning.

### Code Structure and Usage:
- Correlation Matrix Computation: Two correlation measure methods are adopted here, namely SVCCA [^2] and DC [^3]. 

  1. Compute correlation matrix using SVCCA:
  ```
  python svcca_analysis.py --batch_size 32 --iter_num 300 \
                            --thre 0.99 --mode U
  ```
- csdcsd


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

