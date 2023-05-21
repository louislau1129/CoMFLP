## CoMFLP: Correlation Measure based Fast Search on Layer Pruning
This repository contains code for the implementation of CoMFLP, a fast search layer pruning algorithm.

*The corresponding paper will appear in InterSpeech 2023*

### Background:
> The recently proposed DeepNet [^1] model scaled the transformer up to 1,000 layers. Despite the significant performance improvement, the model is inefficient and difficult to apply on the edge device. Layer pruning (LP) is a reasonable method considering the layer redundancy. However, previous LP methods mostly rely on an task-specific evaluation metric to search, which is quite time-consuming especially when layers are very deep. In contrast to previous methods, the proposed CoMFLP has a very fast search speed with a consant time complexity. Also the searched pruning strategy is shown to be high-quality and can serve as a good strating point for later fine-tuning.


### Python Env:
The complete conda environment is exported as `environment.yml`, and `python==3.8.10`.

### Code Structure and Usage:
*We use **clip12to6** as an example for illustration:*

- Correlation Matrix computation: two correlation measure methods are adopted here, namely SVCCA [^2] and DC [^3]. 

  1. Compute correlation matrix using SVCCA:
  ```bash
  python svcca_analysis.py --batch_size 32 --iter_num 300 --thre 0.99 --mode U
  ```
  2. Compute correlation matrix using DC:
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
where `select_measure` can switch to `svcca`. Note that the hyperparamteres for correlation matrix computation should be consitent with the above setting: For SVCCA, you should explicitly provide arguments `bs=32`, `iter=300`, `svcca_mode=U` and `thre=0.99`; For DC, `bs=4` and `iter=10`.

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

- (Optional) Evaluate the (original/pruned/fine-tuned) ASR model on the test set using WER (CER for Chinese character).
```bash
# Merge the 10 best-performed fine-tuned models during the fine-tuning process
python average_model_pts.py --exp_dir clip12to6_exp1 \
                            --last_epoch 50 \
                            --num 10

# By default, it will select the asr_avg_valid10.pt under the output_dir to decode 
python test_asr.py --output_dir clip12to6_exp1 \
                   --resume_epoch 50 \
                   --num_layers 6
```

- (Optional) Implement the WER-based GLP (greedy layer pruning) method:
```bash
cd search_strategy 
python wer_based_greedy_search.py --num_layers 12 \
                                  --num_clip_layers 6
```
Although performed in a greedy manner, still very time-consuming for large `num_layers`. 


## Experimental Data and Model
- **Data**:

> We test the effectiveness of the proposed **CoMFLP** on the ASR task. Specificlly, the dataset used is AISHELL-1 [^4] for Chinese speech recognition. The data_resource paths shown in the scripts are as follows:
```bash
    wav_scp = "/home/userxx/research/prep_mfa/data/AISHELL1/aishell_train/wav.scp"
    text_file = "/home/userxx/research/prep_mfa/data/AISHELL1/aishell_train/text" 
```
We put the above `AISHELL1` directory in the current repository for your reference, the data format follows kaldi style.

- **Model**:

> The original ASR deep model is exported from the Huggingface website. Due to the limited computational resources, we only test `12-layer` and `24-layer` transformers. (Note that only `12-layer` example code is given in this repository, and it is easy to extend it to `24-layer` or other number of layers.)

**We really appreciate if interested people could help using this CoMFLP method to test on much deeper layers under different tasks. Due to the limited computational resources, we only tested 12-layer with fine-tune.**


## Referenced repositories
1. SVCCA measure computation in `SVCCA.py`: https://github.com/google/svcca
2. DC measure computation in `DC.py`: https://github.com/zhenxingjian/Partial_Distance_Correlation


### References
[^1]: H. Wang, S. Ma, L. Dong, S. Huang, D. Zhang, and F. Wei,
“Deepnet: Scaling transformers to 1,000 layers,” arXiv preprint
arXiv:2203.00555, 2022.

[^2]: M. Raghu, J. Gilmer, J. Yosinski, and J. Sohl-Dickstein, “Svcca:
Singular vector canonical correlation analysis for deep learning
dynamics and interpretability,” Advances in neural information
processing systems, vol. 30, 2017.

[^3]: X. Zhen, Z. Meng, R. Chakraborty, and V. Singh, “On the versatile
uses of partial distance correlation in deep learning,” in Computer
Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel,
October 23–27, 2022, Proceedings, Part XXVI. Springer, 2022,
pp. 327–346.

[^4]: H. Bu, J. Du, X. Na, B. Wu, and H. Zheng, “Aishell-1: An open-
source mandarin speech corpus and a speech recognition base-
line,” in 2017 20th conference of the oriental chapter of the inter-
national coordinating committee on speech databases and speech
I/O systems and assessment (O-COCOSDA). IEEE, 2017, pp.
1–5.
