# Dataset

## 1. Version
<!-- We provide two formats of our StepGame dataset, which are completely the same except the file format. -->

- `CompleteVersion` is a complete version of our dataset. There are two parts, the clean data part and noise data part. For k from 1 to 10 in each part, it contains 30000, 1000, and 30000 samples for train, valid, and test respectively. However, only part of data in this version is used in our paper, so we also provide `TrainVersion`.
- `TrainVersion` is  used for baselines models in our paper. During the training, clean samples with k=1,2,3,4,5 are fed into models together, namely there are totally 5 * 10k samples for training. During the validation, clean samples with k=1,2,3,4,5 are used. During the testing, noise samples with k=1,2,3,4,5,6,7,8,9,10 are used. In this `TrainVersion` folder, we combine them together here.


<!-- ## 2. Important Note

* Although we provides 30k samples for each value of the k here, only the first 10k samples for each k are used during the training in our paper. 
* Although we provide valid/test sets for clean data and train sets for noise data, they are not used in our paper. 
* We provide more samples for the sake of further analysis, such as the effect of sample sizes.

## 3. Training setting
* During the training, clean samples with k=1,2,3,4,5 are fed into models together, namely there are totally 5 * 10k samples for training. 

*   During the validation and testing, noise samples with k=1,2,3,4,5 are used in the table 2 and samples with k=6,7,8,9,10 are used in the table 3.  -->

<!-- ## 3. Generate more samples
```
python parameterized_step_game_8relation.py --seed 123
``` -->
