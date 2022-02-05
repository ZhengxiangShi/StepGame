# Code

## Environment
- python=3.8.5
- torch=1.7.1

## Settings
* During the training, clean samples with k=1,2,3,4,5 are fed into models together, namely there are totally 5 * 10k samples for training. 
* During the validation and testing, noise samples with k=1,2,3,4,5 are used in the table 2 and samples with k=6,7,8,9,10 are used in the table 3. 
* **Please note that not all samples in the dataset are used in our paper.** 
* `babi_format` is the same dataset where the only data format is different.
  
## Training
```
python train.py --path saved_models
```

## Evaluation
Please change the data_path in the config file from "clean" to "noise".
```
python train.py --eval-test --path saved_models
```

## Code Reference
```
[1]: https://github.com/thaihungle/SAM
[2]: https://github.com/APodolskiy/TPR-RNN-Torch
```