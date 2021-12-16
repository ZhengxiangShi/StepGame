# Code

## Environment
- python=3.8.5
- torch=1.7.1
   
## Training
```
python train.py --path saved_models
```

## Evaluation
Please change the data_path in the config file from "clean" to "noise".
```
python train.py --eval-test --path saved_models
```