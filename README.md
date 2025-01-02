# END
Extracting Noise and Darkness: Low-Light Image Enhancement via Dual Prior Guidance

Our paper has been accepted by T-CSVT. 

## Environment
* Python
* Pytorch
* numpy
* tqdm
* pandas

## Trained models
You can download our trained models from [BaiduPan [code:9cje]](https://pan.baidu.com/s/15lE-TkscUfVAgL-SGAvjPQ). 

Place the downloaded models in `END/checkpoints` 


## Train
Download LOL dataset from (https://daooshee.github.io/BMVC2018website/) and change the dataset path in `training.yaml`, and then run

```
python train_denoise.py
```


## Test
test LOL dataset. Change the testdataset path of `test.py` and then run

```
python test.py
```

or unpair dataset run

```
python test_unpair.py
```

## Citation
If you find the code helpful in your research or work, please cite the following paper:
```

```
