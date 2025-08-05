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
@ARTICLE{10718327,
  author={Wang, Huake and Yan, Xiaoyang and Hou, Xingsong and Zhang, Kaibing and Dun, Yujie},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Extracting Noise and Darkness: Low-Light Image Enhancement via Dual Prior Guidance}, 
  year={2025},
  volume={35},
  number={2},
  pages={1700--1714},
  doi={10.1109/TCSVT.2024.3480930}}

```
