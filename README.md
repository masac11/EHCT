##  Fusing Convolution and Transformer-based Spiking Neural Network for Object Tracking in Event Domain
 
The code is based on [SiamFC++](https://github.com/MegviiDetection/video_analyst) 

### Installation

```
conda create -n ehct python=3.7
conda activate ehct
pip install -r requirements.txt
```

### Test

Download our trained models from [Google Driver](https://drive.google.com/drive/folders/1KwTM4lbSIi-k6hYQwum4Ty_5zIos6K9K?usp=sharing) and put it into ./models.

- VisEvent

  Download our preprocessed [test dataset](https://drive.google.com/drive/folders/1KwTM4lbSIi-k6hYQwum4Ty_5zIos6K9K?usp=sharing) of VisEvent and  change dataset path at line 40 in experiments/visevent.yaml.

  ```python
    data_root="/your_visevent_data_path/img_120_split"
  ```  

  Run

  ```python
    python main/test.py --config experiments/visevent.yaml
  ``` 

- FE240hz

  Download our preprocessed [test dataset](https://drive.google.com/drive/folders/1KwTM4lbSIi-k6hYQwum4Ty_5zIos6K9K?usp=sharing) of FE240hz and  change dataset path at line 40 in experiments/visevent.yaml.

  ```python
    data_root="/your_fe240hz_data_path/img_120_split"
  ```  

  Run

  ```python
    python main/test.py --config experiments/fe240.yaml
  ``` 

### Evaluate
We use [pytracking](https://github.com/chenxin-dlut/TransT/blob/main/pytracking) to evaluate all trackers. 