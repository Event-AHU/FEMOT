<div align="center">

<img src="https://github.com/Event-AHU/FEMOT/blob/main/pictures/FEMOT_log.png" width="600">

# RGB-Event based Multi-Object Tracking 

</div>

* **FEMOT: Multi-Object Tracking using Frame and Event Cameras**,
  Shiao Wang, Xiao Wang, Chao Wang, Yitao Li, Menghao Liu, Bo Jiang, Yaowei Wang, Yonghong Tian, Jin Tang
  [[Paper](https://arxiv.org/abs/2606.14094)]
  [[Code](https://github.com/Event-AHU/FEMOT)] 


# :dart: Abstract       
Conventional RGB cameras have been widely used in multi-object tracking due to their ability to capture rich appearance and semantic information. However, their performance is often degraded under complex real-world challenges, such as motion blur, low illumination, and overexposure. Bio-inspired event cameras offer high temporal resolution and high dynamic range, providing complementary cues under extreme scenarios. Nevertheless, RGB-event multi-object tracking remains underexplored due to the lack of large-scale and well-annotated datasets. To address this issue, we propose FEMOT, a large-scale RGB-event multi-object tracking dataset that covers diverse real-world scenarios and 14 challenging attributes. With both RGB and event data as well as high-quality annotations, FEMOT provides a reliable platform for systematically evaluating RGB-event multi-object tracking methods. Based on FEMOT, we retrain and evaluate over ten strong trackers, thereby establishing a comprehensive benchmark for future research. Furthermore, we propose FEMOTR, a multimodal tracking framework that decouples RGB and event features and fuses them in the frequency domain, thereby effectively exploiting their complementary characteristics for robust object localization and identity association. Extensive experiments on FEMOT and DSEC-MOT datasets demonstrate the effectiveness of the proposed method.


<p align="center">
  <img src="https://github.com/Event-AHU/FEMOT/blob/main/pictures/First_Image.jpg" alt="First_Image" width="900"/>
</p>

  
# :collision: Update Log 
* :fire: [2026.06.18] arXiv paper, dataset, and benchmark results are all released [[arXiv](https://arxiv.org/abs/2606.14094)]

# :hammer: Environment 
```
conda create -n FEMOT python=3.10
conda activate FEMOT
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install matplotlib pyyaml scipy tqdm tensorboard
pip install opencv-python
```

## Train & Test
```
# train
bash train.sh

# test
bash test.sh
```

# :dvd: FEMOT Dataset 

<p align="center">
  <img src="https://github.com/Event-AHU/FEMOT/blob/main/pictures/DATASET_Attributes.jpg" alt="DATASET_Attributes" width="800"/>
</p>

:floppy_disk: **Baidu Netdisk**: [FEMOT_Dataset](https://pan.baidu.com/s/1subwHuu9sOOC2rjtufPJbw?pwd=AHUE)

:floppy_disk: **Dropbox**: coming soon...

The directory should have the following format:
```Shell
├── FEMOT_Dataset
    ├── labels_with_ids
    ├── annotations
    ├── seqmaps
    ├── train (70 sequences)
        ├── dvSave-2024_06_03_19_25_56
            ├── dvSave-2024_06_03_19_25_56_aps
            ├── dvSave-2024_06_03_19_25_56_dvs
            ├── gt
            ├── det
            ├── seqinfo.ini
        ├── ... 
    ├── test (10 sequences)
        ├── dvSave-2024_06_04_18_07_37
            ├── dvSave-2024_06_04_18_07_37_aps
            ├── dvSave-2024_06_04_18_07_37_dvs
            ├── gt
            ├── det
            ├── seqinfo.ini
        ├── ...
    ├── val (10 sequences)
        ├── dvSave-2024_06_04_17_29_44
            ├── dvSave-2024_06_04_17_29_44_aps
            ├── dvSave-2024_06_04_17_29_44_dvs
            ├── gt
            ├── det
            ├── seqinfo.ini
        ├── ...
```

<p align="center">
  <img src="https://github.com/Event-AHU/FEMOT/blob/main/pictures/Benchmark_result.jpg" alt="Benchmark_result" width="800"/>
</p>



### Citation 
If you find this work useful for your research, please give us a star and cite the following papers: 

```
@misc{wang2026femot,
      title={FEMOT: Multi-Object Tracking using Frame and Event Cameras}, 
      author={Shiao Wang and Xiao Wang and Chao Wang and Yitao Li and Menghao Liu and Bo Jiang and Yaowei Wang and Yonghong Tian and Jin Tang},
      year={2026},
      eprint={2606.14094},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.14094}, 
}


```


  
