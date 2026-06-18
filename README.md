# RGB-Event-based Multi-Object Tracking 


## Works maintained in this GitHub 

* **FEMOT: Multi-Object Tracking using Frame and Event Cameras**,
  Shiao Wang, Xiao Wang, Chao Wang, Yitao Li, Menghao Liu, Bo Jiang, Yaowei Wang, Yonghong Tian, Jin Tang
  [[Paper](https://arxiv.org/abs/2606.14094)]
  [[Code](https://github.com/Event-AHU/FEMOT)] 


# :dart: Abstract       
Conventional RGB cameras have been widely used in multi-object tracking due to their ability to capture rich appearance and semantic information. However, their performance is often degraded under complex real-world challenges, such as motion blur, low illumination, and overexposure. Bio-inspired event cameras offer high temporal resolution and high dynamic range, providing complementary cues under extreme scenarios. Nevertheless, RGB-event multi-object tracking remains underexplored due to the lack of large-scale and well-annotated datasets. To address this issue, we propose FEMOT, a large-scale RGB-event multi-object tracking dataset that covers diverse real-world scenarios and 14 challenging attributes. With both RGB and event data as well as high-quality annotations, FEMOT provides a reliable platform for systematically evaluating RGB-event multi-object tracking methods. Based on FEMOT, we retrain and evaluate over ten strong trackers, thereby establishing a comprehensive benchmark for future research. Furthermore, we propose FEMOTR, a multimodal tracking framework that decouples RGB and event features and fuses them in the frequency domain, thereby effectively exploiting their complementary characteristics for robust object localization and identity association. Extensive experiments on FEMOT and DSEC-MOT datasets demonstrate the effectiveness of the proposed method.
  
# :collision: Update Log 


# :hammer: Environment 


# :dvd: FEMOT Dataset 

:floppy_disk: **Baidu Netdisk**: [FEMOT_Dataset](https://pan.baidu.com/s/1subwHuu9sOOC2rjtufPJbw?pwd=AHUE)

:floppy_disk: **Dropbox**: coming soon...

The directory should have the following format:
```Shell
├── FEMOT_Dataset
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

   ├── labels_with_ids
   ├── annotations
   ├── seqmaps
```

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


  
