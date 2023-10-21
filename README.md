# Counting Crowds in Bad Weather

**[ICCV 2023] Official Pytorch based implementation.** 

[[Paper]](https://arxiv.org/abs/2306.01209) / [[Website]](https://awccnet.github.io/)

<hr />

> **Abstract:** *Crowd counting has recently attracted significant attention in the field of computer vision due to its wide applications to image understanding. Numerous methods have been proposed and achieved state-of-the-art performance for real-world tasks. However, existing approaches do not perform well under adverse weather such as haze, rain, and snow since the visual appearances of crowds in such scenes are drastically different from those images in clear weather of typical datasets. In this paper, we propose a method for robust crowd counting in adverse weather scenarios. Instead of using a two-stage approach that involves image restoration and crowd counting modules, our model learns effective features and adaptive queries to account for large appearance variations. With these weather queries, the proposed model can learn the weather information according to the degradation of the input image and optimize with the crowd counting module simultaneously. Experimental results show that the proposed algorithm is effective in counting crowds under different weather types on benchmark datasets.* 


## Architecture
<table>
  <tr>
    <td align="center"> <img src = "https://github.com/awccnet/AWCC-Net/blob/master/images/architecture.png"> </td>
  </tr>
  <tr>
    <td align="center"><p><b>Overall Architecture</b></p></td>
  </tr>
  <tr>
    <td align="center"> <img src = "https://github.com/awccnet/AWCC-Net/blob/master/images/weather-adaptive-quries.png" height="250"> </td>
  </tr>
  <tr>
    <td align="center"><p><b>Weather-adaptive Queries</b></p></td>
  </tr>
</table>



## Quantitative Result

<table>
  <tr>
    <td align="center"> <img src = "https://github.com/awccnet/AWCC-Net/blob/master/images/table1.png"> </td>
    <td align="center"> <img src = "https://github.com/awccnet/AWCC-Net/blob/master/images/table2.png"> </td>
  </tr>
  <tr>
    <td align="center"><p><b>JHU-Crowd++</b></p></td>
    <td align="center"><p><b>Several Datasets</b></p></td>
  </tr>
</table>



## Usage

#### Pre-trained Models

* **JHU-Crowd++** &rarr; [google drive](https://drive.google.com/file/d/1Tu9VH0FmWyMTTwe8rqQt3gq_U2mUZGY3/view?usp=share_link)

#### Install

```sh
git clone https://github.com/awccnet/AWCC-Net.git
```

#### Inference

```shell
python main.py --dir_path DIR_OF_TEST_IMAGES --checkpoint CHECKPOINT_PATH
```



## Other Works for Image Restoration

You can also refer to our previous works:

* Desnowing &rarr; [[JSTASR]](https://github.com/weitingchen83/JSTASR-DesnowNet-ECCV-2020) (ECCV'20) and [[HDCW-Net]](https://github.com/weitingchen83/ICCV2021-Single-Image-Desnowing-HDCWNet) (ICCV'21)
* Dehazing &rarr; [[PMS-Net]](https://github.com/weitingchen83/PMS-Net) (CVPR'19) and [[PMHLD]](https://github.com/weitingchen83/Dehazing-PMHLD-Patch-Map-Based-Hybrid-Learning-DehazeNet-for-Single-Image-Haze-Removal-TIP-2020) (TIP'20)
* Deraining &rarr; [[ContouletNet]](https://github.com/cctakaet/ContourletNet-BMVC2021) (BMVC'21)
* Image Relighting &rarr; [[MB-Net]](https://github.com/weitingchen83/NTIRE2021-Depth-Guided-Image-Relighting-MBNet) (NTIRE'21 1st solution) and [[S3Net]](https://github.com/dectrfov/NTIRE-2021-Depth-Guided-Image-Any-to-Any-relighting) (NTIRE'21 3 rd solution)
* Multi-Weather Removal &rarr; [[Unified]](https://github.com/fingerk28/Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal) (CVPR'22)



## Citation
Please cite this paper in your publications if it is helpful for your tasks.
```bib
@inproceedings{huang2023counting,
  title={Counting Crowds in Bad Weather},
  author={Zhi-Kai Huang and Wei-Ting Chen and Yuan-Chun Chiang and Sy-Yen Kuo and Ming-Hsuan Yang},
  journal={ICCV},
  year={2023}
}
```

