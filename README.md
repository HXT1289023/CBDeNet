# Lightweight Visual Perception Architecture for Classroom Behaviour Detection via Hierarchical Feature Fusion and Adaptive Loss
Intelligent education relies on automated visual assessment of student engagement to improve teaching quality. Existing detection methods struggle to balance accuracy and efficiency in dense, occluded classroom environments with limited computational resources. This work presents a lightweight perception framework for university classroom behaviour detection, termed CBDeNet, which integrates hybrid feature enhancement, hierarchical multi-scale fusion, and dynamic adaptive boundary loss. The model employs a dual-branch enhancement module to refine fine-grained feature representation and uses a streamlined feature pyramid to improve multi-scale information aggregation. A dynamic boundary loss is designed to alleviate sample imbalance, and layer-adaptive pruning further compresses the model. Experiments on the SCB-Dataset show that CBDeNet-Tiny achieves 68.51% mean average precision across intersection-over-union thresholds from 0.50 to 0.95 (mAP50:95), with only 1.02 million parameters, indicating potential for resource-constrained deployment.
<p align="center">
  <img src="images/1.png" width="70%">
</p>  


## Dataset  
The experimental performance validation is based on the [SCB-Dataset3](https://github.com/Whiffe/SCB-dataset), proposed by Yang et al., which comprises 5,686 images and 45,578 annotations covering a wide range of teaching scenarios from kindergarten to university level. To align precisely with the requirements of the university lecture theatre application, this study primarily selected the SCB3-U subset. This subset contains over 1,000 high-resolution images and more than 19,000 fine-grained annotations derived from multi-angle classroom recordings. The subset encompasses 6 typical behavioural categories, but the sample distribution is uneven. Reading and mobile phone usage account for 7,826 and 6,976 annotations, respectively; meanwhile, bowing the head comprises just 947 annotations. While this scarcity of samples accurately reflects classroom dynamics, it also predisposes models to overfitting high-frequency, simple samples during training, which poses significant challenges for object detection tasks.


## How to Use  
### Requirements  
#### Note  

The code for this model requires execution within the mmdetection environment, thus necessitating the prior installation of the requisite runtime environment. The installation repository is located at https://github.com/open-mmlab/mmdetection, with installation instructions available at https://mmdetection.readthedocs.io/en/latest/get_started.html.  
The model's source code is housed within the Polyp-Lite folder. The configuration file path is located at [configs/CBDeNet.py](configs/CBDeNet.py)


1.Train  

`python tools/train.py configs/my_model/<your-config-file>`  

2.Test  

`python tools/test.py <your-config-file> <your-model-weights-file> --out <save-pickle-path>`  

## Experiments Results  
### Performance Comparison with SOTA Models

The following table reports detection accuracy, parameter count, computational cost, and model size on the SCB-Dataset3.

| Model | mAP50:95 (%) | Params (M) | FLOPs (G) | Model Size (MB) |
|---|---:|---:|---:|---:|
| YOLOv8n [30] | 68.62 | 2.69 | 6.8 | 5.4 |
| YOLOv10n [31] | 68.69 | 2.27 | 6.5 | 5.5 |
| Faster R-CNN [32] | 65.32 | 41.35 | 214.0 | 83.4 |
| Cascade R-CNN [33] | 65.81 | 69.16 | 186.2 | 125.7 |
| DETR-R50 [34] | 63.10 | 41.56 | 85.74 | 81.4 |
| RT-DETR-R18 [35] | 69.87 | 19.88 | 56.9 | 38.6 |
| LeYOLO-Nano [36] | 56.22 | 1.09 | 2.5 | 2.7 |
| PLA-YOLO11n [7] | 51.36 | 1.84 | 3.08 | 3.8 |
| PACR-DETR [16] | 64.52 | 19.61 | 55.2 | 37.5 |
| SSD-Lite [37] | 51.62 | 3.03 | 0.7 | 5.0 |
| CBDeNet | 69.87 | 2.56 | 8.1 | 5.1 |
| CBDeNet-Tiny | 68.51 | 1.02 | 3.0 | 2.2 |


## Generalisation Ability  
### Performance Comparison on the SCB3-S Subset

| Model | Hand-raising mAP50:95 (%) | Reading mAP50:95 (%) | Writing mAP50:95 (%) | All mAP50:95 (%) | Params (M) | FLOPs (G) | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 55.50 | 53.44 | 41.48 | 50.14 | 2.59 | 6.3 | 138.3 |
| YOLOv8n | 55.73 | 55.49 | 43.28 | 51.50 | 2.69 | 6.8 | 151.6 |
| CBDeNet-Tiny | 56.44 | 55.87 | 43.95 | 52.09 | 1.02 | 3.0 | 160.9 |

## Visualisation  
Fig. 10 further analyses the attention distribution of CBDeNet using Grad-CAM visualisation. The baseline model shows relatively diffuse activation in distant or cluttered classroom regions, indicating that its responses may be affected by background structures such as desks, seats, or neighbouring students. By contrast, CBDeNet produces more concentrated activation around behaviour-related regions, including the upper body, head area, and local interaction regions of students. This more focused response is consistent with the design motivation of HFEBlock and HiFusionFPN, which aim to strengthen fine-grained representation and multi-scale semantic propagation. The Grad-CAM results provide qualitative support for CBDeNet’s extraction of behaviour-oriented visual cues in dense classroom scenes.  

<p align="center">
  <img src="images/2.png" width="80%">
</p> 

