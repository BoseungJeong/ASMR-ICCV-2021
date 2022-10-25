# ASMR: Learning Attribute-Based Person Search with Adaptive Semantic Margin Regularizer
This repository is official implementation of "[ASMR: Learning Attribute-Based Person Search with Adaptive Semantic Margin Regularizer](https://openaccess.thecvf.com/content/ICCV2021/papers/Jeong_ASMR_Learning_Attribute-Based_Person_Search_With_Adaptive_Semantic_Margin_Regularizer_ICCV_2021_paper.pdf)" (ICCV 2021)

__Requirements__ 
*  Python = 3.7.7
*  Pytorch = 1.6.0

__Installation__
```bash
git clone https://github.com/BoseungJeong/ASMR.git
cd ASNR
conda env create -f ASMR.yaml
conda activate ASMR
```

__Download data__
[Market1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf) [Market-1501 attribute](https://arxiv.org/abs/1703.07220) (`market1501`)

# GPU information
* NVIDIA Titan XP




# Training and Evaluation
After installing requirements and downloading datasets, you can run our model by following:
* Train and evaluation on Market-1501
```bash
sh Train_and_Test_ASMR_Market.sh
```
