# Shadow-Aware 3D Gaussian Splatting via Lambertian-Guided Neural Masking

> **Update (June 2026)**

![Teaser](https://github.com/tang-shan/ShadowGS/blob/master/static/images/teaser.png)
When images are collected using mobile phones or other handheld devices in unconstrained environments,
shadows cast by the photographer or surrounding pedestrians are usually captured in the training views.
Vanilla 3D Gaussian Splatting tends to interpret these transient shadows as intrinsic surface properties,
thereby encoding them into the reconstructed material and producing view-dependent artifacts.

---
![Mehtod](https://github.com/tang-shan/ShadowGS/blob/master/static/images/framework.png)
Given training views, a hybrid 3D Gaussian representation models geometry and appearance using hash encoding and lightweight neural predictors.
A shadow-aware masking module first generates a candidate shadow prior from distractor-free images and refines it via residual-driven neural prediction, producing the final shadow mask $\mathcal{M}$.
This mask acts as a pixel-wise reliability map to modulate photometric supervision,
while a Lambertian reflectance constraint suppresses view-dependent compensation within shadow regions.
Both losses are jointly optimized in an end-to-end manner to achieve shadow-robust 3D Gaussian reconstruction.

## Environmental Setups
```bash
git clone https://github.com/tang-shan/ShadowGS.git
cd ShadowGS/src
conda env create -f environment.yml
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```
## Data
The real-world dataset can be downloaded from [here](https://pan.baidu.com/s/1O3oddWDeUhkZmR-Zst0vJA?pwd=ddk9).
The synthetic dataset can be downloaded from [here](https://pan.baidu.com/s/1AZmu_WvzA3ZFSbhRNvjdyQ?pwd=hgf3).
Both datasets provide predefined training and test splits via \texttt{train_list.txt} and \texttt{test\_list.txt}. These files are used to strictly separate training and evaluation views.


## Data Preparation
For Shadow-distractor scene construction, using 3D Gaussian Splatting and clean image dataset to prepare shadow mask. Put the mask into the "shadow_masks" folder.
For object-distractor scene construction, you need to generate the object segmentation masks using SegmentAnything. Note that you need to install SegmentAnything and use the latest vit_b checkpoint. See [here](https://github.com/facebookresearch/segment-anything) for a tutorial.



## Training
```
python train.py -s path/to/data
```


## Acknowledgments
Parts of this project page were adopted from the [Nerfies](https://nerfies.github.io/) page.

