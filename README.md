[![Build Status](https://travis.ibm.com/art-zurich/spatial-heterogeneity.svg?token=bmUqdLriQp1g3yv7TJC6&branch=master)](https://travis.ibm.com/art-zurich/spatial-heterogeneity)
[![GitHub Pages](https://img.shields.io/badge/docs-sphinx-blue)](https://ai4scr.github.io/ATHENA/)

![athena logo](tutorials/img/athena_logo.png)

ATHENA is an open-source computational framework written in Python that facilitates the visualization, processing and analysis of (spatial) heterogeneity from spatial omics data. ATHENA supports any spatially resolved dataset that contains spatial transcriptomic or proteomic measurements, including Imaging Mass Cytometry (IMC), Multiplexed Ion Beam Imaging (MIBI), multiplexed Immunohistochemisty (mIHC) or Immunofluorescence (mIF), seqFISH, MERFISH, Visium.

## Main functionalities
![overview](tutorials/img/overview.png)

1. ATHENA accomodates raw multiplexed images from spatial omics measurements. Together with the images, segmentation masks, cell-level, feature-level and sample-level annotations can be uploaded.

2. Based on the cell masks, ATHENA constructs graph representations of the data. The framework currently supports three flavors, namely radius, knn, and contact graphs.

3. ATHENA incorporates a variety of methods to quantify heterogeneity, such as global and local entropic scores. Furthermore, cell type interaction strength scores or measures of spatial clustering and dispersion.

4. Finally, the large collection of computed scores can be extracted and used as input in downstream machine learning models to perform tasks such as clinical data prediction, patient stratification or discovery of new (spatial) biomarkers.

## Manuscript
ATHENA has been published as an Applications Note in _Bioinformatics_ ([Martinelli and Rapsomaniki, 2022](https://academic.oup.com/bioinformatics/article/38/11/3151/6575886)). In our Online [Supplementary material](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bioinformatics/38/11/10.1093_bioinformatics_btac303/1/btac303_supplementary_data.pdf?Expires=1660121444&Signature=nNOXyBaPzIuE5inrrA97SQMVAlmH~A42Phehnla68hL2-G79c1xzI4OFewWLX~l1R9QMGW-7YNX8isTOMD9xzwH9~xCVJxLuBtcpobKOlx16Ha4tEcdme-LiFM7MC4H3LQrQT~~JRMaTNCN7TSDn8pcfkLsBK1WHbZ9C8qTAwUfJek~tt6fzH~ZwA5dJ0KZ49HzZpwA1DvYU0luxJbgzj3mSs6OczQw3b3B6qm7EV45ijdR447jfCLsz5pbtZ2J6yAuKbsEN5KmkIbUfujMo9vw7YrQOwJjaMWol1Cus5mbebpB6QOfP5jGU7LfiFR1SPQTZ~A0phAssndE~0p1ilQ__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA) you will find details on the methodology behind ATHENA. If you find ATHENA useful in your research, please consider citing:
```
@article{10.1093/bioinformatics/btac303,
    author = {Martinelli, Adriano Luca and Rapsomaniki, Maria Anna},
    title = "{ATHENA: analysis of tumor heterogeneity from spatial omics measurements}",
    journal = {Bioinformatics},
    volume = {38},
    number = {11},
    pages = {3151-3153},
    year = {2022},
    doi = {10.1093/bioinformatics/btac303},
}
```

## Installation and Tutorials
In our detailed [Online Documentation](https://ai4scr.github.io/ATHENA) you'll find:
* Installation [instructions](https://ai4scr.github.io/ATHENA/source/installation.html).  
* An overview of ATHENA's [main components](https://ai4scr.github.io/ATHENA/source/overview.html) and [API](https://ai4scr.github.io/ATHENA/api_overview.html)
* An end-to-end [tutorial](https://ai4scr.github.io/ATHENA/source/tutorial.html) using a publicly available Imaging Mass Cytometry dataset.
* A detailed tutorial on ATHENA's `SpatialOmics` data container with examples on how to load your own spatial omics data ([link](https://ai4scr.github.io/ATHENA/source/introduction-spatialOmics.html)).

