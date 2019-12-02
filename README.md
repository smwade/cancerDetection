# Cancer Project

## Overview

Cervical cancer is one of the daeliest cancers amound women worldwide. Fortunatly, it is also one of the most preventative.
Using a Pap-smear test, trained technicians can detect cancerous cells early in devopment and provide life saving treatment.
T. In 2018 alone, there were well over 570,000 cases of cervical cancer. Out of these a staggaring 90% were from low to middle income
countries. This highlights a crucial factor for stopping this cancer: cost and availability.

The goal of this project is to help make pap-smear tests and diagnosis available to poor communities. I will be focusing on algorithms
to detect the cancer. A key challange in this is lack of data. Lots of the work is augmenting the existing data and creating synthetic
datasets as well. 

## Datasets

###  SMEAR

The SMEAR dataset is 917 indavidual cells. They are segmented by nucleus and cytoplasm.

<https://mde-lab.aegean.gr/downloads>

### SPIaKMeD

The SIPaKMeD Database consists of 4049 images of isolated cells that have been manually cropped from 966 cluster cell images of Pap smear slides. These images were acquired through a CCD camera adapted to an optical microscope. The cell images are divided into five categories containing normal, abnormal and benign cells.

<http://cs.uoi.gr/~marina/sipakmed.html>

## Install

```
bash setup.py
```

For notebook

```
python -m venv venv
source venv/bin/activate
pip install -r requirments.txt
python -m ipykernel install --user --name=cancer
```

