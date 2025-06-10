# IMC 2025 – 14th Place Solution
Official implementation of our solution for the **Image Matching Challenge 2025**, where we ranked **14th place**.

## Overview
Our pipeline combines:
- **Global feature clustering** using DINOv2 + HDBSCAN
- **Local matching** with ALIKED + LightGlue
- **Scene-wise reconstruction** using PyCOLMAP
- **Dynamic fallbacks** with iterative reconstruction

## Run the code

```bash
python inference.py
```

## References
Kaggle kernel is available [here](https://www.kaggle.com/code/saifdaoud2/pycolmap-imc-2025/notebook?scriptVersionId=243122336)
Our full write-up is available [here](https://www.kaggle.com/competitions/image-matching-challenge-2025/discussion/583977)
