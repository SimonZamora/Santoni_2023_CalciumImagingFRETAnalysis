# Santoni_2023_CalciumImagingFRETAnalysis

![alt text](https://github.com/SimonZamora/Santoni_2023_CalciumImagingFRETAnalysis/blob/main/graphicalabstract.png?raw=true)

Welcome!

This repository contains the pipeline analysis used to analyze the FRET-CalciumImaging experiment in Santoni et al. 2024. 
The pipeline consists of three Jupyter notebooks and one helpers.py file containing useful functions. It expects the file to be preprocessed (as discussed in the materials and methods). An example is presented here for the first biological replicates. 
1a_ProcessingFRET: 
-	Compute pixel by pixel the NFRET value and average over the whole detected ROI associated with each nucleus 
-	Exclude saturated nucleus and/or nucleus showing an abnormal signal
-	Compute the distribution over all nuclei in the current biological replicate
-	Attribute to each nucleus a low-mid-high label
-	Save the result in a file

1b_ProcessingCalcium: 
NB: This notebook should be run in a Cascade environment as we have implemented a calcium event detection algorithm using their calcium peak detection (https://github.com/HelmchenLabSoftware/Cascade).
-	Compute ΔF/F0 for each time series
-	Detect calcium event 
-	Exclude cells not showing a robust event detection
-	Save the result in a file
  
2_CombineFRETCA: 
-	Import output files produced by 1a & 1b
-	Attribute to each nucleus the corresponding calcium signal (if it exists)
-	Find double-positive cells (cells validated both for calcium and FRET)
-	Produce summary Excel used for the analysis presented in the paper

Every notebook has been designed to be self-explanatory and well-commented. If you have any questions about the proposed analysis, please feel free to reach out! 


