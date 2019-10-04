Substructured Resolution Enhancement for Healthy Weight Bearing Bones using a Neural Network
=============
***
# File Informations
## 0. Datas and Preprocessing
* Transform_HRtoLR.py
  * Make all subject's HR DV file to LR DV files, LR image files, and LR flag files
## 1. FEA
* FEA_Bone.py
  * The main of this file is to do FEA to DV files. 
* WriteLRTecplot.py
  * Write LR Tecplot files
***
# Data Structure
## 1. LR_set
* [index, y, x] LR patch가 8by8로 들어가 있음. (ratio = 10)
## 2. HR_set
* [index, y, x]: HR patch가 80by80으로 들어가있음
## 3. Dis_set
* [index,  y, x, channel]: displacment가 들어가있음 channel 1은 하중 케이스 1의 UX channel 2는 하중 케이스 1의 UY channel 3는 하중 케이스 2의 UX, 이하 동일
***
# Variables
## 1. patch_n
* The side length of LR patch
## 2. ratio
* Upscaling Ratio
## 3. rp
* The side lenght of HR patch (ratio * patch_n)
## 4. height, width
* The side lengths of HR Image (2080, 1883)
## 5. resolution
* resolution of HR image
