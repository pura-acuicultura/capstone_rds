# capstone_rds
User defined project for HarvardX Data Science Capstone

The project consists of 3x files:

- bllc_capstone_report.pdf
- bllc_capstone_scripts.R
- bllc_capstone_report.Rmd

"bllc_capstone_scripts.R" was created in R version 4.5.0 and RStudio/2025.05.0+496. The file will run without error but please note:

- 10 packages will be installed if not already existing
- To download the dataset directly from Kaggle, a fine-grained personal access token will need to be obtained and present in the working drive as "kaggle.json". If you don't know how to do this, read the "Authentication" section at this link: https://www.kaggle.com/docs/api
- The dataset consists of ~160 MB of images, 2x .csv files and 1x .txt file
- 11 computationally intensive models (4x high predictor n KNN, 4x large matrix PCA-KNN, and 3x CNN) are trained and tested. Scripts took ~ 6 hours to run on a Windows CPU with 32 GB RAM, 10 cores, and 16 logical processors.
- KNN and PCA-KNN model output is saved to the working directory in .rds files
- Epoch-by-epoch CNN model checkpoints are saved in folders within the working directory
- Many of the generated files are needed to knit "bllc_capstone_report.Rmd" should this be desired

"bllc_capstone_report.Rmd" will not re-run all the models created in "bllc_capstone_scripts.R". To run without error, it will need some example data and the ancillary files created in "bllc_capstone_scripts.R". As an alternative to running the scripts, the file set has been uploaded to github. Download and unzip to the working drive. 
