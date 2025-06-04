# capstone_rds
User defined project for HarvardX Data Science Capstone

The project consists of 3x files:

- capstone_report.pdf
- capstone_scripts.R
- capstone_report.Rmd

"capstone_scripts.R" was created in R version 4.5.0 and RStudio/2025.05.0+496. The file will run without error but please note:

- 10 packages will be installed if not already existing
- To download the dataset directly from Kaggle, a fine-grained personal access token will need to be obtained and present in the working drive as "kaggle.json". If you don't know how to do this, read the "Authentication" section at this link: https://www.kaggle.com/docs/api
- The dataset consists of ~160 MB of images, 2x .csv files and 1x .txt file
- 11 computationally intensive models (4x high predictor n KNN, 4x large matrix PCA-KNN, and 3x CNN) are trained and tested. Scripts took ~ 9 hours to run on a Windows CPU with 32 GB RAM, 10 cores, and 16 logical processors.
- KNN, PCA-KNN, and CNN model summaries are saved to a "results" directory as .csv files
- Epoch-by-epoch CNN model checkpoints, logs, and learning rates are also saved to the "results" directory
- Many of the generated files are needed to knit "capstone_report.Rmd" should this be desired

"capstone_report.Rmd" will not re-run all the models created in "capstone_scripts.R". To run without error, however, it will need the "data" and "results" folders generated through "capstone_scripts.R". 
