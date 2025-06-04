#00.CAPSTONE_PROJECT_SANTIAGO_1975 #####

# install packages if needed...
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(flextable)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(httr2)) install.packages("httr2", repos = "http://cran.us.r-project.org")
if(!require(jsonlite)) install.packages("jsonlite", repos = "http://cran.us.r-project.org")
if(!require(luz)) install.packages("luz", repos = "http://cran.us.r-project.org")
if(!require(magick)) install.packages("magick", repos = "http://cran.us.r-project.org")
if(!require(mdatools)) install.packages("mdatools", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(torch)) install.packages("torch", repos = "http://cran.us.r-project.org")
if(!require(torchvision)) install.packages("torchvision", repos = "http://cran.us.r-project.org")
if(!require(zip)) install.packages("zip", repos = "http://cran.us.r-project.org")

# load libraries...
library(caret)
library(flextable)
library(httr2) # for kaggle api request...
library(jsonlite) # for kaggle api request...
library(luz)
library(magick)
library(mdatools) # for randomized PCA...
library(tidyverse)
library(torch)
library(torchvision) # for resnet18 model...
library(zip) # for extracting zip file...

#01.DATA_WRANGLING #####
#__01.data_acquisition ####
# data is available from kaggle...
# see: https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification

# make sure your Kaggle Token is saved as "kaggle.json" in your working directory...
# see: https://www.kaggle.com/docs/api

# for help with the script to download kaggle data to R...
# see: https://stackoverflow.com/questions/78666449/download-file-by-using-r

credentials <- read_json("kaggle.json")
request(paste0("https://www.kaggle.com/api/v1/datasets/download/", 
               "marquis03/bean-leaf-lesions-classification")) |> 
  req_auth_basic(credentials$username, credentials$key) |>
  req_perform(path = paste(getwd(), "bllc_data.zip", sep = "/"))

unzip("bllc_data.zip", exdir = "data")

file.remove("bllc_data.zip")

# clean up after the data acquisition...
rm(credentials)

#__02.image_indexing #####
#____01.training data set ####
# NB: if your normal working directory is a Google drive, performance might be...
# improved by downloading data to the machine itself... 
# path_r <- "C:/Users/james/.torch-datasets/bllc_data"

path_r <- paste(getwd(), "data", sep = "/")
file_r <- "train.csv"

df_train_index <- read_csv(paste(path_r, file_r, sep = "/"),
                           show_col_types = FALSE) %>%
  rename(path = `image:FILE`,
         label = category) %>%
  mutate(path = paste(path_r, path, sep = "/")) %>%
  mutate(label = factor(label + 1, levels = c(1:3))) %>%
  mutate(class = case_when(label == 1 ~ "healthy",
                           label == 2 ~ "angular_leaf_spot",
                           label == 3 ~ "bean_rust"))

nrow(df_train_index) # 1034 training images...
head(df_train_index)

# visualization for .rmd...
df_train_index %>%
  mutate(path = paste0(str_sub(path, 0, 20), "..........", str_sub(path, -40))) %>%
  group_by(class) %>%
  slice(1:2) %>%
  arrange(label) %>%
  flextable(.) %>%
  fontsize(., size = 10, part = 'all') %>%
  align(., align = 'center', part = 'header') %>%
  align(., align = 'center', j = 2) %>%
  width(., width = 4.5, j = 1) %>%
  width(., width = 0.5, j = 2) %>%
  width(., width = 1, j = 3)

#____02.validation data set ####

file_r <- "val.csv"

df_val_index <- read_csv(paste(path_r, file_r, sep = "/"),
                         show_col_types = FALSE) %>%
  rename(path = `image:FILE`,
         label = category) %>%
  mutate(path = paste(path_r, path, sep = "/")) %>%
  mutate(label = factor(label + 1, levels = c(1:3))) %>%
  mutate(class = case_when(label == 1 ~ "healthy",
                           label == 2 ~ "angular_leaf_spot",
                           label == 3 ~ "bean_rust"))

nrow(df_val_index) # 133 validation images...
head(df_val_index)

#__03.exploratory_data_analysis ####
#____01.image_explorer ####
# get the path & class of 3 images from each class...
eda_img <- df_train_index %>%
  group_by(class) %>%
  slice(1:3) %>%
  select(path, class)

# make an empty image list...
img_list <- image_join()

# loop through the previously defined images... 
for(i in 1:length(eda_img$path)){
  img_list <- image_read(eda_img$path[i]) %>%
    image_scale(., "128x128") %>%
    image_annotate(., eda_img$class[i],  location = "+5+5", font = 'Arial', size = 14, color='white') %>%
    image_join(., img_list)
}

# print the images and labels as a montage...
image_montage(img_list,
              geometry = 'x180+10+10',
              tile = '3x3')

# clean up after the montage...
rm(eda_img)
rm(img_list)

#____02.class_balance ####

df_train_index %>%
  group_by(class) %>%
  summarize(n = n()) %>%
  ggplot() +
  geom_bar(aes(x = class, y = n), stat = "identity", width = 0.8,
           fill = "skyblue1", alpha = 0.8, color = "grey30") +
  geom_text(aes(x = class, y = (n+20), label = paste0("n = ", n))) +
  geom_hline(yintercept = 0) +
  scale_y_continuous(breaks = seq(0, 400, by = 50),
                     minor_breaks = seq(0, 400, by = 10),
                     limits = c(0, 400)) +
  xlab("") +
  ylab("number of images") +
  theme_bw()

#02.KNN #####
#__01.gray_scale ####
#____01.image_vectorization ####
#______01.train_gray ####

train_knn_gray <- NULL

for(i in 1:nrow(df_train_index)){
  tensor_local <- image_read(df_train_index$path[i]) %>%
    image_scale(., "128x128") %>%
    image_convert(colorspace = "Gray") %>%
    transform_to_tensor()
  
  train_flat <- as.numeric(tensor_local[1,,]) # in gray-scale we only want the 1st layer...
  train_knn_gray <- rbind(train_knn_gray, train_flat)
}

# wrangle gray channel for knn... 
rownames(train_knn_gray) <- paste("img",
                              str_pad(1:dim(train_knn_gray)[1], width = 4, pad="0"),
                              sep = "_")
colnames(train_knn_gray) <- paste("x",
                              str_pad(1:dim(train_knn_gray)[2], width = 4, pad="0"),
                              sep = "_") 

df_train_knn_gray <- data.frame(y = df_train_index$label, train_knn_gray) %>%
  mutate(y = factor(y)) %>%
  remove_rownames(.)

df_train_knn_gray %>%
  select(1:5) %>%
  head(.)  

#______02.val_gray ####
val_knn_gray <- NULL

for(i in 1:nrow(df_val_index)){
  tensor_local <- image_read(df_val_index$path[i]) %>%
    image_scale(., "128x128") %>%
    image_convert(colorspace = "Gray") %>%
    transform_to_tensor()
  
  val_flat <- as.numeric(tensor_local[1,,]) # in gray-scale we only want the 1st layer...
  val_knn_gray <- rbind(val_knn_gray, val_flat)
}

# wrangle gray channel for knn... 
rownames(val_knn_gray) <- paste("img",
                            str_pad(1:dim(val_knn_gray)[1], width = 4, pad="0"),
                            sep = "_")
colnames(val_knn_gray) <- paste("x",
                            str_pad(1:dim(val_knn_gray)[2], width = 4, pad="0"),
                            sep = "_") 
df_val_knn_gray <- data.frame(y = df_val_index$label, val_knn_gray) %>%
  mutate(y = factor(y)) %>%
  remove_rownames(.)

df_val_knn_gray %>%
  select(1:5) %>%
  head(.)

#____02.cross_validation ####
set.seed(5446)

control <- trainControl(method = "cv",
                        number = 5,
                        p = 0.8)

train_knn <- train(y ~ .,
                   data = df_train_knn_gray,
                   method = "knn", 
                   tuneGrid = data.frame(k = seq(10, 100, by = 10)),
                   trControl = control)

# view accuracy as a function of k...
ggplot(train_knn, highlight = TRUE) +
  scale_x_continuous(breaks = seq(10, 100, by = 10),
                     minor_breaks = seq(0, 100, by = 5)) +
  theme_bw()

k_knn_gray <- train_knn$bestTune # 90... 

#____03.training_and_validation ####
set.seed(5446)

train_knn <- knn3(y ~ .,
                  data = df_train_knn_gray,
                  k = k_knn_gray)

y_hat_knn_gray <- predict(train_knn,
                          df_val_knn_gray,
                          type = "class")

cm_knn_gray <- confusionMatrix(y_hat_knn_gray,
                               factor(df_val_knn_gray$y)) 
cm_knn_gray # accuracy = 0.5038...

# clean_up...
rm(train_knn)
rm(train_flat)
rm(val_flat)
rm(control)
rm(tensor_local)
rm(df_train_knn_gray)
rm(df_val_knn_gray)

#__02.red_channel ####
#____01.image_vectorization ####
#______01.train_red ####
train_knn_red <- NULL

for(i in 1:nrow(df_train_index)){
  tensor_local <- image_read(df_train_index$path[i]) %>%
    image_scale(., "128x128") %>%
    transform_to_tensor()
  
  train_flat <- as.numeric(tensor_local[1,,]) # red is the 1st tensor layer...
  train_knn_red <- rbind(train_knn_red, train_flat)
}

# wrangle red channel for knn... 
rownames(train_knn_red) <- paste("img",
                                  str_pad(1:dim(train_knn_red)[1], width = 4, pad="0"),
                                  sep = "_")
colnames(train_knn_red) <- paste("x",
                                  str_pad(1:dim(train_knn_red)[2], width = 4, pad="0"),
                                  sep = "_") 

df_train_knn_red <- data.frame(y = df_train_index$label, train_knn_red) %>%
  mutate(y = factor(y)) %>%
  remove_rownames(.)

df_train_knn_red %>%
  select(1:5) %>%
  head(.)  

#______02.val_red ####
val_knn_red <- NULL

for(i in 1:nrow(df_val_index)){
  tensor_local <- image_read(df_val_index$path[i]) %>%
    image_scale(., "128x128") %>%
    transform_to_tensor()
  
  val_flat <- as.numeric(tensor_local[1,,]) # red is the 1st tensor layer...
  val_knn_red <- rbind(val_knn_red, val_flat)
}

# wrangle red channel for knn... 
rownames(val_knn_red) <- paste("img",
                                str_pad(1:dim(val_knn_red)[1], width = 4, pad="0"),
                                sep = "_")
colnames(val_knn_red) <- paste("x",
                                str_pad(1:dim(val_knn_red)[2], width = 4, pad="0"),
                                sep = "_") 

df_val_knn_red <- data.frame(y = df_val_index$label, val_knn_red) %>%
  mutate(y = factor(y)) %>%
  remove_rownames(.)

df_val_knn_red %>%
  select(1:5) %>%
  head(.)

#____02.cross_validation ####
set.seed(5446)

control <- trainControl(method = "cv",
                        number = 5,
                        p = 0.8)

train_knn <- train(y ~ .,
                   data = df_train_knn_red,
                   method = "knn", 
                   tuneGrid = data.frame(k = seq(10, 100, by = 10)),
                   trControl = control)

# view accuracy as a function of k...
ggplot(train_knn, highlight = TRUE) +
  scale_x_continuous(breaks = seq(0, 100, by = 10),
                     minor_breaks = seq(0, 100, by = 5)) +
  theme_bw()

k_knn_red <- train_knn$bestTune # 100... 

#____03.training_and_validation ####
set.seed(5446)

train_knn <- knn3(y ~ .,
                  data = df_train_knn_red,
                  k = k_knn_red)

y_hat_knn_red <- predict(train_knn,
                         df_val_knn_red,
                         type = "class")

cm_knn_red <- confusionMatrix(y_hat_knn_red,
                              factor(df_val_knn_red$y)) 
cm_knn_red # accuracy = 0.4135...

# clean_up...
rm(train_knn)
rm(train_flat)
rm(val_flat)
rm(control)
rm(tensor_local)
rm(df_train_knn_red)
rm(df_val_knn_red)

#__03.green_channel ####
#____01.image_vectorization ####
#______01.train_green ####
train_knn_green <- NULL

for(i in 1:nrow(df_train_index)){
  tensor_local <- image_read(df_train_index$path[i]) %>%
    image_scale(., "128x128") %>%
    transform_to_tensor()
  
  train_flat <- as.numeric(tensor_local[2,,]) # green is the 2nd tensor layer...
  train_knn_green <- rbind(train_knn_green, train_flat)
}

# wrangle green channel for knn... 
rownames(train_knn_green) <- paste("img",
                                  str_pad(1:dim(train_knn_green)[1], width = 4, pad="0"),
                                  sep = "_")
colnames(train_knn_green) <- paste("x",
                                  str_pad(1:dim(train_knn_green)[2], width = 4, pad="0"),
                                  sep = "_") 

df_train_knn_green <- data.frame(y = df_train_index$label, train_knn_green) %>%
  mutate(y = factor(y)) %>%
  remove_rownames(.)

df_train_knn_green %>%
  select(1:5) %>%
  head(.)  

#______02.val_green ####
val_knn_green <- NULL

for(i in 1:nrow(df_val_index)){
  tensor_local <- image_read(df_val_index$path[i]) %>%
    image_scale(., "128x128") %>%
    transform_to_tensor()
  
  val_flat <- as.numeric(tensor_local[2,,]) # green is the 2nd tensor layer...
  val_knn_green <- rbind(val_knn_green, val_flat)
}

# wrangle green channel for knn... 
rownames(val_knn_green) <- paste("img",
                                str_pad(1:dim(val_knn_green)[1], width = 4, pad="0"),
                                sep = "_")
colnames(val_knn_green) <- paste("x",
                                str_pad(1:dim(val_knn_green)[2], width = 4, pad="0"),
                                sep = "_") 

df_val_knn_green <- data.frame(y = df_val_index$label, val_knn_green) %>%
  mutate(y = factor(y)) %>%
  remove_rownames(.)

df_val_knn_green %>%
  select(1:5) %>%
  head(.)

#____02.cross_validation ####
set.seed(5446)

control <- trainControl(method = "cv",
                        number = 5,
                        p = 0.8)

train_knn <- train(y ~ .,
                   data = df_train_knn_green,
                   method = "knn", 
                   tuneGrid = data.frame(k = seq(10, 100, by = 10)),
                   trControl = control)

# view accuracy as a function of k...
ggplot(train_knn, highlight = TRUE) +
  scale_x_continuous(breaks = seq(10, 100, by = 10),
                     minor_breaks = seq(0, 100, by = 5)) +
  theme_bw()

k_knn_green <- train_knn$bestTune # 80... 

#____03.training_and_validation ####
set.seed(5446)

train_knn <- knn3(y ~ .,
                  data = df_train_knn_green,
                  k = k_knn_green)

y_hat_knn_green <- predict(train_knn,
                           df_val_knn_green,
                           type = "class")

cm_knn_green <- confusionMatrix(y_hat_knn_green,
                                factor(df_val_knn_green$y)) 
cm_knn_green # accuracy = 0.5038...

# clean_up...
rm(train_knn)
rm(train_flat)
rm(val_flat)
rm(control)
rm(tensor_local)
rm(df_train_knn_green)
rm(df_val_knn_green)

#__04.blue_channel ####
#____01.image_vectorization ####
#______01.train_blue ####
train_knn_blue <- NULL

for(i in 1:nrow(df_train_index)){
  tensor_local <- image_read(df_train_index$path[i]) %>%
    image_scale(., "128x128") %>%
    transform_to_tensor()
  
  train_flat <- as.numeric(tensor_local[3,,]) # blue is the 3rd tensor layer...
  train_knn_blue <- rbind(train_knn_blue, train_flat)
}

# wrangle blue channel for knn... 
rownames(train_knn_blue) <- paste("img",
                                   str_pad(1:dim(train_knn_blue)[1], width = 4, pad="0"),
                                   sep = "_")
colnames(train_knn_blue) <- paste("x",
                                   str_pad(1:dim(train_knn_blue)[2], width = 4, pad="0"),
                                   sep = "_") 

df_train_knn_blue <- data.frame(y = df_train_index$label, train_knn_blue) %>%
  mutate(y = factor(y)) %>%
  remove_rownames(.)

df_train_knn_blue %>%
  select(1:5) %>%
  head(.)  

#______02.val_blue ####
val_knn_blue <- NULL

for(i in 1:nrow(df_val_index)){
  tensor_local <- image_read(df_val_index$path[i]) %>%
    image_scale(., "128x128") %>%
    transform_to_tensor()
  
  val_flat <- as.numeric(tensor_local[3,,]) # blue is the 3rd tensor layer...
  val_knn_blue <- rbind(val_knn_blue, val_flat)
}

# wrangle blue channel for knn... 
rownames(val_knn_blue) <- paste("img",
                                 str_pad(1:dim(val_knn_blue)[1], width = 4, pad="0"),
                                 sep = "_")
colnames(val_knn_blue) <- paste("x",
                                 str_pad(1:dim(val_knn_blue)[2], width = 4, pad="0"),
                                 sep = "_") 

df_val_knn_blue <- data.frame(y = df_val_index$label, val_knn_blue) %>%
  mutate(y = factor(y)) %>%
  remove_rownames(.)

df_val_knn_blue %>%
  select(1:5) %>%
  head(.)

#____02.cross_validation ####
set.seed(5446)

control <- trainControl(method = "cv",
                        number = 5,
                        p = 0.8)

train_knn <- train(y ~ .,
                   data = df_train_knn_blue,
                   method = "knn", 
                   tuneGrid = data.frame(k = seq(10, 100, by = 10)),
                   trControl = control)

# view accuracy as a function of k...
ggplot(train_knn, highlight = TRUE) +
  scale_x_continuous(breaks = seq(10, 100, by = 10),
                     minor_breaks = seq(0, 100, by = 5)) +
  theme_bw()

k_knn_blue <- train_knn$bestTune # 100... 

#____03.training_and_validation ####
set.seed(5446)

train_knn <- knn3(y ~ .,
                  data = df_train_knn_blue,
                  k = k_knn_blue)

y_hat_knn_blue <- predict(train_knn,
                          df_val_knn_blue,
                          type = "class")

cm_knn_blue <- confusionMatrix(y_hat_knn_blue,
                                factor(df_val_knn_blue$y)) 
cm_knn_blue # accuracy = 0.4436...

# clean_up...
rm(train_knn)
rm(train_flat)
rm(val_flat)
rm(control)
rm(tensor_local)
rm(df_train_knn_blue)
rm(df_val_knn_blue)

#__05.knn_ensemble ####
y_hat_knn_ensemble <- df_val_index %>%
  select(-path) %>%
  mutate(y_gray = y_hat_knn_gray,
         y_red = y_hat_knn_red,
         y_green = y_hat_knn_green,
         y_blue = y_hat_knn_blue) %>%
  mutate(observation = row_number()) %>%
  gather(key = "key", value = "value", -c(label, class, observation)) %>%
  group_by(observation, label, value) %>%
  summarize(n = n()) %>%
  group_by(observation) %>%
  arrange(desc(n)) %>%
  group_by(observation) %>%
  slice(1) %>%
  arrange(observation) %>%
  pull(value)

cm_knn_ensemble <- confusionMatrix(factor(y_hat_knn_ensemble),
                                   factor(df_val_index$label)) 
cm_knn_ensemble # accuracy = 0.5263...

#__06.knn_summary ####

# get key results into a summary table...
knn_summary <- data.frame(
  id = 1:5,
  model = c("knn_gray", "knn_red", "knn_green",
            "knn_blue", "knn_ensemble"),
  k = c(k_knn_gray$k, k_knn_red$k, k_knn_green$k, k_knn_blue$k, NA),
  accuracy = c(cm_knn_gray$overall[1], cm_knn_red$overall[1],
               cm_knn_green$overall[1], cm_knn_blue$overall[1],
               cm_knn_ensemble$overall[1])) %>%
  mutate(k = round(k, 3),
         accuracy = round(accuracy, 3))

# write the summary to file for .Rmd...
dir.create(path = paste(getwd(), "results", sep = "/"))
path_w <- paste(getwd(), "results", sep = "/")
file_w <- "knn_summary.csv"
write_csv(knn_summary, paste(path_w, file_w, sep = "/"))

# flextable for reporting...
knn_summary %>% 
  flextable(.) %>%
  fontsize(., size = 10, part = 'all') %>%
  align(., align = 'center', part = 'header') %>%
  align(., align = 'center', j = c(1, 3, 4)) %>%
  width(., width = 1.25, j = 1:4)

# cleanup after KNN...
rm(k_knn_gray, k_knn_red, k_knn_green, k_knn_blue)
rm(y_hat_knn_gray, y_hat_knn_red, y_hat_knn_green, y_hat_knn_blue,
   y_hat_knn_ensemble)

#03.PCA_KNN #####
#__01.gray_scale ####
#____01.compute_principle_components_for_train_gray ####

# check the matrix to be used...
train_knn_gray[1:10,1:10]

# compute the first 10 principle components...
set.seed(5446)
m2_train = pca(train_knn_gray, ncomp = 10, rand = c(5, 1))
summary(m2_train) # 48.57 cumulative explained variance...

# save the cumulative explained variance for reporting...
var_pca_gray <- as.numeric(m2_train$res$cal$cumexpvar[10])

dim(m2_train$res$cal$scores) # 1034 rows x 10 columns...
m2_train$res$cal$scores[1:10,] # take a look at a few scores...

# wrangle the principle components into a data frame for knn... 
train_pca_gray <- m2_train$res$cal$scores

rownames(train_pca_gray) <- paste("img",
                                  str_pad(1:dim(train_pca_gray)[1], width = 4, pad="0"),
                                  sep = "_")
colnames(train_pca_gray) <- paste("x",
                                  str_pad(1:dim(train_pca_gray)[2], width = 4, pad="0"),
                                  sep = "_")

df_train_pca_gray <- data.frame(y = df_train_index$label,
                                train_pca_gray) %>%
  mutate(y = factor(y)) %>%
  remove_rownames(.)

# take a quick look at the data frame...
df_train_pca_gray %>%
  head(.)

# plot the first 2x principle components...
plot <- df_train_pca_gray %>%
  mutate(class = df_train_index$class) %>%
  ggplot() +
  geom_point(aes(x = x_0001, y = x_0002, color = class, shape = class)) +
  theme_bw()
plot

# save the plot for .Rmd...
path_i <- paste(getwd(), "results", sep = "/")
file_i <- "plot_pca_gray.png"

png(filename = paste(path_i, file_i, sep = '/'), 
    units = "in", 
    width = 6.5, 
    height = 4, 
    pointsize = 14, 
    res = 256)
plot
dev.off()

#____02.predict_principle_components_for_val_gray ####

# check the matrix to be used...
val_knn_gray[1:10,1:10]

# predict the first 10 principle components from train pca...
m2_val <- predict(m2_train, val_knn_gray)
summary(m2_val)

dim(m2_val$scores) # 133 rows x 10 columns...
m2_val$scores[1:10,] # take a look at a few scores...

# wrangle the principle components into a data frame for knn...
val_pca_gray <- m2_val$scores

rownames(val_pca_gray) <- paste("img",
                                str_pad(1:dim(val_pca_gray)[1], width = 4, pad="0"),
                                sep = "_")
colnames(val_pca_gray) <- paste("x",
                                str_pad(1:dim(val_pca_gray)[2], width = 4, pad="0"),
                                sep = "_")

df_val_pca_gray <- data.frame(y = df_val_index$label,
                              val_pca_gray) %>%
  mutate(y = factor(y)) %>%
  remove_rownames(.)

# take a quick look at the data frame...
df_val_pca_gray %>%
  head(.)

#____03.model_and_evaluate ####
set.seed(5446)

control <- trainControl(method = "cv",
                        number = 5,
                        p = 0.8)

train_knn <- train(y ~ .,
                   data = df_train_pca_gray,
                   method = "knn", 
                   tuneGrid = data.frame(k = seq(10, 100, by = 10)),
                   trControl = control)

ggplot(train_knn, highlight = TRUE) +
  scale_x_continuous(breaks = seq(0, 100, by = 10),
                     minor_breaks = seq(0, 100, by = 5)) +
  theme_bw()

k_pca_gray <- train_knn$bestTune # 30... 

# re-train with best k & predict for validation set...
set.seed(5446)

train_knn <- knn3(y ~ .,
                  data = df_train_pca_gray,
                  k = k_pca_gray)

y_hat_pca_gray <- predict(train_knn,
                              df_val_pca_gray,
                              type = "class")

cm_pca_gray <- confusionMatrix(y_hat_pca_gray,
                               factor(df_val_pca_gray$y))
cm_pca_gray # accuracy = 0.5188...

# clean_up...
rm(m2_train)
rm(m2_val)
rm(control)
rm(train_knn)
rm(train_knn_gray)
rm(val_knn_gray)
rm(train_pca_gray)
rm(val_pca_gray)
rm(df_train_pca_gray)
rm(df_val_pca_gray)

#__02.red_channel ####
#____01.compute_principle_components_for_train_red ####

# check the matrix to be used...
train_knn_red[1:10,1:10]

# compute the first 10 principle components...
set.seed(5446)
m2_train = pca(train_knn_red, ncomp = 10, rand = c(5, 1))
summary(m2_train) # 53.18 cumulative variance explained...

# save the cumulative explained variance for reporting...
var_pca_red <- as.numeric(m2_train$res$cal$cumexpvar[10])

dim(m2_train$res$cal$scores) # 1034 rows x 10 columns...
m2_train$res$cal$scores[1:10,] # take a look at a few scores...

# wrangle the principle components into a data frame for knn... 
train_pca_red <- m2_train$res$cal$scores

rownames(train_pca_red) <- paste("img",
                                 str_pad(1:dim(train_pca_red)[1], width = 4, pad="0"),
                                 sep = "_")
colnames(train_pca_red) <- paste("x",
                                 str_pad(1:dim(train_pca_red)[2], width = 4, pad="0"),
                                 sep = "_")

df_train_pca_red <- data.frame(y = df_train_index$label,
                               train_pca_red) %>%
  mutate(y = factor(y)) %>%
  remove_rownames(.)

# take a quick look at the data frame...
df_train_pca_red %>%
  head(.)

# plot the first 2x principle components...
plot <- df_train_pca_red %>%
  mutate(class = df_train_index$class) %>%
  ggplot() +
  geom_point(aes(x = x_0001, y = x_0002, color = class, shape = class)) +
  theme_bw()
plot

# save the plot for .Rmd...
path_i <- paste(getwd(), "results", sep = "/")
file_i <- "plot_pca_red.png"

png(filename = paste(path_i, file_i, sep = '/'), 
    units = "in", 
    width = 6.5, 
    height = 4, 
    pointsize = 14, 
    res = 256)
plot
dev.off()

#____02.predict_principle_components_for_val_red ####

# check the matrix to be used...
val_knn_red[1:10,1:10]

# predict the first 10 principle components from train pca...
m2_val <- predict(m2_train, val_knn_red)
summary(m2_val)

dim(m2_val$scores) # 133 rows x 10 columns...
m2_val$scores[1:10,] # take a look at a few scores...

# wrangle the principle components into a data frame for knn...
val_pca_red <- m2_val$scores

rownames(val_pca_red) <- paste("img",
                               str_pad(1:dim(val_pca_red)[1], width = 4, pad="0"),
                               sep = "_")
colnames(val_pca_red) <- paste("x",
                               str_pad(1:dim(val_pca_red)[2], width = 4, pad="0"),
                               sep = "_")

df_val_pca_red <- data.frame(y = df_val_index$label,
                             val_pca_red) %>%
  mutate(y = factor(y)) %>%
  remove_rownames(.)

# take a quick look at the data frame...
df_val_pca_red %>%
  head(.)

#____03.model_and_evaluate ####
set.seed(5446)

control <- trainControl(method = "cv",
                        number = 5,
                        p = 0.8)

train_knn <- train(y ~ .,
                   data = df_train_pca_red,
                   method = "knn", 
                   tuneGrid = data.frame(k = seq(10, 100, by = 10)),
                   trControl = control)

ggplot(train_knn, highlight = TRUE) +
  scale_x_continuous(breaks = seq(0, 100, by = 10),
                     minor_breaks = seq(0, 100, by = 5)) +
  theme_bw()

k_pca_red <- train_knn$bestTune # 80... 

# re-train with best k & predict for validation set...
set.seed(5446)

train_knn <- knn3(y ~ .,
                  data = df_train_pca_red,
                  k = k_pca_red)

y_hat_pca_red <- predict(train_knn,
                         df_val_pca_red,
                         type = "class")

cm_pca_red <- confusionMatrix(y_hat_pca_red,
                               factor(df_val_pca_red$y))
cm_pca_red # accuracy = 0.4887...

# clean_up...
rm(m2_train)
rm(m2_val)
rm(control)
rm(train_knn)
rm(train_knn_red)
rm(val_knn_red)
rm(train_pca_red)
rm(val_pca_red)
rm(df_train_pca_red)
rm(df_val_pca_red)

#__03.green_channel ####
#____01.compute_principle_components_for_train_green ####

# check the matrix to be used...
train_knn_green[1:10,1:10]

# compute the first 10 principle components...
set.seed(5446)
m2_train = pca(train_knn_green, ncomp = 10, rand = c(5, 1))
summary(m2_train) # 49.81% cumulative variance explained...

dim(m2_train$res$cal$scores) # 1034 rows x 10 columns...
m2_train$res$cal$scores[1:10,] # take a look at a few scores...

# save the cumulative explained variance for reporting...
var_pca_green <- as.numeric(m2_train$res$cal$cumexpvar[10])

# wrangle the principle components into a data frame for knn... 
train_pca_green <- m2_train$res$cal$scores

rownames(train_pca_green) <- paste("img",
                                   str_pad(1:dim(train_pca_green)[1], width = 4, pad="0"),
                                   sep = "_")
colnames(train_pca_green) <- paste("x",
                                   str_pad(1:dim(train_pca_green)[2], width = 4, pad="0"),
                                   sep = "_")

df_train_pca_green <- data.frame(y = df_train_index$label,
                                 train_pca_green) %>%
  mutate(y = factor(y)) %>%
  remove_rownames(.)

# take a quick look at the data frame...
df_train_pca_green %>%
  head(.)

# plot the first 2x principle components...
plot <- df_train_pca_green %>%
  mutate(class = df_train_index$class) %>%
  ggplot() +
  geom_point(aes(x = x_0001, y = x_0002, color = class, shape = class)) +
  theme_bw()
plot

# save the plot for .Rmd...
path_i <- paste(getwd(), "results", sep = "/")
file_i <- "plot_pca_green.png"

png(filename = paste(path_i, file_i, sep = '/'), 
    units = "in", 
    width = 6.5, 
    height = 4, 
    pointsize = 14, 
    res = 256)
plot
dev.off()

#____02.predict_principle_components_for_val_green ####

# check the matrix to be used...
val_knn_green[1:10,1:10]

# predict the first 10 principle components from train pca...
m2_val <- predict(m2_train, val_knn_green)
summary(m2_val)

dim(m2_val$scores) # 133 rows x 10 columns...
m2_val$scores[1:10,] # take a look at a few scores...

# wrangle the principle components into a data frame for knn...
val_pca_green <- m2_val$scores

rownames(val_pca_green) <- paste("img",
                                 str_pad(1:dim(val_pca_green)[1], width = 4, pad="0"),
                                 sep = "_")
colnames(val_pca_green) <- paste("x",
                                 str_pad(1:dim(val_pca_green)[2], width = 4, pad="0"),
                                 sep = "_")

df_val_pca_green <- data.frame(y = df_val_index$label,
                               val_pca_green) %>%
  mutate(y = factor(y)) %>%
  remove_rownames(.)

# take a quick look at the data frame...
df_val_pca_green %>%
  head(.)

#____03.models_and_evaluate ####
set.seed(5446)

control <- trainControl(method = "cv",
                        number = 5,
                        p = 0.8)

train_knn <- train(y ~ .,
                   data = df_train_pca_green,
                   method = "knn", 
                   tuneGrid = data.frame(k = seq(10, 100, by = 10)),
                   trControl = control)

ggplot(train_knn, highlight = TRUE) +
  scale_x_continuous(breaks = seq(0, 100, by = 10),
                     minor_breaks = seq(0, 100, by = 5)) +
  theme_bw()

k_pca_green <- train_knn$bestTune # 20... 

# re-train with best k & predict for validation set...
set.seed(5446)

train_knn <- knn3(y ~ .,
                  data = df_train_pca_green,
                  k = k_pca_green)

y_hat_pca_green <- predict(train_knn,
                           df_val_pca_green,
                           type = "class")

cm_pca_green <- confusionMatrix(y_hat_pca_green,
                                factor(df_val_pca_green$y))
cm_pca_green # accuracy = 0.4812...

# clean_up...
rm(m2_train)
rm(m2_val)
rm(control)
rm(train_knn)
rm(train_knn_green)
rm(val_knn_green)
rm(train_pca_green)
rm(val_pca_green)
rm(df_train_pca_green)
rm(df_val_pca_green)

#__04.blue_channel ####
#____01.compute_principle_components_for_train_blue ####

# check the matrix to be used...
train_knn_blue[1:10,1:10]

# compute the first 10 principle components...
set.seed(5446)
m2_train = pca(train_knn_blue, ncomp = 10, rand = c(5, 1))
summary(m2_train) # 54.80 % cumulative variance explained...

# save the cumulative explained variance for reporting...
var_pca_blue <- as.numeric(m2_train$res$cal$cumexpvar[10])

dim(m2_train$res$cal$scores) # 1034 rows x 10 columns...
m2_train$res$cal$scores[1:10,] # take a look at a few scores...

# wrangle the principle components into a data frame for knn... 
train_pca_blue <- m2_train$res$cal$scores

rownames(train_pca_blue) <- paste("img",
                                  str_pad(1:dim(train_pca_blue)[1], width = 4, pad="0"),
                                  sep = "_")
colnames(train_pca_blue) <- paste("x",
                                  str_pad(1:dim(train_pca_blue)[2], width = 4, pad="0"),
                                  sep = "_")

df_train_pca_blue <- data.frame(y = df_train_index$label,
                                train_pca_blue) %>%
  mutate(y = factor(y)) %>%
  remove_rownames(.)

# take a quick look at the data frame...
df_train_pca_blue %>%
  head(.)

# plot the first 2x principle components...
plot <- df_train_pca_blue %>%
  mutate(class = df_train_index$class) %>%
  ggplot() +
  geom_point(aes(x = x_0001, y = x_0002, color = class, shape = class)) +
  theme_bw()
plot

# save the plot for .Rmd...
path_i <- paste(getwd(), "results", sep = "/")
file_i <- "plot_pca_blue.png"

png(filename = paste(path_i, file_i, sep = '/'), 
    units = "in", 
    width = 6.5, 
    height = 4, 
    pointsize = 14, 
    res = 256)
plot
dev.off()

#____02.predict_principle_components_for_val_blue ####

# check the matrix to be used...
val_knn_blue[1:10,1:10]

# predict the first 10 principle components from train pca...
m2_val <- predict(m2_train, val_knn_blue)
summary(m2_val)

dim(m2_val$scores) # 133 rows x 10 columns...
m2_val$scores[1:10,] # take a look at a few scores...

# wrangle the principle components into a data frame for knn...
val_pca_blue <- m2_val$scores

rownames(val_pca_blue) <- paste("img",
                                str_pad(1:dim(val_pca_blue)[1], width = 4, pad="0"),
                                sep = "_")
colnames(val_pca_blue) <- paste("x",
                                str_pad(1:dim(val_pca_blue)[2], width = 4, pad="0"),
                                sep = "_")

df_val_pca_blue <- data.frame(y = df_val_index$label,
                              val_pca_blue) %>%
  mutate(y = factor(y)) %>%
  remove_rownames(.)

# take a quick look at the data frame...
df_val_pca_blue %>%
  head(.)

#____03.models_and_evaluate ####
set.seed(5446)

control <- trainControl(method = "cv",
                        number = 5,
                        p = 0.8)

train_knn <- train(y ~ .,
                   data = df_train_pca_blue,
                   method = "knn", 
                   tuneGrid = data.frame(k = seq(10, 100, by = 10)),
                   trControl = control)

ggplot(train_knn, highlight = TRUE) +
  scale_x_continuous(breaks = seq(0, 100, by = 10),
                     minor_breaks = seq(0, 100, by = 5)) +
  theme_bw()

k_pca_blue <- train_knn$bestTune # 60... 

# re-train with best k & predict for validation set...
set.seed(5446)

train_knn <- knn3(y ~ .,
                  data = df_train_pca_blue,
                  k = k_pca_blue)

y_hat_pca_blue <- predict(train_knn,
                              df_val_pca_blue,
                              type = "class")

cm_pca_blue <- confusionMatrix(y_hat_pca_blue,
                               factor(df_val_pca_blue$y))
cm_pca_blue # accuracy = 0.6165...

# clean_up...
rm(m2_train)
rm(m2_val)
rm(control)
rm(train_knn)
rm(train_knn_blue)
rm(val_knn_blue)
rm(train_pca_blue)
rm(val_pca_blue)
rm(df_train_pca_blue)
rm(df_val_pca_blue)

#__05.ensemble ####
y_hat_pca_ensemble <- df_val_index %>%
  select(-path) %>%
  mutate(y_gray = y_hat_pca_gray,
         y_red = y_hat_pca_red,
         y_green = y_hat_pca_green,
         y_blue = y_hat_pca_blue) %>%
  mutate(observation = row_number()) %>%
  gather(key = "key", value = "value", -c(label, class, observation)) %>%
  group_by(observation, label, value) %>%
  summarize(n = n()) %>%
  group_by(observation) %>%
  arrange(desc(n)) %>%
  group_by(observation) %>%
  slice(1) %>%
  arrange(observation) %>%
  pull(value)

cm_pca_ensemble <- confusionMatrix(factor(y_hat_pca_ensemble),
                                   factor(df_val_index$label)) 
cm_pca_ensemble # accuracy = 0.5865...

#__06.pca_summary ####

# get key results into a summary table...
pca_summary <- data.frame(
  id = 6:10,
  model = c("pca_gray", "pca_red", "pca_green",
            "pca_blue", "pca_ensemble"),
  per_var = c(var_pca_gray, var_pca_red, var_pca_green, var_pca_blue, NA),
  k = c(k_pca_gray$k, k_pca_red$k, k_pca_green$k, k_pca_blue$k, NA),
  accuracy = c(cm_pca_gray$overall[1], cm_pca_red$overall[1],
               cm_pca_green$overall[1], cm_pca_blue$overall[1],
               cm_pca_ensemble$overall[1])) %>%
  mutate(per_var = round(per_var, 3),
         k = round(k, 3),
         accuracy = round(accuracy, 3))

# write the summary to file for .Rmd...
dir.create(path = paste(getwd(), "results", sep = "/"))
path_w <- paste(getwd(), "results", sep = "/")
file_w <- "pca_summary.csv"
write_csv(pca_summary, paste(path_w, file_w, sep = "/"))

# flextable for reporting...
pca_summary %>% 
  flextable(.) %>%
  fontsize(., size = 10, part = 'all') %>%
  align(., align = 'center', part = 'header') %>%
  align(., align = 'center', j = c(1, 3:5)) %>%
  width(., width = 1, j = 1:5)

# cleanup after KNN...
rm(k_pca_gray, k_pca_red, k_pca_green, k_pca_blue)
rm(y_hat_pca_gray, y_hat_pca_red, y_hat_pca_green, y_hat_pca_blue,
   y_hat_pca_ensemble)

#04.CNN #####
#__01.cnn_gray ####
#____01.train_dataset_and_loader ####

set.seed(5446)
torch_manual_seed(5446)

# set up the train data set generator...
train_dsg <- dataset(
  name = "train_dsg",
  initialize = function(df_train_index) {
    self$train <- df_train_index
  },
  .getitem = function(i) {
    x1 <- image_read(self$train$path[i]) %>%
      image_scale(., "128x128") %>%
      image_convert(colorspace = "Gray") %>%
      transform_to_tensor()
    x <- x1[1]$unsqueeze(1)
    y <- torch_tensor(self$train$label[i])$squeeze(1)
    list(x = x, y = y)
  },
  .length = function() {
    nrow(self$train)
  }
)

# generate the train data set generator...
train_ds <- train_dsg(df_train_index)

# set up the train data loader...
train_dl <- dataloader(train_ds,
                       batch_size = 64,
                       shuffle = TRUE)

# load a test batch and check tensor dimensions...
# batch <- train_dl %>%
#   dataloader_make_iter() %>%
#   dataloader_next()
# 
# batch[[1]]$size() # 64 images, 1 gray, 128 length, 128 width...
# batch[[2]]$size() # 64 labels...

#____02.val_dataset_and_loader ####
# set up the val data set generator...
val_dsg <- dataset(
  name = "val_dsg",
  initialize = function(df_val_index) {
    self$val <- df_val_index
  },
  .getitem = function(i) {
    x1 <- image_read(self$val$path[i]) %>%
      image_scale(., "128x128") %>%
      image_convert(colorspace = "Gray") %>%
      transform_to_tensor()
    x <- x1[1]$unsqueeze(1)
    y <- torch_tensor(self$val$label[i])$squeeze(1)
    list(x = x, y = y)
  },
  .length = function() {
    nrow(self$val)
  }
)

# generate the val data set...
val_ds <- val_dsg(df_val_index)

# set up the val data loader...
val_dl <- dataloader(val_ds,
                     batch_size = 64)

# load a test batch and check tensor dimensions...
# batch <- val_dl %>%
#   dataloader_make_iter() %>%
#   dataloader_next()
# 
# batch[[1]]$size() # 64 images, 1 gray, 128 length, 128 width...
# batch[[2]]$size() # 64 labels...

#____03.model_architecture_and_learning_rate ####

convnet <- nn_module(
  "convnet",
  initialize = function(){
    self$features <- nn_sequential(
      nn_conv2d(1, 128, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_dropout2d(p = 0.05),
      nn_conv2d(128, 256, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_dropout2d(p = 0.05),
      nn_conv2d(256, 512, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_dropout2d(p = 0.05),
      nn_conv2d(512, 1024, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_dropout2d(p = 0.05),
      nn_conv2d(1024, 1024, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_adaptive_avg_pool2d(c(1, 1)),
      nn_dropout2d(p = 0.05)
    )
    self$classifier <- nn_sequential(
      nn_linear(1024, 1024),
      nn_relu(),
      nn_dropout(p = 0.05),
      nn_linear(1024, 1024),
      nn_relu(),
      nn_dropout(p = 0.05),
      nn_linear(1024, 3)
    )
  },
  forward = function(x){
    x <- self$features(x)$squeeze()
    x <- self$classifier(x)
  }
)

model <- convnet %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz_metric_accuracy()
    ))

# find the optimum learning rate...
lr_cnn_gray <- model %>%
  lr_finder(train_dl)
 
# write the lr summary to file for .Rmd...
dir.create(path = paste(getwd(), "results/cnn_gray", sep = "/"))
path_w <- paste(getwd(), "results/cnn_gray", sep = "/")
file_w <- "lr_cnn_gray.csv"
write_csv(lr_cnn_gray, paste(path_w, file_w, sep = "/"))

# plot loss b log(lr) to select learning rate...
lr_cnn_gray %>%
  mutate(log = log(lr)) %>%
  mutate(label = ifelse(row_number() %% 5 == 1,
                        lr,
                        NA)) %>%
  mutate(label = round(label, 5)) %>%
  ggplot() +
  geom_line(aes(x = log, y = loss),
            linewidth = 2, color = "blue", alpha = 0.3) +
  geom_point(data = . %>% filter(!is.na(label)),
             aes(x = log, y = loss),
             size = 1.5, color = "blue") +
  geom_text(aes(x = log, y = 0, label = label),
            size = 3, angle = 270, hjust = 1, vjust = 0) +
  theme_bw()

#____04.model_training ####
# NB: after lr estimation, make sure to redefine the "data set/loader", "convnet" and "model" before fitting...
set.seed(5446)
torch_manual_seed(5446)

# set up the train data set generator...
train_dsg <- dataset(
  name = "train_dsg",
  initialize = function(df_train_index) {
    self$train <- df_train_index
  },
  .getitem = function(i) {
    x1 <- image_read(self$train$path[i]) %>%
      image_scale(., "128x128") %>%
      image_convert(colorspace = "Gray") %>%
      transform_to_tensor()
    x <- x1[1]$unsqueeze(1)
    y <- torch_tensor(self$train$label[i])$squeeze(1)
    list(x = x, y = y)
  },
  .length = function() {
    nrow(self$train)
  }
)

# generate the train data set generator...
train_ds <- train_dsg(df_train_index)

# set up the train data loader...
train_dl <- dataloader(train_ds,
                       batch_size = 64,
                       shuffle = TRUE)

# set up the val data set generator...
val_dsg <- dataset(
  name = "val_dsg",
  initialize = function(df_val_index) {
    self$val <- df_val_index
  },
  .getitem = function(i) {
    x1 <- image_read(self$val$path[i]) %>%
      image_scale(., "128x128") %>%
      image_convert(colorspace = "Gray") %>%
      transform_to_tensor()
    x <- x1[1]$unsqueeze(1)
    y <- torch_tensor(self$val$label[i])$squeeze(1)
    list(x = x, y = y)
  },
  .length = function() {
    nrow(self$val)
  }
)

# generate the val data set...
val_ds <- val_dsg(df_val_index)

# set up the val data loader...
val_dl <- dataloader(val_ds,
                     batch_size = 64)

convnet <- nn_module(
  "convnet",
  initialize = function(){
    self$features <- nn_sequential(
      nn_conv2d(1, 128, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_dropout2d(p = 0.05),
      nn_conv2d(128, 256, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_dropout2d(p = 0.05),
      nn_conv2d(256, 512, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_dropout2d(p = 0.05),
      nn_conv2d(512, 1024, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_dropout2d(p = 0.05),
      nn_conv2d(1024, 1024, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_adaptive_avg_pool2d(c(1, 1)),
      nn_dropout2d(p = 0.05)
    )
    self$classifier <- nn_sequential(
      nn_linear(1024, 1024),
      nn_relu(),
      nn_dropout(p = 0.05),
      nn_linear(1024, 1024),
      nn_relu(),
      nn_dropout(p = 0.05),
      nn_linear(1024, 3)
    )
  },
  forward = function(x){
    x <- self$features(x)$squeeze()
    x <- self$classifier(x)
  }
)

model <- convnet %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz_metric_accuracy()
    ))

fitted <- model %>%
  fit(
    train_dl,
    epochs = 50,
    valid_data = val_dl,
    callbacks = list(
      luz_callback_early_stopping(patience = 5),
      luz_callback_lr_scheduler(
        lr_one_cycle,
        max_lr = 0.001,
        epochs = 50,
        steps_per_epoch = length(train_dl),
        call_on = "on_batch_end"),
      luz_callback_model_checkpoint(path = "results/cnn_gray/checkpoints/"),
      luz_callback_csv_logger("results/cnn_gray/log_gray.csv")
    ),
    verbose = TRUE)

#____05.evaluate ####

# evaluate epochs and plot loss/accuracy...
log <- read_csv("results/cnn_gray/log_gray.csv",
                show_col_types = FALSE)

log %>%
  rename(dataset = set) %>%
  gather(key = "key", value = "value", -c("epoch", "dataset")) %>%
  ggplot() +
  geom_line(aes(x = epoch, y = value, color = dataset),
            linewidth = 1.5, alpha = 0.3) +
  geom_point(aes(x = epoch, y = value, color = dataset)) +
  scale_color_manual(values = c("grey60", "blue")) +
  facet_wrap(~key, ncol = 1, strip.position = "right", scales = "free_y") +
  scale_x_continuous(breaks = seq(0, 50, by = 5),
                     minor_breaks = seq(0, 50, by = 1)) +
  theme_bw()

# select the best epoch...
epoch_cnn_gray <- log %>%
  filter(set == "valid") %>%
  filter(acc == max(acc))
epoch_cnn_gray

# load the selected checkpoint for evaluation...
path_r <- paste(getwd(), "results/cnn_gray/checkpoints", sep = "/")
pattern <- paste0("epoch-", epoch_cnn_gray$epoch)

epoch_oi <- list.files(path = path_r, pattern = pattern)
epoch_oi <- paste(getwd(), "results/cnn_gray/checkpoints", epoch_oi, sep = "/")

luz_load_checkpoint(fitted, epoch_oi)

y_hat_cnn_gray <- fitted %>%
  predict(val_dl) %>%
  nnf_softmax(., dim = 2) %>%
  torch_argmax(., dim = 2) %>%
  as.numeric(.) %>%
  factor(., levels = c(1:3))

# save the confusion matrix for later analysis...
cm_cnn_gray <- confusionMatrix(y_hat_cnn_gray, df_val_index$label) 
cm_cnn_gray # accuracy = 0.5414 at epoch 25...

#__02.cnn_rgb ####
#____01.train_dataset_and_loader ####

set.seed(5446)
torch_manual_seed(5446)

# set up the train data set generator...
train_dsg <- dataset(
  name = "train_dsg",
  initialize = function(df_train_index) {
    self$train <- df_train_index
  },
  .getitem = function(i) {
    x <- image_read(self$train$path[i]) %>%
      image_scale(., "128x128") %>%
      transform_to_tensor() %>%
      transform_normalize(mean = c(0.485, 0.456, 0.406),
                          std = c(0.229, 0.224, 0.225))
    y <- torch_tensor(self$train$label[i])$squeeze(1)
    list(x = x, y = y)
  },
  .length = function() {
    nrow(self$train)
  }
)

# generate the train data set generator...
train_ds <- train_dsg(df_train_index)

# set up the train data loader...
train_dl <- dataloader(train_ds,
                       batch_size = 64,
                       shuffle = TRUE)

# load a test batch and check tensor dimensions...
# batch <- train_dl %>%
#   dataloader_make_iter() %>%
#   dataloader_next()
# 
# batch[[1]]$size() # 64 images, 3 RGB, 128 length, 128 width...
# batch[[2]]$size() # 64 labels...

#____02.val_dataset_and_loader ####
# set up the val data set generator...
val_dsg <- dataset(
  name = "val_dsg",
  initialize = function(df_val_index) {
    self$val <- df_val_index
  },
  .getitem = function(i) {
    x <- image_read(self$val$path[i]) %>%
      image_scale(., "128x128") %>%
      transform_to_tensor() %>%
      transform_normalize(mean = c(0.485, 0.456, 0.406),
                          std = c(0.229, 0.224, 0.225))
    y <- torch_tensor(self$val$label[i])$squeeze(1)
    list(x = x, y = y)
  },
  .length = function() {
    nrow(self$val)
  }
)

# generate the val data set...
val_ds <- val_dsg(df_val_index)

# set up the val data loader...
val_dl <- dataloader(val_ds,
                     batch_size = 64)

# load a test batch and check tensor dimensions...
# batch <- val_dl %>%
#   dataloader_make_iter() %>%
#   dataloader_next()
# 
# batch[[1]]$size() # 64 images, 3 RGB, 128 length, 128 width...
# batch[[2]]$size() # 64 labels...

#____03.model_architecture_and_learning_rate ####

convnet <- nn_module(
  "convnet",
  initialize = function(){
    self$features <- nn_sequential(
      nn_conv2d(3, 128, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_dropout2d(p = 0.05),
      nn_conv2d(128, 256, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_dropout2d(p = 0.05),
      nn_conv2d(256, 512, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_dropout2d(p = 0.05),
      nn_conv2d(512, 1024, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_dropout2d(p = 0.05),
      nn_conv2d(1024, 1024, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_adaptive_avg_pool2d(c(1, 1)),
      nn_dropout2d(p = 0.05)
    )
    self$classifier <- nn_sequential(
      nn_linear(1024, 1024),
      nn_relu(),
      nn_dropout(p = 0.05),
      nn_linear(1024, 1024),
      nn_relu(),
      nn_dropout(p = 0.05),
      nn_linear(1024, 3)
    )
  },
  forward = function(x){
    x <- self$features(x)$squeeze()
    x <- self$classifier(x)
  }
)

model <- convnet %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz_metric_accuracy()
    ))

# find the optimum learning rate...
lr_cnn_rgb <- model %>%
  lr_finder(train_dl)

# write the lr summary to file for .Rmd...
dir.create(path = paste(getwd(), "results/cnn_rgb", sep = "/"))
path_w <- paste(getwd(), "results/cnn_rgb", sep = "/")
file_w <- "lr_cnn_rgb.csv"
write_csv(lr_cnn_rgb, paste(path_w, file_w, sep = "/"))

# plot loss b log(lr) to select learning rate...
lr_cnn_rgb %>%
  mutate(log = log(lr)) %>%
  mutate(label = ifelse(row_number() %% 5 == 1,
                        lr,
                        NA)) %>%
  mutate(label = round(label, 5)) %>%
  ggplot() +
  geom_line(aes(x = log, y = loss),
            linewidth = 2, color = "blue", alpha = 0.3) +
  geom_point(data = . %>% filter(!is.na(label)),
             aes(x = log, y = loss),
             size = 1.5, color = "blue") +
  geom_text(aes(x = log, y = 0, label = label),
            size = 3, angle = 270, hjust = 1, vjust = 0) +
  theme_bw()

#____04.model_training ####
# NB: after lr estimation, make sure to redefine the "data set/loader", "convnet" and "model" before fitting...
set.seed(5446)
torch_manual_seed(5446)

# set up the train data set generator...
train_dsg <- dataset(
  name = "train_dsg",
  initialize = function(df_train_index) {
    self$train <- df_train_index
  },
  .getitem = function(i) {
    x <- image_read(self$train$path[i]) %>%
      image_scale(., "128x128") %>%
      transform_to_tensor() %>%
      transform_normalize(mean = c(0.485, 0.456, 0.406),
                          std = c(0.229, 0.224, 0.225))
    y <- torch_tensor(self$train$label[i])$squeeze(1)
    list(x = x, y = y)
  },
  .length = function() {
    nrow(self$train)
  }
)

# generate the train data set generator...
train_ds <- train_dsg(df_train_index)

# set up the train data loader...
train_dl <- dataloader(train_ds,
                       batch_size = 64,
                       shuffle = TRUE)

val_dsg <- dataset(
  name = "val_dsg",
  initialize = function(df_val_index) {
    self$val <- df_val_index
  },
  .getitem = function(i) {
    x <- image_read(self$val$path[i]) %>%
      image_scale(., "128x128") %>%
      transform_to_tensor() %>%
      transform_normalize(mean = c(0.485, 0.456, 0.406),
                          std = c(0.229, 0.224, 0.225))
    y <- torch_tensor(self$val$label[i])$squeeze(1)
    list(x = x, y = y)
  },
  .length = function() {
    nrow(self$val)
  }
)

# generate the val data set...
val_ds <- val_dsg(df_val_index)

# set up the val data loader...
val_dl <- dataloader(val_ds,
                     batch_size = 64)

convnet <- nn_module(
  "convnet",
  initialize = function(){
    self$features <- nn_sequential(
      nn_conv2d(3, 128, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_dropout2d(p = 0.05),
      nn_conv2d(128, 256, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_dropout2d(p = 0.05),
      nn_conv2d(256, 512, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_dropout2d(p = 0.05),
      nn_conv2d(512, 1024, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_dropout2d(p = 0.05),
      nn_conv2d(1024, 1024, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_adaptive_avg_pool2d(c(1, 1)),
      nn_dropout2d(p = 0.05)
    )
    self$classifier <- nn_sequential(
      nn_linear(1024, 1024),
      nn_relu(),
      nn_dropout(p = 0.05),
      nn_linear(1024, 1024),
      nn_relu(),
      nn_dropout(p = 0.05),
      nn_linear(1024, 3)
    )
  },
  forward = function(x){
    x <- self$features(x)$squeeze()
    x <- self$classifier(x)
  }
)

model <- convnet %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz_metric_accuracy()
    ))

fitted <- model %>%
  fit(
    train_dl,
    epochs = 50,
    valid_data = val_dl,
    callbacks = list(
      luz_callback_early_stopping(patience = 3),
      luz_callback_lr_scheduler(
        lr_one_cycle,
        max_lr = 0.05,
        epochs = 50,
        steps_per_epoch = length(train_dl),
        call_on = "on_batch_end"),
      luz_callback_model_checkpoint(path = "results/cnn_rgb/checkpoints/"),
      luz_callback_csv_logger("results/cnn_rgb/log_rgb.csv")
    ),
    verbose = TRUE)

#____05.evaluate ####

# evaluate epochs and plot loss/accuracy...
log <- read_csv("results/cnn_rgb/log_rgb.csv",
                show_col_types = FALSE)
log %>%
  rename(dataset = set) %>%
  gather(key = "key", value = "value", -c("epoch", "dataset")) %>%
  ggplot() +
  geom_line(aes(x = epoch, y = value, color = dataset),
            linewidth = 1.5, alpha = 0.3) +
  geom_point(aes(x = epoch, y = value, color = dataset)) +
  scale_color_manual(values = c("grey60", "blue")) +
  facet_wrap(~key, ncol = 1, strip.position = "right", scales = "free_y") +
  scale_x_continuous(breaks = seq(0, 50, by = 5),
                     minor_breaks = seq(0, 50, by = 1)) +
  theme_bw()

# select the best epoch...
epoch_cnn_rgb <- log %>%
  filter(set == "valid") %>%
  filter(acc == max(acc))
epoch_cnn_rgb

# load the selected checkpoint for evaluation...
path_r <- paste(getwd(), "results/cnn_rgb/checkpoints", sep = "/")
pattern <- paste0("epoch-", epoch_cnn_rgb$epoch)

epoch_oi <- list.files(path = path_r, pattern = pattern)
epoch_oi <- paste(getwd(), "results/cnn_rgb/checkpoints", epoch_oi, sep = "/")

luz_load_checkpoint(fitted, epoch_oi)

y_hat_cnn_rgb <- fitted %>%
  predict(val_dl) %>%
  nnf_softmax(., dim = 2) %>%
  torch_argmax(., dim = 2) %>%
  as.numeric(.) %>%
  factor(., levels = c(1:3))

# save the confusion matrix for later analysis...
cm_cnn_rgb <- confusionMatrix(y_hat_cnn_rgb, df_val_index$label) 
cm_cnn_rgb # accuracy = 0.6767 at epoch 17...

#__03.cnn_resnet ####
#____01.train_dataset_and_loader ####

set.seed(5446)
torch_manual_seed(5446)

# set up the train data set generator...
train_dsg <- dataset(
  name = "train_dsg",
  initialize = function(df_train_index) {
    self$train <- df_train_index
  },
  .getitem = function(i) {
    x <- image_read(self$train$path[i]) %>%
      image_scale(., "128x128") %>%
      transform_to_tensor() %>%
      transform_normalize(mean = c(0.485, 0.456, 0.406),
                          std = c(0.229, 0.224, 0.225))
    y <- torch_tensor(self$train$label[i])$squeeze(1)
    list(x = x, y = y)
  },
  .length = function() {
    nrow(self$train)
  }
)

# generate the train data set generator...
train_ds <- train_dsg(df_train_index)

# set up the train data loader...
train_dl <- dataloader(train_ds,
                       batch_size = 64,
                       shuffle = TRUE)

# load a test batch and check tensor dimensions...
# batch <- train_dl %>%
#   dataloader_make_iter() %>%
#   dataloader_next()
# 
# batch[[1]]$size() # 64 images, 3 RGB, 128 length, 128 width...
# batch[[2]]$size() # 64 labels...

#____02.val_dataset_and_loader ####
# set up the val data set generator...
val_dsg <- dataset(
  name = "val_dsg",
  initialize = function(df_val_index) {
    self$val <- df_val_index
  },
  .getitem = function(i) {
    x <- image_read(self$val$path[i]) %>%
      image_scale(., "128x128") %>%
      transform_to_tensor() %>%
      transform_normalize(mean = c(0.485, 0.456, 0.406),
                          std = c(0.229, 0.224, 0.225))
    y <- torch_tensor(self$val$label[i])$squeeze(1)
    list(x = x, y = y)
  },
  .length = function() {
    nrow(self$val)
  }
)

# generate the val data set...
val_ds <- val_dsg(df_val_index)

# set up the val data loader...
val_dl <- dataloader(val_ds,
                     batch_size = 64)

# load a test batch and check tensor dimensions...
# batch <- val_dl %>%
#   dataloader_make_iter() %>%
#   dataloader_next()
# 
# batch[[1]]$size() # 64 images, 3 RGB, 128 length, 128 width...
# batch[[2]]$size() # 64 labels...

#____03.model_architecture_and_learning_rate ####
# NB: resnet model uses the same dataset and loader as cnn_rgb...

convnet <- nn_module(
  initialize = function(){
    self$model <- model_resnet18(pretrained = TRUE)
    for(par in self$parameters){
      par$requires_grad_(FALSE)
    }
    self$model$fc <- nn_sequential(
      nn_linear(self$model$fc$in_features, 1024),
      nn_relu(),
      nn_linear(1024, 1024),
      nn_relu(),
      nn_linear(1024, 3)
    )
  },
  forward = function(x){
    self$model(x)
  }
)

model <- convnet %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz_metric_accuracy()
    ))

# find the optimum learning rate...
lr_cnn_resnet <- model %>%
  lr_finder(train_dl)

# write the lr summary to file for .Rmd...
dir.create(path = paste(getwd(), "results/cnn_resnet", sep = "/"))
path_w <- paste(getwd(), "results/cnn_resnet", sep = "/")
file_w <- "lr_cnn_resnet.csv"
write_csv(lr_cnn_resnet, paste(path_w, file_w, sep = "/"))

# plot loss b log(lr) to select learning rate...
lr_cnn_resnet %>%
  mutate(log = log(lr)) %>%
  mutate(label = ifelse(row_number() %% 5 == 1,
                        lr,
                        NA)) %>%
  mutate(label = round(label, 5)) %>%
  ggplot() +
  geom_line(aes(x = log, y = loss),
            linewidth = 2, color = "blue", alpha = 0.3) +
  geom_point(data = . %>% filter(!is.na(label)),
             aes(x = log, y = loss),
             size = 1.5, color = "blue") +
  geom_text(aes(x = log, y = 0, label = label),
            size = 3, angle = 270, hjust = 1, vjust = 0) +
  theme_bw() 

#____04.model_training ####
# NB: after lr estimation, make sure to redefine the "data set/loader", "convnet" and "model" before fitting...
set.seed(5446)
torch_manual_seed(5446)

# set up the train data set generator...
train_dsg <- dataset(
  name = "train_dsg",
  initialize = function(df_train_index) {
    self$train <- df_train_index
  },
  .getitem = function(i) {
    x <- image_read(self$train$path[i]) %>%
      image_scale(., "128x128") %>%
      transform_to_tensor() %>%
      transform_normalize(mean = c(0.485, 0.456, 0.406),
                          std = c(0.229, 0.224, 0.225))
    y <- torch_tensor(self$train$label[i])$squeeze(1)
    list(x = x, y = y)
  },
  .length = function() {
    nrow(self$train)
  }
)

# generate the train data set generator...
train_ds <- train_dsg(df_train_index)

# set up the train data loader...
train_dl <- dataloader(train_ds,
                       batch_size = 64,
                       shuffle = TRUE)

# set up the val data set generator...
val_dsg <- dataset(
  name = "val_dsg",
  initialize = function(df_val_index) {
    self$val <- df_val_index
  },
  .getitem = function(i) {
    x <- image_read(self$val$path[i]) %>%
      image_scale(., "128x128") %>%
      transform_to_tensor() %>%
      transform_normalize(mean = c(0.485, 0.456, 0.406),
                          std = c(0.229, 0.224, 0.225))
    y <- torch_tensor(self$val$label[i])$squeeze(1)
    list(x = x, y = y)
  },
  .length = function() {
    nrow(self$val)
  }
)

# generate the val data set...
val_ds <- val_dsg(df_val_index)

# set up the val data loader...
val_dl <- dataloader(val_ds,
                     batch_size = 64)

convnet <- nn_module(
  initialize = function(){
    self$model <- model_resnet18(pretrained = TRUE)
    for(par in self$parameters){
      par$requires_grad_(FALSE)
    }
    self$model$fc <- nn_sequential(
      nn_linear(self$model$fc$in_features, 1024),
      nn_relu(),
      nn_linear(1024, 1024),
      nn_relu(),
      nn_linear(1024, 3)
    )
  },
  forward = function(x){
    self$model(x)
  }
)

model <- convnet %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz_metric_accuracy()
    ))

fitted <- model %>%
  fit(
    train_dl,
    epochs = 50,
    valid_data = val_dl,
    callbacks = list(
      luz_callback_early_stopping(patience = 3),
      luz_callback_lr_scheduler(
        lr_one_cycle,
        max_lr = 0.001,
        epochs = 50,
        steps_per_epoch = length(train_dl),
        call_on = "on_batch_end"),
      luz_callback_model_checkpoint(path = "results/cnn_resnet/checkpoints/"),
      luz_callback_csv_logger("results/cnn_resnet/log_resnet.csv")
    ),
    verbose = TRUE)

#____05.evaluate ####

# evaluate epochs and plot loss/accuracy...
log <- read_csv("results/cnn_resnet/log_resnet.csv",
                show_col_types = FALSE)
log %>%
  rename(dataset = set) %>%
  gather(key = "key", value = "value", -c("epoch", "dataset")) %>%
  ggplot() +
  geom_line(aes(x = epoch, y = value, color = dataset),
            linewidth = 1.5, alpha = 0.3) +
  geom_point(aes(x = epoch, y = value, color = dataset)) +
  scale_color_manual(values = c("grey60", "blue")) +
  facet_wrap(~key, ncol = 1, strip.position = "right", scales = "free_y") +
  scale_x_continuous(breaks = seq(0, 50, by = 5),
                     minor_breaks = seq(0, 50, by = 1)) +
  theme_bw()

# select the best epoch...
epoch_cnn_resnet <- log %>%
  filter(set == "valid") %>%
  filter(acc == max(acc))
epoch_cnn_resnet

# load the selected checkpoint for evaluation...
path_r <- paste(getwd(), "results/cnn_resnet/checkpoints", sep = "/")
pattern <- paste0("epoch-", epoch_cnn_resnet$epoch)

epoch_oi <- list.files(path = path_r, pattern = pattern)
epoch_oi <- paste(getwd(), "results/cnn_resnet/checkpoints", epoch_oi, sep = "/")

luz_load_checkpoint(fitted, epoch_oi)

y_hat_cnn_resnet <- fitted %>%
  predict(val_dl) %>%
  nnf_softmax(., dim = 2) %>%
  torch_argmax(., dim = 2) %>%
  as.numeric(.) %>%
  factor(., levels = c(1:3))

# save the confusion matrix for later analysis...
cm_cnn_resnet <- confusionMatrix(y_hat_cnn_resnet, df_val_index$label) 
cm_cnn_resnet # accuracy = 0.8647 at epoch 11...

#__04.cnn_summary #####

# get key results into a summary table...
cnn_summary <- data.frame(
  id = 11:13,
  model = c("cnn_gray", "cnn_rgb", "resnet_18"),
  input = c("gray-scale", "rgb", "rgb"),
  type = c("custom", "custom", "pre-trained"),
  layers = c(7, 7, 18),
  learn_rate = c(0.001, 0.05, 0.001),
  epoch = c(epoch_cnn_gray$epoch, epoch_cnn_rgb$epoch, epoch_cnn_resnet$epoch),
  loss = c(epoch_cnn_gray$loss, epoch_cnn_rgb$loss, epoch_cnn_resnet$loss),
  accuracy = c(epoch_cnn_gray$acc, epoch_cnn_rgb$acc, epoch_cnn_resnet$acc)) %>%
  mutate(loss = round(loss, 3),
         accuracy = round(accuracy, 3))

# write the summary to file for .Rmd...
dir.create(path = paste(getwd(), "results", sep = "/"))
path_w <- paste(getwd(), "results", sep = "/")
file_w <- "cnn_summary.csv"
write_csv(cnn_summary, paste(path_w, file_w, sep = "/"))

# flextable for reporting...
cnn_summary %>% 
  flextable(.) %>%
  fontsize(., size = 10, part = 'all') %>%
  align(., align = 'center', part = 'header') %>%
  align(., align = 'center', j = c(1, 5:9)) %>%
  width(., width = 0.8, j = c(2:4)) %>%
  width(., width = 0.6, j = c(1, 5:9))

# cleanup after CNN...
rm(train_dsg, train_dl, val_dsg, val_dl)
rm(lr_cnn_gray, lr_cnn_rgb, lr_cnn_resnet)
rm(epoch_cnn_gray, epoch_cnn_rgb, epoch_cnn_resnet)
rm(y_hat_cnn_gray, y_hat_cnn_rgb, y_hat_cnn_resnet)
rm(batch, log, pattern, epoch_oi, plot)
rm(train_ds, val_ds, fitted, model, convnet)

#05.SUMMARY RESULTS ####

# create a summary table of confusion matrix results...
cm_summary <- cm_knn_ensemble$table %>%
  data.frame() %>%
  mutate(model = "knn_ensemble") %>%
  rbind(., cm_pca_blue$table %>%
          data.frame() %>%
          mutate(model = "pca_blue")) %>%
  rbind(., cm_cnn_resnet$table %>%
          data.frame() %>%
          mutate(model = "cnn_resnet")) %>%
  select(model, everything()) %>%
  mutate(model = factor(model, levels = c("knn_ensemble", "pca_blue", "cnn_resnet")))

# write the confusion matrix summary to file for .Rmd...
dir.create(path = paste(getwd(), "results", sep = "/"))
path_w <- paste(getwd(), "results", sep = "/")
file_w <- "cm_summary.csv"
write_csv(cm_summary, paste(path_w, file_w, sep = "/"))

# plot confusion matrices...
cm_summary %>%
  mutate(Prediction = factor(Prediction, levels = 1:3)) %>%
  ggplot() +
  geom_tile(aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_text(aes(x = Reference, y = Prediction, label = Freq)) +
  facet_wrap(~model, nrow = 1) +
  scale_fill_gradientn(colors = c("white", "yellow1", "green4"),
                       values = c(0, 0.5, 1)) +
  scale_y_discrete(limits=rev) +
  xlab("actual class") +
  ylab("predicted class") +
  theme_bw()

# create a summary table of performance metrics...
perf_summary <- cm_knn_ensemble$byClass[,5:7] %>%
  data.frame() %>%
  rownames_to_column(., var = "class") %>%
  mutate(class = str_extract(class, "[0-9]")) %>%
  mutate(model = "knn_ensemble") %>%
  rbind(., cm_pca_blue$byClass[,5:7] %>%
          data.frame() %>%
          rownames_to_column(., var = "class") %>%
          mutate(class = str_extract(class, "[0-9]")) %>%
          mutate(model = "pca_blue")) %>%
  rbind(., cm_cnn_resnet$byClass[,5:7] %>%
          data.frame() %>%
          rownames_to_column(., var = "class") %>%
          mutate(class = str_extract(class, "[0-9]")) %>%
          mutate(model = "cnn_resnet")) %>%
  select(model, everything()) %>%
  mutate(model = factor(model, levels = c("knn_ensemble", "pca_blue", "cnn_resnet"))) %>%
  gather(key = "metric", value = "value", -c(model, class)) %>%
  mutate(metric = factor(metric, levels = c("Precision", "Recall", "F1")))

# write the confusion matrix summary to file for .Rmd...
dir.create(path = paste(getwd(), "results", sep = "/"))
path_w <- paste(getwd(), "results", sep = "/")
file_w <- "perf_summary.csv"
write_csv(perf_summary, paste(path_w, file_w, sep = "/"))

perf_summary %>%
  ggplot() +
  geom_bar(aes(x = class, y = value, fill = model),
           stat = "identity", position = "dodge",
           color = "grey40", alpha = 0.7) +
  geom_text(aes(x = class, y = value + 0.1, group = model, label = round(value, 2)),
            position = position_dodge(width = 0.9),
            size = 4) +
  scale_y_continuous(breaks = seq(0, 1.1, by = 0.2),
                     minor_breaks = seq(0, 1.1, by = 0.1),
                     limits = c(0, 1.1)) +
  facet_wrap(~metric, ncol = 1, strip.position = "left") +
  theme_bw()
