# ----- CleanImageData -----
# The uniform size of the square image for training
targetSize=256
# The folder where all images are stored
imageFolder="./images"
# The folder where all cleaned images are stored
targetFolder="./Data/CleanImage"
# The maximum range of pixels in the image
dataRange=255
# File path of image information
imageInformationPath="./Data/Images.csv"
# File path of image loading information
imageLoadingPath="./Data/ImageLoading.csv"

# ----- CleanTabularData -----
# File path of product information
productPath="./Data/Products.csv"
# The terminator used in the csv file
lineTerminator="\n"
# The name of the processed product information
cleanProductName="CleanProduct"
# File path of processed product information
cleanProductPath="./Data"

# ----- TextProcessing -----
# The setting of text dataset for simple regression model
textDatasetSetting={
    # The name of columns that could be used as features
    "featureColumn":["product_name", "product_description", "location"],
    # The name of the column that could be used as labels
    "targetColumn":"price",
    # The proportion of test set within the whole dataset
    "testSize":0.33
}

# The setting of TFIDFVectorizer
TFIDFVectorizerSetting={
    # Only keep features with top frequencies
    "max_features":10000,
    # The lower threshold when building the vocabulary
    "min_df":0.01,
    # The upper threshold when building the vocabulary
    "max_df":0.9,
    # Use the built-in stop word list for English
    "stop_words":"english"
}

# ----- ImageProcessing -----
# The setting of image dataset for simple classification model
imageDatasetSetting={
    # The name of the column that contains loading paths
    "pathColumn":"path",
    # The name of the column that could be used as labels
    "targetColumn":"label",
    # The proportion of test set within the whole dataset
    "testSize":0.33
}





