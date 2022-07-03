# ----- CleanImageData -----
# The uniform size of the square image for training
targetSize=512
# The folder where all images are stored
imageFolder="images/"
# The folder where all cleaned images are stored
targetFolder="cleanImage/"
# The maximum range of pixels in the image
dataRange=255

# ----- CleanTabularData -----
# File path of product information
productPath="./Data/Products.csv"
# The terminator used in the csv file
lineTerminator="\n"

# ----- TextProcessing -----
# The setting of dataset
datasetSetting={
    # The name of columns that could be used as features
    "featureColumn":["product_name", "product_description", "location"],
    # The name of the column that could be used as label
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




