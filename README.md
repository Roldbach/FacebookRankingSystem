# FacebookRankingSystem
## Milestone 2
- Within this project there are mainly 2 types of data that could be used for training machine learning models: **text data** and **image data**.
- To clean the **text data**, the following steps are performed:

    1. Delete the first column *Unamed: 0*
    
    1. Format the column *Price* so that each value is a valid float without the currency symbol "<span>&#163;</span>"

- To clean the **image data**, the following steps are performed:

    1. Convert the image into grayscale so it has only 1 channel.

    1. Crop the central part of the image into a square with the same length.
    
    1. Normalise the pixel values in the image so that they are all within the range 0~1 as float32.
    
    1. Save those arrays directly.

    1. Construct a data frame that contains all image ID, labels and the file paths to load pixel arrays.

## Milestone 3
- The text data was used to train a simple linear regression model:

    1. Apply TF-IDF to each feature column.

    1. Concatenate all feature columns as a whole

    1. Use **product name**, **product description** and **location** as feature information to predict the **price**.

- The linear regression model didn't perform well. There are several reason for this:

    1. The quality of the feature itself can't be guaranteed, which is highly likely to be very messy

    1. The linear regression model is not sufficient to learn the complex relationship, this might be able to sovled by either using more data and features or using more complex models like deep-learning models

- The image data was used to train a Support Vector Classifier model. However the training time was extremely long as there were too much data with no GPU suport for scikit-learn API.

- The performance of the model was not great and the reason has been mentinoed above.
