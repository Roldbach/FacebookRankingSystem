# FacebookRankingSystem
## Milestone 2
- Within this project there are mainly 2 types of data that could be used for training machine learning models: **text data** and **image data**.
- To clean the **text data**, the following steps are performed:

    1. Delete the first column *Unamed: 0*
    2. Format the column *Price* so that each value is a valid float without the currency symbol "<span>&#163;</span>"
    3. Change the column *category* from object data type to category data type, which could save some memory space
    4. Change the column *page_id* from integer data type to object data type
    
- To clean the **image data**, the following steps are performed:

    1. Resize the image into a square with the same length, this will crop the image if originally it is a rectangle.
    2. Normalise the pixel values in the image so that they are all within the range 0~1 as float
    3. Save those new images to a new space so they could directly loaded for further model training.