import pandas as pd

from Configuration import productPath

def cleanProduct(productPath:str=productPath, lineterminator:str="\n") -> pd.DataFrame:
    '''
        Return a filtered dataframe that could be used for
    further training

        To clean the dataframe, the following steps are performed:
        1. Delete the column "Unamed: 0"
        2. Format the column "Price" so each value is a float without symbol
        3. Change the column "category" to category data type
        4. Change the column "page_id" to object data type
    
    Argument:
        productPath: string, the file path to the product data file
        lineterminator: string, the terminator used to represent the termination
                        in the csv file
                    
    Return:
        dataFrame: pd.DataFrame,
    '''
    dataFrame=pd.read_csv(productPath, lineterminator="\n")

    dataFrame=dataFrame.drop("Unnamed: 0", axis=1)

    dataFrame["price"]=dataFrame["price"].apply(formatPrice)

    dataFrame["category"]=dataFrame["category"].astype("category")
    dataFrame["page_id"]=dataFrame["page_id"].astype("str")

    return dataFrame

def formatPrice(price:str) -> float:
    '''
        Format the price and return it as a float

        Generally the price is in the format: £xxx,xxx.xxx
    
    Argument:
        price: string, the price information with currency symbol
    
    Return:
        result: float, the price information as float
    '''
    if "," not in price:
        return float(price[1:])
    else:
        digit=price[1:].split(",")
        digit=[float(digit[i])*1000**(len(digit)-1-i) for i in range(len(digit))]
        return sum(digit)







