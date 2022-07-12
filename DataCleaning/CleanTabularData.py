import pandas as pd

from Configuration import lineTerminator, cleanProductName, productPath, cleanProductPath

def cleanProduct(productPath:str=productPath, lineterminator:str=lineTerminator, cleanProductName:str=cleanProductName, cleanProductPath:str=cleanProductPath):
    '''
        Clean the product csv file and save the result to the
    target position

        To clean the dataframe, the following steps are performed:
        1. Delete the column "Unamed: 0"
        2. Format the column "Price" so each value is a float without symbol
        3. Add an extra column "Label" representing the numerical label of the
           highest-level category information
    
    Argument:
        productPath: string, the file path to the product data file
        lineterminator: string, the terminator used to represent the termination
                        in the csv file
        cleanProductName: string, the name of the processed product information
        cleanProductPath: string, file path of processed product information
    '''
    dataFrame=pd.read_csv(productPath, lineterminator=lineterminator)

    dataFrame=dataFrame.drop("Unnamed: 0", axis=1)
    dataFrame["price"]=dataFrame["price"].apply(formatPrice)
    dataFrame=addLabel(dataFrame)

    dataFrame.to_csv(cleanProductPath+"/"+cleanProductName+".csv", index=False, line_terminator=lineterminator)

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

def addLabel(dataFrame:pd.DataFrame) -> pd.DataFrame:
    '''
        Add an extra column of numerical labels which
    represents the highest level category information of
    a product to the product information

    Argument:
        dataFrame: pd.DataFrame, the product information after cleaning
    
    Return:
        result; pd.DataFrame, the product information with additional labels
    '''
    relation=constructRelation(dataFrame["category"])
    dataFrame["label"]=dataFrame["category"].apply(lambda category: relation[formatCategory(category)])

    return dataFrame

def constructRelation(column:pd.Series) -> dict[str,int]:
    '''
        Return a dictionary which could convert the high-level
    category information into numerical labels

    Argument:
        column: pd.Series, the column containing original category information
    
    Return:
        result: dict[str,int], key=distinct high-level categories, value=correspondig numerical label
    '''
    result={}
    label=0

    column=[formatCategory(category) for category in column]
    for category in column:
        if category not in result:
            result[category]=label
            label+=1

    return result

def formatCategory(category:str) -> str:
    '''
        Extract the highest level of category from
    the original category information
    
    Argument:
        category: str, the original category information
    
    Return:
        result: str, the highest category information
    '''
    result=category.split("/")
    return result[0].strip(" ")

def mapLabel(category:str, relation:dict[str,int]) -> int:
    '''
        Map the high-level category information to its
    corresponding numerical label

    Argument:
        category: string, the original category information
        relation: dict[str,int], key=distinct high-level categories, value=correspondig numerical label
    
    Return:
        result: int, the corresponding numerical label of the given category information
    '''
    return relation[category]





