import numpy as np
import os
import pandas as pd

from Configuration import dataRange, imageFolder, imageInformationPath, imageLoadingPath, targetFolder, targetSize
from DataLoading.TextLoading import loadProduct
from PIL import Image, ImageOps

def cleanImage(imageFolder:str=imageFolder, targetFolder:str=targetFolder, targetSize:int=targetSize, dataRange:int=dataRange,
               imageInformationPath:str=imageInformationPath, imageLoadingPath:str=imageLoadingPath):
    '''
        Clean every image in the given folder and store it
    to another place for further training

        A loading file could be constructed simultaneously containing:
        1. The image id
        2. The numerical label
        3. The file path of the stored array

        Each image is cleaned by:
        1. Convert to grayscale if it's not
        2. Crop the central part of the image to the target dimension
        3. Normalise the image and convert it to a np.ndarray
        4. Directly save the np.ndarray as a npy file

    Argument:
        imageFolder: string, the folder where all images are stored
        targetFolder: string, the folder where all cleaned images are stored
        targetSize: int, the uniform dimension of the square image for training
        dataRange: int, the maximum range of the pixel
        imageInformationPath: string, file path of image information
        imageLoadingPath: string, file path of image loading information
    '''
    imageID, label=constructIDLabel(imageFolder, imageInformationPath)
    targetPath=[targetFolder+"/"+imageID[i]+".npy" for i in range(len(imageID))]

    for i in range(len(imageID)):
        image=Image.open(imageFolder+"/"+imageID[i]+".jpg")
        image=grayImage(image)
        image=cropImage(image, targetSize, targetSize)
        imageArray=normaliseImage(image, dataRange)

        np.save(targetPath[i], imageArray)
    
    constructImageLoading(imageID, label, targetPath, imageLoadingPath)

def constructIDLabel(imageFolder:str, imageInformationPath:str) -> tuple[list[str], list[int]]:
    '''
        Return a list of image ID and their corresponding labels after filtering

    Argument:
        imageFolder: string, the folder where all images are stored
        imageInformationPath: string, file path of image information
    '''
    imageID=constructImageID(imageFolder)
    label=constructLabel(imageID, imageInformationPath)
    return cleanLabel(imageID, label)

def constructImageID(imageFolder:str) -> list[str]:
    '''
        Return a list of unique ID for every image stored in the
    target folder

    Argument:
        imageFolder: str, the folder where all images are stored
    
    Return:
        result: list[str], contains the unique ID for each image
    '''
    imageList=sorted(os.listdir(imageFolder))
    return [path[:-4] for path in imageList]

def constructLabel(imageID:list[str], imageInformationPath:str) -> list[int]:
    '''
        Return a list of labels for the given image ID
    
    Argument:
        imageID: list[str], contains the unique ID for each image
        imageInformationPath: string, file path of image information
    
    Return:
        label: list[int], contains corresponding labels
    '''
    product=loadProduct()
    imageInformation=pd.read_csv(imageInformationPath)
    
    return [checkLabel(ID, product, imageInformation) for ID in imageID]

def checkLabel(imageID:str, product:pd.DataFrame, imageInformation:pd.DataFrame) -> int:
    '''
        Check and return the corresponding label according to the category information
    of the image

        If the label can't be found, return "-1" instead

    Argument:
        imageID: string, the unique ID of the image
        product: pd.DataFrame, the product information after adding labels
        imageInformation: pd.DataFrame, contains the corresponding product ID information
    
    Return:
        result: int, the corresponding label of the image according to the category
    '''
    try:
        productID=imageInformation.loc[imageInformation["id"]==imageID, "product_id"].values[0]
        return product.loc[product["id"]==productID, "label"].values[0]
    except:
        return -1

def cleanLabel(imageID:list[str], label:list[int]) -> tuple[list[str], list[int]]:
    '''
        Filter out unknown labels

        "-1" means no label could be matched in the product information

    Argument:
        imageID: list[str], contains the unique ID for each image
        label: list[int], contains corresponding labels of images

    Return:
        cleanImageID: list[str], filtered image ID
        cleanLabel: list[int], filtered label
    '''
    result=list(zip(imageID, label))
    return [item[0] for item in result if item[1]!=-1], [item[1] for item in result if item[1]!=-1]

def grayImage(image:Image) -> Image:
    '''
        Convert the image to grayscale if it is not

        If it's already grayscale, simply return it
    
    Argument:
        image: PIL.Image, the image with multiple channels

    Return:
        result: PIL.Image, the image in grayscale 
    '''
    if image.mode!="L":
        return ImageOps.grayscale(image)
    else:
        return image

def cropImage(image:Image, targetWidth:int, targetHeight:int) -> Image:
    '''
        Crop the central part of the image with the target dimension

        This function assumes that the input image is larger than the
    target dimension

    Argument:
        image: PIL.Image, the image in grayscale
        targetWidth: int, the width of the target dimension
        targetHeight: int, the height of the target dimension
    
    Return:
        result: PIL.Image, the central cropped image in grayscale
    '''
    width, height=image.size

    left=(width-targetWidth)/2
    top=(height-targetHeight)/2
    right=(width+targetWidth)/2
    bottom=(height+targetHeight)/2

    return image.crop((left, top, right, bottom))

def normaliseImage(image:Image, dataRange:int) -> np.ndarray:
    '''
        Normalise the image so all pixels are floats within 0~1
    and return as a numpy ndarray

    Argument:
        image: PIL.Image, the image in grayscale
        dataRange: int, the maximum range of the pixel

    Return:
        result: np.ndarray, array represents the normalised image
    '''
    result=np.array(image).astype("float32")
    return result/dataRange

def constructImageLoading(imageID:list[str], label:list[int], filePath:list[str], imageLoadingPath:str):
    '''
        Save a loading list of images which contains all file paths and their corresponding labels

    Argument:
        imageID: list[str], contains the unique ID for each image
        label: list[int], contains corresponding labels of images
        filePath: list[str], contains corresponding file path for loading the image
        imageLoadingPath: str, file path of image loading information
    '''
    imageLoading=pd.DataFrame({
        "id":imageID,
        "label":label,
        "path":filePath
    })

    imageLoading.to_csv(imageLoadingPath, index=False)