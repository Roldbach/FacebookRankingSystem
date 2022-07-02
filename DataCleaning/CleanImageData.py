import numpy as np
import os

from Configuration import dataRange, imageFolder, targetFolder, targetSize
from PIL import Image

def cleanImage(imageFolder:str=imageFolder, targetFolder:str=targetFolder, targetSize:int=targetSize, dataRange:int=dataRange):
    '''
        Clean every image in the given folder and store it
    to another place for further training

        Each image is cleaned by:
        1. Resize to a square with the target length
        3. Rename the file using the index and save it as jpg file

    Argument:
        imageFolder: string, the folder where all images are stored
        targetFolder: string, the folder where all cleaned images are stored
        targetSize: int, the uniform dimension of the square image for training
        dataRange: int, the maximum range of the pixel
    '''
    imageList=os.listdir(imageFolder)

    for i in range(len(imageList)):
        image=Image.open(imageFolder+"/"+imageList[i])
        image=resizeImage(image, targetSize)
        image=normaliseImage(image, dataRange)

        image.save(targetFolder+"/"+str(i)+".jpg")

def resizeImage(image:Image, targetSize:int) -> Image:
    '''
        Resize the given image to the target size

        This function could crop the image into a square
    with the target size as length if the input image is
    a rectangle
    
    Argument:
        image: PIL.Image, the image in RGB channels
        targetSize: int, the uniform dimension of the square image for training
    
    Return:
        result: PIL.Image, the resized image in RGB channels
    '''
    currentSize=image.size
    scaleFactor=float(targetSize)/max(currentSize)
    newSize=tuple([int(dimension*scaleFactor) for dimension in currentSize])

    image=image.resize(newSize, Image.ANTIALIAS)
    result=Image.new("RGB", (targetSize, targetSize))
    result.paste(image, ((targetSize-newSize[0])//2, (targetSize-newSize[1])//2,))

    return result

def normaliseImage(image:Image, dataRange:int) -> Image:
    '''
        Normalise the image so all pixels are floats within 0~1 

        This function could only accepts RGB images as input

    Argument:
        image: PIL.Image, the image in RGB channels
        dataRange: int, the maximum range of the pixel

    Return:
        result: PIL.Image, the normalised image in RGB channels
    '''
    imageArray=np.array(image)
    imageArray=imageArray/dataRange

    result=Image.fromarray(imageArray.astype("float32"), "RGB")
    return result