"""
    Contains all helper function to read image or text listed in another file.
"""
import numpy as np
import pandas as pd 
import cv2


#TODO : Add docstring + test for this class.

class Reader:
    """Read data to make them as text or image for tensorflow"""

    def __init__(self,path,batch_size=4,initial_step=0,custom_transformer=None):
        """
        Args:
            batch_size (int)            : size of the batch
            initial_step (int)          : the step where we are in the dataset
            transformer (func,*args)    : the function we want to use to transform our data.
        """
        self.reader=pd.read_csv(path,header=None,chunksize=batch_size,index_col=False,skiprows=initial_step*batch_size)
        if custom_transformer is not None:
            if custom_transformer[0]=="Image":
                self.trans=lambda x : _get_images(x,*custom_transformer[1])
            else:
                self.trans=lambda x : custom_transformer[0](x,*custom_transformer[1])

    def next_batch(self):
        for chunk in self.reader:
            t_chunk=chunk.reset_index(drop=True)
            return self.trans(t_chunk)

def _check_size(item1,length,message="Length mismatch"):
    """Check if the size is the same and send an exception if not"""
    if item1 is None:
        return None
    else:
        if len(item1)==length:
            return True
        else:
            raise Exception(message)


#TODO : Add a way to normalize maybe?

def _get_images(chunks,heigth=None,width=None,channels=None,normalization=None):
    """Returns images from the lists.

    Args :

        heigth (list int)           : list containing the heigth for each different kind of image
        width (list int)            : list containing the width for each diffrent image
        channels (list int)         : list containing the number of channels for each diffrent image
        normalization (list str)    : list containing string representing the possible normalization.

    Note :
        If arg is None, then the original value of the image will be taken.

        for normalization AFF:a:b will do affine transform of data to make them belong to [a,b]

    Returns
        tuple of image (3-D Tensor): (heigth, width, channels)
    """
    length=len(chunks.columns)
    if _check_size(heigth,length,"Heigths length mismatch in _check_size") is None:
        heigth=[-1]*length
    if _check_size(width,length,"Widths length mismatch in _check_size") is None:
        width=[-1]*length
    if _check_size(channels,length,"Channels length mismatch in _check_size") is None:
        channels=[-1]*length
    if _check_size(normalization,length,"Channels length mismatch in _check_size") is None:
        normalization=["NO"]*length

    total_t=[]
    for j in range(len(chunks.columns)):
        temp=[]
        for i in range(len(chunks)):
            if channels[j]==1:
                image=cv2.imread(chunks[j][i],cv2.IMREAD_GRAYSCALE)
            else:
                image=cv2.imread(chunks[j][i])
            image=cv2.resize(image,(width[j],heigth[j]))
            image=np.array(image,dtype=np.float32)
            if normalization[j][:3]=="AFF":
                #affine transormation of data between [a,b]
                a=float(normalization[j].split(":")[1])
                b=float(normalization[j].split(":")[2])
                image=image/np.max(image)*(b-a)+a
            temp+=[image]
        temp=np.array(temp)
        if channels[j]==1:
            temp=np.expand_dims(temp,-1)

        total_t+=[temp]

    return tuple(total_t)
