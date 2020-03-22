import tensorflow as tf 
from typing import List, Tuple
import os
import time
import pandas as pd
import numpy as np
import cv2
H=640
W=960
"""
    This model comes from https://arxiv.org/pdf/1505.04597.pdf , if this is not the original paper then open an issue to say which one is.
"""

# We will set the numbers of layers only and the first numbers of filters.

#TODO :  put it in utils

class Reader:

    def __init__(self,path,batch_size=4,initial_step=0):
        self.step=initial_step
        self.reader=pd.read_csv(path,header=None,chunksize=batch_size)

    def next_batch(self,h=H,w=W):
        for chunk in self.reader:
            return get_image_mask(chunk)


def get_image_mask(frame_path,h=H,w=W):
    """Returns `image` and `mask`

    Input pipeline:
        Queue -> CSV -> FileRead -> Decode JPEG

    (1) Queue contains a CSV filename
    (2) Text Reader opens the CSV
        CSV file contains two columns
        ["path/to/image.jpg", "path/to/mask.jpg"]
    (3) File Reader opens both files
    (4) Decode JPEG to tensors

    Notes:
        height, width = 640, 960

    Returns
        image (3-D Tensor): (640, 960, 3)
        mask (3-D Tensor): (640, 960, 1)
    """
    images_p=[]
    masks_t=[]
    for z in range(len(frame_path)):
        i=cv2.imread(frame_path[0][z])
        m=cv2.imread(frame_path[1][z],cv2.IMREAD_GRAYSCALE)
        i=cv2.resize(i,(w,h))
        m=cv2.resize(m,(w,h))
        i=np.array(i,dtype=np.float32)
        m=np.array(m,dtype=np.float32)
        images_p+=[i]
        masks_t+=[m]
    masks_p=[ mask / (np.max(mask) + 1e-7) for mask in masks_t]
    images=np.array(images_p)
    masks=np.array(masks_p)
    masks=np.expand_dims(masks,-1)
    return images, masks

def simpleConv2d(inputs : tf.Tensor,filters : (int)=8,kernel_size : Tuple[int,int]=(3,3),strides: (int)=2 ,padding : (str)="SAME",name : (str)="default_name",batch_norm : (bool)=True) -> tf.Tensor:
    """Wrapper to get batchnorm and activation inside

        Args:
            inputs (Tensor)         : the input tensor with channel last (N,H,W,C)
            filters (int)           : number of filters of the convolution.
            kernel_size (int,int)   : size of the kernels.
            strides (int)           : stride of the convolution.
            padding (str)           : padding argument convolution.
            name (str)              : the basis name for the layers inside.
            batch_norm (bool)       : if we want to do batch normalization.

        Returns:
            Tensor depending of padding and kernel_size with filters numbers of channels.
    """
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, padding = padding,strides=strides,name=name+"_conv")(inputs)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    return x

# TODO : maybe need a mother class for all model to do argument parsing, etc...

class UnetStruct():

    def __init__(self,n_layer : int=3,n_filter : int=8,kernel_size : Tuple[int,int]=(3,3),batch_norm : bool=True,dropout : bool=False):
        """Initiate the UnetStruct Object.

        Args:
            n_layer (int)           : depth of the unet.
            n_filter (int)          : number of filters of the first layer.
            kernel_size (int,int)   : size of the kernels (for all the layer).
            batch_norm              : if we want to do batch normalization at each layer.
            dropout                 : if we want to do dropout.

        Returns:
            None

        Examples:
            >>> n_layer = 5
            >>> n_filter = 2
            >>> kernel_size = (2,2)
            >>> unet = UnetStruct(n_layer, n_filter , kernel_size)

        Notes:
            1. n_layer is not the number of layer, but the number of layers depend of it.
            2. We don't do maxpooling but stride convolution.
        """

        #we design all the stride,and number of filters for each convolutionnal layer down and up

        # TODO : Add some parameter like use_bias, activation function or batch normalization.
        #down way
        self.down_convs=[]
        for n in range(1,2*n_layer+1):
            if n%2==0:
                self.down_convs+=[{"filters" : n_filter*2**(int(n/2)-1),"kernel_size":kernel_size,"strides": 2 ,"padding" : "SAME","name" : "down_{}".format(n),"batch_norm" : batch_norm}]
            else:
                self.down_convs+=[{"filters" : n_filter*2**(int(n/2)),"kernel_size":kernel_size,"strides": 1,"padding" : "SAME","name" : "down_{}".format(n) ,"batch_norm" : batch_norm}]

        #middle

        self.middle_conv = {"filters" : n_filter*2**(n_layer),"kernel_size":kernel_size,"strides": 1 ,"padding" : "SAME","name" : "middle","batch_norm" : batch_norm}

        #up way
        self.up_convs=[]
        for n in range(2*n_layer,0,-1):
            if n%2==0:
                self.up_convs+=[{"filters" : n_filter*2**(int(n/2)-1),"kernel_size":kernel_size,"strides": 1 ,"padding" : "SAME","name" : "up_{}".format(n)}]
            else:
                self.up_convs+=[{"filters" : n_filter*2**(int(n/2)),"kernel_size":kernel_size,"strides": 1,"padding" : "SAME","name" : "up_{}".format(n)}]
            
        self.n_filter=n_filter
        self.n_layer=n_layer
        self.kernel_size=kernel_size

    def _createBody(self,inputs):

        #down (encoder) 
        self.down_value=[]
        z=inputs
        for (k,x) in enumerate(self.down_convs):
            z=simpleConv2d(z,**x)
            if k%2==0:
                #k=0,2 etc.
                self.down_value+=[z]
        
        #middle

        z = simpleConv2d(z,**self.middle_conv)
        #up (decoder)

        for (k,x) in enumerate(self.up_convs):
            if k%2==0:
                z=tf.keras.layers.Conv2DTranspose(filters=self.n_filter*2**(int((2*self.n_layer-k)/2)-1), kernel_size=self.kernel_size, strides=(2, 2), padding='same')(z)
                z=tf.keras.layers.concatenate([z,self.down_value[self.n_layer-int(k/2)-1]])
            z=simpleConv2d(z,**x)
            z=simpleConv2d(z,**x)
        
        z=simpleConv2d(z,1,strides=1,name="last")
        self.y_pred=z

    def _createUnetLoss(self,y_true,loss : (str)="LSQ"):
        if loss=="LSQ":
            self.loss = tf.reduce_sum(tf.pow(tf.abs(self.y_pred-y_true),2))

    def _createOptimizer(self,method,learning_rate=0.001):
        if method=="ADAM":
            self.optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate)
        elif method=="SGD":
            self.optimizer=tf.train.GradientDescentOptimizer(learning_rate)

    def _createTrain(self):
        """Returns a training operation

        Loss function defined in createUnetLoss

        IOU is

            (the area of intersection)
            --------------------------
            (the area of two boxes)

        Args:
            y_pred (4-D Tensor): (N, H, W, 1)
            y_true (4-D Tensor): (N, H, W, 1)

        Returns:
            train_op: minimize operation
        """

        global_step = tf.compat.v1.train.get_or_create_global_step()

        self.train_op=self.optimizer.minimize(self.loss, global_step=global_step)

    def fit(self,train_path,test_path="",heigth=H,width=W,channels_in=3,epochs=2,batch_size=4,logdir="logdir",ckdir="ckdir"):
        """Fit the model with X inputs and Y the true value

        Args:
            X (path)    : path to csv containing the path to the input image.
            Y (path)    : path to csv containing the true image.
            epochs      : number of epochs
            batch_size  : the size of the batch
            logdir      : the path to the logdirectory for tensorboard
            ckdir       : the path to the checkpoint directory for save and restore

        """
        #Reading of data
        train = pd.read_csv(train_path)
        n_train = train.shape[0]

        if test_path!="":
            test = pd.read_csv(test_path)
            n_test = test.shape[0]

        #Creation of logdir
        current_time = time.strftime("%m/%d/%H/%M/%S")
        train_logdir = os.path.join(logdir, "train", current_time)
        test_logdir = os.path.join(logdir, "test", current_time)

        

        

        #Creation of model body
        tf.compat.v1.reset_default_graph()
        X = tf.compat.v1.placeholder(tf.float32, shape=[None, heigth, width, channels_in], name="X")
        Y = tf.compat.v1.placeholder(tf.float32, shape=[None, heigth, width, 1], name="y")
        self._createBody(X)

        #Creation of train_op
        self._createUnetLoss(Y)
        self._createOptimizer("ADAM")
        self._createTrain()

        #Creation of summary_op

        tf.compat.v1.summary.image("Predicted", self.y_pred)
        tf.summary.scalar("Loss", self.loss)
        tf.compat.v1.summary.image("True",X)
        summary_op = tf.compat.v1.summary.merge_all()

        #Batch setting
        reader=Reader(train_path)
        
        #train_csv=tf.data.Dataset.from_tensor_slices([train_path])
        #train_image, train_mask = get_image_mask(train_csv,heigth,width)
        #X_op, y_op = tf.train.batch(
        #[train_image, train_mask],
        #batch_size=4,
        #capacity=4 * 2,
        #allow_smaller_final_batch=True)
        #train_csv = tf.train.string_input_producer(['train.csv'])
        #test_csv = tf.train.string_input_producer(['test.csv'])
        with tf.compat.v1.Session() as sess:
            train_summary_writer = tf.summary.FileWriter(train_logdir, sess.graph)
            #test_summary_writer = tf.summary.FileWriter(test_logdir)
            #Initialisation
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
            #Reload or Saving Setting
            saver = tf.train.Saver()
            if os.path.exists(ckdir) and tf.train.checkpoint_exists(ckdir):
                latest_check_point = tf.train.latest_checkpoint(ckdir)
                #saver.restore(sess, latest_check_point)
            else:
                latest_check_point=None
                try:
                    os.rmdir(ckdir)
                except FileNotFoundError:
                    pass
                os.mkdir(ckdir)

            if latest_check_point is not None:
                saver.restore(sess, latest_check_point)
            try:
                global_step = tf.train.get_global_step(sess.graph)

                #coord = tf.train.Coordinator()
                #threads = tf.train.start_queue_runners(coord=coord)

                for epoch in range(epochs):
                    for step in range(0,n_train,batch_size):
                        a,b=reader.next_batch()
                        print(a.shape)
                        print(b.shape)
                        _, step_loss, step_summary, global_step_value = sess.run([self.train_op,self.loss,summary_op,global_step],feed_dict={X: a,Y: b})
                        train_summary_writer.add_summary(step_summary,global_step_value)
                saver.save(sess, "{}/model.ckpt".format(ckdir))

            finally:
                saver.save(sess, "{}/model.ckpt".format(ckdir))



        




        

    
            


