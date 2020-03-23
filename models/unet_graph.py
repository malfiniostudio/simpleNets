import tensorflow as tf 
from typing import List, Tuple
import os
import time
import pandas as pd
import numpy as np
import cv2
from pipeline.filelist import Reader
from layers.deformablelayer import DeformableConvLayer
H=int(640/4)
W=int(960/4)
"""
    This model comes from https://arxiv.org/pdf/1505.04597.pdf , if this is not the original paper then open an issue to say which one is.
"""

# We will set the numbers of layers only and the first numbers of filters.

#TODO :  put it in utils


def IOU(y_pred, y_true):
    """Returns a (approx) IOU score

    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7

    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)

    Returns:
        float: IOU score
    """
    H, W, _ = y_pred.get_shape().as_list()[1:]

    pred_flat = tf.reshape(y_pred, [-1, H * W])
    true_flat = tf.reshape(y_true, [-1, H * W])

    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    denominator = tf.reduce_sum(
        pred_flat, axis=1) + tf.reduce_sum(
            true_flat, axis=1) + 1e-7

    return tf.reduce_mean(intersection / denominator)



def simpleConv2d(inputs : tf.Tensor,filters : (int)=8,kernel_size : Tuple[int,int]=(3,3),strides: (int)=2 ,padding : (str)="SAME",name : (str)="default_name",batch_norm : (bool)=True,deformable: (bool)=False,features : (bool)=False) -> tf.Tensor:
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
    if deformable:
        x = DeformableConvLayer(filters = filters, kernel_size = kernel_size, padding = padding,strides=strides,name=name+"_conv")(inputs)
    else:
        x = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, padding = padding,strides=strides,name=name+"_conv")(inputs)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Activation('relu')(x)
    if features:
        length=x.get_shape()[-1]
        b=x.get_shape()[0]
        h=x.get_shape()[1]
        w=x.get_shape()[2]
        for m in range(length):
            tf.compat.v1.summary.image("Feature_"+name+"_{}".format(m),tf.compat.v1.image.resize(tf.reshape(x[:,:,:,m],[b,h,w,1]),(H,W)))
    return x

# TODO : maybe need a mother class for all model to do argument parsing, etc...

class UnetStruct():

    def __init__(self,n_layer : int=3,n_filter : int=8,kernel_size : Tuple[int,int]=(3,3),batch_norm : bool=True,dropout : bool=False,deformable : bool=True):
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
                self.down_convs+=[{"filters" : n_filter*2**(int(n/2)-1),"kernel_size":kernel_size,"strides": 2 ,"padding" : "SAME","name" : "down_{}".format(n),"batch_norm" : batch_norm,"deformable":False}]
            else:
                self.down_convs+=[{"filters" : n_filter*2**(int(n/2)),"kernel_size":kernel_size,"strides": 1,"padding" : "SAME","name" : "down_{}".format(n) ,"batch_norm" : batch_norm,"deformable":False}]

        #middle

        self.middle_conv = {"filters" : n_filter*2**(n_layer),"kernel_size":kernel_size,"strides": 1 ,"padding" : "SAME","name" : "middle","batch_norm" : batch_norm,"deformable":False,"features":True}

        #up way
        self.up_convs=[]
        for n in range(2*n_layer,0,-1):
            if n%2==0:
                self.up_convs+=[{"filters" : n_filter*2**(int(n/2)-1),"kernel_size":kernel_size,"strides": 1 ,"padding" : "SAME","name" : "up_{}".format(n),"deformable":False}]
            else:
                self.up_convs+=[{"filters" : n_filter*2**(int(n/2)),"kernel_size":kernel_size,"strides": 1,"padding" : "SAME","name" : "up_{}".format(n),"deformable":False}]
            
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
        z=tf.keras.layers.Activation('sigmoid')(z)
        self.y_pred=z

    def _createUnetLoss(self,y_true,loss : (str)="LSQ"):
        if loss=="LSQ":
            self.loss = tf.reduce_sum(tf.pow(tf.abs(self.y_pred-y_true),2))
        elif loss=="IOU":
            self.loss= -IOU(self.y_pred,y_true)

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
        X = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, heigth, width, channels_in], name="X")
        Y = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, heigth, width, 1], name="y")
        self._createBody(X)

        #Creation of train_op
        self._createUnetLoss(Y,"IOU")
        self._createOptimizer("ADAM")
        self._createTrain()

        #Creation of summary_op

        tf.compat.v1.summary.image("Predicted", self.y_pred)
        tf.summary.scalar("Loss", self.loss)
        tf.compat.v1.summary.image("True",Y)
        summary_op = tf.compat.v1.summary.merge_all()

        #Batch setting
        reader=Reader(train_path,custom_transformer=("Image",([H,H],[W,W],[3,1],["NO","AFF:0:1"])))
        
        
        with tf.compat.v1.Session() as sess:
            train_summary_writer = tf.summary.FileWriter(train_logdir, sess.graph)
            #test_summary_writer = tf.summary.FileWriter(test_logdir)
            #Initialisation
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
            #Reload or Saving Setting
            #saver = tf.train.Saver()
            #if os.path.exists(ckdir) and tf.train.checkpoint_exists(ckdir):
                #latest_check_point = tf.train.latest_checkpoint(ckdir)
                #saver.restore(sess, latest_check_point)
            #else:
                #latest_check_point=None
                #try:
                    #os.rmdir(ckdir)
                #except FileNotFoundError:
                    #pass
                #os.mkdir(ckdir)

            #if latest_check_point is not None:
                #saver.restore(sess, latest_check_point)
            
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
                #saver.save(sess, "{}/model.ckpt".format(ckdir))

            
                #saver.save(sess, "{}/model.ckpt".format(ckdir))



        




        

    
            


