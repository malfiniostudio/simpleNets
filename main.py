from models import unet_graph

DNN = unet_graph.UnetStruct()

DNN.fit("train.csv")