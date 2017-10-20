# Ethayananth, Monica Rani
# 1001-417-942
#2016-12-3
import numpy as np
import os
from scipy import misc
from keras.models import  Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import RMSprop
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from itertools import product
#loading the input data
def load_data(path):
    Images = []
    for i,j,k in os.walk(path):
        for l in k:
            Images.append(misc.imread(os.path.join(i,l)).reshape(-1)/255.0)
    Images = np.array(Images)
    Images = shuffle(Images)
    return Images

#hidden layer and their function
def Model(hidden_layer):
    model = Sequential()
    model.add(Dense(hidden_layer, input_shape=(784,)))
    model.add(Activation("relu"))
    model.add(Dense(784))
    model.add(Activation("linear"))
    return model
#plot for task 1
def task1(**kwargs):
    model = Model(hidden_layer=kwargs["hidden_layer"])
    model.compile(loss="mse",optimizer=RMSprop())
    history = model.fit(kwargs["first_dataset"],kwargs["first_dataset"],verbose=2,nb_epoch=kwargs["epoch"],batch_size=kwargs["batch_size"],validation_data=(kwargs["second_dataset"],kwargs["second_dataset"]))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.xlabel("Number of epochs")
    plt.ylabel("mse")
    plt.savefig("task1.png")
    plt.show()

def plot_weights(layer,figsize=(6,6),name="task4.png"):
    shape = layer.shape
    nr = np.ceil(np.sqrt(shape[0])).astype(int)
    nc = nr
    fig = plt.figure()
    dim = np.sqrt(shape[1])
    for i in range(1,nr+1):
        for j in range(nc):
            pplot = fig.add_subplot(10,10,10*j+i)
            pplot.matshow(layer[i+j,:].reshape(dim,dim),cmap=plt.cm.binary)
            plt.xticks([])
            plt.yticks([])
    plt.savefig(name)
def task2(**kwargs):
    model =  Model(hidden_layer=kwargs["hidden_layer"])
    model.compile(loss="mse",optimizer=RMSprop())
    return model.fit(kwargs["first_dataset"],kwargs["first_dataset"],verbose=2,nb_epoch=kwargs["epoch"],batch_size=kwargs["batch_size"],validation_data=(kwargs["second_dataset"],kwargs["second_dataset"]))

def task3(**kwargs):
    model = Model(hidden_layer=kwargs["hidden_layer"])
    model.compile(loss="mse",optimizer=RMSprop())
    model.fit(kwargs["first_dataset"], kwargs["first_dataset"], verbose=2, nb_epoch=kwargs["epoch"],
                 batch_size=kwargs["batch_size"], validation_data=(kwargs["second_dataset"], kwargs["second_dataset"]))
    weights = model.layers[-2].get_weights()
    return weights

def task4(**kwargs):
    model = Model(hidden_layer=kwargs["hidden_layer"])
    model.compile(loss="mse",optimizer=RMSprop())
    model.fit(kwargs["first_dataset"], kwargs["first_dataset"], verbose=2, nb_epoch=kwargs["epoch"],
                 batch_size=kwargs["batch_size"], validation_data=(kwargs["second_dataset"], kwargs["second_dataset"]))
    encoder = Sequential()
    encoder.add(Dense(100,input_shape=(784,)))
    encoder.add(Activation("relu"))
    data = encoder.predict(kwargs["second_dataset"])
    decoder = Sequential()
    decoder.add(Dense(784,input_shape=(100,)))
    decoder.add(Activation("linear"))
    img = decoder.predict(data)
    print img.shape
    return img

if __name__ == "__main__":
    task = "task3"
    if task == "task1":
        first_dataset = load_data("C:\\Users\\Monica\\Downloads\\set1_20k\\train")
        second_dataset = load_data("C:\\Users\\Monica\\Downloads\\set2_2k")
        task1(first_dataset=first_dataset,second_dataset=second_dataset,hidden_layer=100,epoch=50,batch_size=256)
    elif task == "task2":
        first_dataset = load_data("C:\\Users\\Monica\\Downloads\\set1_20k\\train")
        second_dataset = load_data("C:\\Users\\Monica\\Downloads\\set2_2k")
        training_loss = []
        val_loss = []
        for i in [20,40,60,80,100]: #diffferent number of nodes in the hidden layer

            history = task2(first_dataset=first_dataset, second_dataset=second_dataset, hidden_layer=i, epoch=50, batch_size=256)
            training_loss.append(np.mean(history.history["loss"]))
            val_loss.append(np.mean(history.history["val_loss"]))
        plt.plot(training_loss)
        plt.plot(val_loss)
        plt.savefig("task2.png")
        plt.show()
    elif task == "task3":
        first_dataset = load_data("C:\\Users\\Monica\\Downloads\\set1_20k\\train")
        second_dataset = load_data("C:\\Users\\Monica\\Downloads\\set2_2k")
        weights = task3(first_dataset=first_dataset,second_dataset=second_dataset,hidden_layer=100,epoch=100,batch_size=256)
        plot_weights(weights[0],name="task3.png")
    elif task == "task4":
        first_dataset = load_data("C:\\Users\\Monica\\Downloads\\set1_20k\\train")
        third_dataset = load_data("C:\\Users\\Monica\\Downloads\\set3_100")
        plot_weights(third_dataset,name="task4_1.png")
        outputs = task4(first_dataset=first_dataset, second_dataset=third_dataset, hidden_layer=100, epoch=100,
                    batch_size=256)

        plot_weights(outputs,name="task4_2.png")

    elif task == "task5":
        first_dataset = load_data("C:\\Users\\Monica\\Downloads\\set2_2k")
        pca1 = PCA(n_components=100)
        pca = pca1.fit(first_dataset).fit_transform(first_dataset)
        plot_weights(pca[:100],name="task5_1.png")
        outputs = task4(first_dataset=first_dataset, second_dataset=first_dataset, hidden_layer=100, epoch=10,
                        batch_size=256)
        pca2 = pca1.fit(outputs).fit_transform(outputs)
        plot_weights(outputs[:100], name="task5_2.png")
