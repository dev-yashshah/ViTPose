import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential
#from tensorflow.keras.datasets import mnist
from tqdm import tqdm
from keras.layers import LeakyReLU, Dense, Dropout, Input
#from tensorflow.keras.optimizers import Adam
import json
import sys
sys.path.append('../')
import random

from common.h36m_dataset import Human36mDataset
from common.camera import world_to_camera, project_to_2d, image_coordinates
from utils.utils import wrap

def build_generator():
    #initializing the neural network
    generator= Sequential()
    #adding an input layer to the network
    generator.add(Dense(units=256, input_shape=(16, 3)))
    #activating the layer with LeakyReLU activation function
    generator.add(LeakyReLU(0.2))
    #applying batch Normalization
    generator.add(Dense(units=512))
    #adding the third layer
    generator.add(Dense(units=1024))
    generator.add(LeakyReLU(0.2))
    #the output layer with 784(28x28) nodes
    generator.add(Dense(units=1 , activation='tanh'))
    #compiling the generator network with loss and optimizer functions
    generator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5))
    return generator

def build_discriminator():
    #initializing a neural network
    discriminator=Sequential()
    #adding an input layer to the network
    discriminator.add(Dense(units=1024, input_dim=1))
    #activating the layer with leakyReLU activation function
    discriminator.add(LeakyReLU(0.2))
    #adding a dropout layer to reduce overfitting
    discriminator.add(Dropout(0.2))
    
    #adding a second layer
    discriminator.add(Dense(units=512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    
    #adding a third layer
  
    discriminator.add(Dense(units=256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
  
    #adding a forth layer
    discriminator.add(Dense(units=128, input_shape = (16,3)))
    discriminator.add(LeakyReLU(0.2))
  
    #adding the output layer with sigmoid activation
  
    discriminator.add(Dense(units=1,activation='sigmoid'))
  
    #compiling the disciminator Network with a loss and optimizer functions
  
    discriminator.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=0.0002,beta_1=0.5))
  
    return discriminator

#stacking the generator and discriminator networks to form a GAN.

def gan_net(generator, discriminator):
    #setting the trainable parameter of discriminator or false.
    discriminator.trainable=False
    #instantiates a keras tensor of shape 100 (Noise shape)
    inp=Input(shape=(16,3))
    #feeds the output from generator(X) to the discriminator and stores the results in out
    X=generator(inp)
    #feeds the output from generator (X) to the discriminator and stores the results in out
    out=discriminator(X)
    #creates a model include all layers required in the computation of out given inputs
    gan = Model(inp, out)
    #compiling the GAN Network
    gan.compile(loss='binary_crossentropy',optimizer='adam')

#method to plot the images

def plot_images(epoch, generator,dim=(10,10),figsize=(10,10)):
    #generate a normally distributed noise of shape (100x100)
    noise=np.random.normal(loc=0,scale=1,size=[100,100])
    #generate an image for the input noise
    noise=np.random.normal(loc=0,scale=1,size=[100,100])
  
    #generate an image for the input noise
  
    generated_images=generator.predict(noise)
  
    #reshape the generated image
    generated_images=generated_images.reshape(100,28,28)
  
    #plot the image
    plt.figure(figsize=figsize)
  
    #plot for each pixel
  
    for i in range(generated_images.shape[0]):
        
        plt.subplot(dim[0],dim[1],i+1)
        plt.imshow(generated_images[i],cmap='gray',interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()

#Training method with training set, default epoch and default batch_size as arguments

def train(X_train, epochs=5, batch_size=128):
    #initializing the GAN
    generator=build_generator()
    discriminator=build_discriminator()
    gan=gan_net(generator,discriminator)
    
    #training the model for specified epochs
    for epoch in range (1, epochs+1):
        print("###### @ Epoch",epoch)
        #tqdm module helps to generate a status bar for training
        for _ in tqdm(range(batch_size)):
            #random noise with size batch_size*4
            noise=np.random.normal(0, 1, [batch_size*4, 16, 3])
            #print(noise.shape)
            #generating images from noise
            generated_images=generator.predict(noise)
            #taking random images from the training
            #image_batch=X_train[np.random.randint(low=0,high=X_train.shape[0]
            #                                     ,size=batch_size)]
            #Choose random action
            subject = random.choice(list(X_train.keys()))
            #print(subject)
            #print(X_train[subject])
            action = random.choice(list(X_train[subject].keys()))

            image_batch = X_train[subject][action][:, :batch_size, :, :]
            image_batch = image_batch.reshape([batch_size*4, 16, 3])
            #creating a new training set with real and fake images
            print(image_batch.shape)
            print(generated_images.shape)   #Make them equal
            X=np.concatenate([image_batch,generated_images])
            print(X.shape)
            #labels for generated and real data
            y_dis=np.zeros((8*batch_size,16,3))
            
            #label for real images
            y_dis[:batch_size*4,:,:]=1.0
            
            #training the discrminator with real and generated images
            discriminator.trainable=True
            discriminator.train_on_batch(X,y_dis)
      
            #labelling the generated images a sreal images(1) to trick the discriminator
      
            noise=np.random.normal(0, 1, [batch_size*4, 16, 3])
            y_gen=np.ones((8*batch_size,16,3))
      
            #freezing the weights of the discriminant or while training generator
      
            discriminator.trainable=False
      
            #training the gan network
      
            gan.train_on_batch(noise,y_gen)
      
            #plotting the images for every 10 epoch
            if epoch==1 or epoch %10==0:
                
                
                plot_images(epoch,generator,dim=(10,10),figsize=(15,15))

def load_ground_truth(file):
    dataset = Human36mDataset(file + '.npz')
    output_2d_poses = {}
    output_3d_poses = {}
    for subject in dataset.subjects():
        output_2d_poses[subject] = {}
        output_3d_poses[subject] = {}
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            positions_2d = []
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                positions_3d.append(pos_3d)
                pos_2d = wrap(project_to_2d, True, pos_3d, cam['intrinsic'])
                pos_2d_pixel_space = image_coordinates(pos_2d, w=cam['res_w'], h=cam['res_h'])
                positions_2d.append(pos_2d_pixel_space.astype('float32'))
            output_2d_poses[subject][action] = np.array(positions_2d)
            output_3d_poses[subject][action] = np.array(positions_3d)
    return output_2d_poses, output_3d_poses

def main():
    gt_2d, gt_3d = load_ground_truth("../data/data_3d_h36m")
    X_train = gt_3d.copy()
    #print(X_train.items())
    #print(X_train.shape)
    train(X_train,epochs=5,batch_size=128)

if __name__ == "__main__":
    main()