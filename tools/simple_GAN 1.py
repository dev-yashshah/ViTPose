import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import torch.nn as nn
from keras.models import Sequential, load_model
from keras.layers import Dense, LeakyReLU, BatchNormalizationV2
from tqdm.auto import tqdm
import sys
sys.path.append('../')

from common.h36m_dataset import Human36mDataset
from common.camera import project_to_2d, image_coordinates
from utils.utils import wrap
from keypoint_visualization import kpt_visualization
import tensorflow as tf
from keras import Model
from keras.layers import Input
import json


torch.manual_seed(333)

class GAN:
    def __init__(self) -> None:
        dev = 'cuda:0' if torch.cuda.is_available() == True else 'cpu'
        self.device = torch.device(dev)
        self.generator_model = None
        self.discriminator_model = None

    def generator(self):
        gen = Sequential()
        gen.add(tf.keras.Input(shape=(128,)))

        gen.add(Dense(16))
        gen.add(BatchNormalizationV2())
        gen.add(LeakyReLU(0.2))

        gen.add(Dense(16*2))
        gen.add(BatchNormalizationV2())
        gen.add(LeakyReLU(0.2))

        gen.add(Dense(units=16*3, activation= 'tanh'))
        
        gen.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5))
        return gen
    
    def discriminator(self):
        '''
        Objective
        max { log D(x) + log (1- D(G(z))) }

        D: Discriminator
        G: Generator
        x: real image
        z: noise vector
        '''
        dis = Sequential()
        dis.add(tf.keras.Input((16 * 3,)))
        dis.add(Dense(16 * 3))
        dis.add(BatchNormalizationV2())
        dis.add(LeakyReLU(0.2))

        dis.add(Dense(16 * 4))
        dis.add(BatchNormalizationV2())
        dis.add(LeakyReLU(0.2))

        dis.add(Dense(16 * 5))
        dis.add(BatchNormalizationV2())
        dis.add(LeakyReLU(0.2))

        dis.add(Dense(1, activation = 'sigmoid'))
        dis.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.002,beta_1=0.5))
        return dis
    
    def plot_images(self, kpts, grid_size = 2):
        """
        kpts: Numpy array containing all the keypoint positions
        grid_size: 2x2 or 5x5 grid containing images
        """
        
        fig = plt.figure(figsize = (8, 8))
        columns = rows = grid_size
        plt.title("Generated Keypoints")
        kv = kpt_visualization()
        for i in range(1, columns*rows +1):
            plt.axis("off")
            img = kv.imshow_keypoints_3d(kpts[i])
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
        plt.show()

    def gan_net(self, generator, discriminator):
        #discriminator.trainable=False
        inp = Input(shape=(128,))
        X=generator(inp)
        #feeds the output from generator (X) to the discriminator and stores the results in out
        out=discriminator(X)
        #creates a model include all layers required in the computation of out given inputs
        gan = Model(inp, out)
        #compiling the GAN Network
        gan.compile(loss='binary_crossentropy',optimizer='adam')
        return gan

    def train(self, data, epochs=5, batch_size=128):
        generator = self.generator()
        discriminator = self.discriminator()
        gan = self.gan_net(generator, discriminator)
        for epoch in range (1, epochs+1):
            print("###### @ Epoch",epoch)
            for _ in tqdm(range(batch_size)):

                noise=np.random.normal(loc=0,scale=1,size=[128,128])
                generated_numbers = generator.predict(noise)
                subject = random.choice(list(data.keys()))
                action = random.choice(list(data[subject].keys()))
                
                X_batch = data[subject][action]["training"][:batch_size]
                X_batch = X_batch.reshape((batch_size, 16*3))
                #print(X_batch.shape, generated_numbers.shape)
                discriminator.trainable = True
                #Real Data
                y_dis=np.ones(batch_size)
                real_loss = discriminator.train_on_batch(X_batch, y_dis)
                
                #Fake Data
                y_dis = np.zeros(batch_size)
                fake_loss = discriminator.train_on_batch(generated_numbers, y_dis)
                #X = np.concatenate([X_batch, generated_numbers])
                #y_dis=np.zeros(2*batch_size)

                #y_dis[:batch_size]=1.0

                #discriminator.trainable=True
                #discriminator.train_on_batch(X, y_dis)
                noise=np.random.normal(0, 1, [batch_size, batch_size])
                y_gen=np.ones(batch_size)

                discriminator.trainable=False

                gan.train_on_batch(noise, y_gen)
        
        self.generator_model = gan
        self.discriminator_model = discriminator
        self.generator_model.summary()
        self.discriminator_model.summary()
        self.save()

    def save(self):
        print("Saving")
        try:
            self.generator_model.save("generator.h5")
            self.discriminator_model.save("discriminator.h5")
            print("Model saved")
        except:
            print("Model could not be saved")
    
    def load(self):
        print("Loading Model")
        try:
            self.generator_model = load_model("generator.h5")
            self.discriminator_model = load_model("discriminator.h5")
            print(self.generator_model.summary())
            print("Model Loaded")
        except:
            print("Model could not be loaded")
    
    def test(self, dataset, kpts):
        self.load()
        #print(dataset["S1"]["Directions"]["training"][0])
        #return
        #print(self.discriminator_model.predict(dataset["S9"]["Directions"]["testing"][0].reshape([1,48])))
        counter = 0
        for kpt in kpts:
            kpt = kpt["keypoints_3d"]
            kpt_reoriented = np.array(kpt[:10] + kpt[11:])
            result = self.discriminator_model.predict(kpt_reoriented.reshape([1,48]))[0][0]
            
            if result != 1:
                counter += 1
                #self.generator_model.predict(kpt)
            
        print(len(kpts), "Frame error counter: ", counter)

        self.plot_images(dataset["S1"]["Directions"]["testing"])

def load_ground_truth(file, cam_number = 4):
    dataset = Human36mDataset(file + '.npz')
    output_2d_poses = {}
    output_3d_poses = {}
    for subject in dataset.subjects():
        output_2d_poses[subject] = {}
        output_3d_poses[subject] = {}
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            if cam_number == 4:
                positions_2d = []
                positions_3d = []
                for cam in anim['cameras']:
                    #pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d = anim['positions']
                    positions_3d.append(pos_3d)
                    pos_2d = wrap(project_to_2d, True, pos_3d, cam['intrinsic'])
                    pos_2d_pixel_space = image_coordinates(pos_2d, w=cam['res_w'], h=cam['res_h'])
                    positions_2d.append(pos_2d_pixel_space.astype('float32'))
            else:
                #positions_3d = world_to_camera(anim['positions'], R=anim['cameras'][cam_number]['orientation'], t=anim['cameras'][cam_number]['translation'])
                positions_3d = anim['positions']
                pos_2d = wrap(project_to_2d, True, positions_3d, anim['cameras'][cam_number]['intrinsic'])
                pos_2d_pixel_space = image_coordinates(pos_2d, w=anim['cameras'][cam_number]['res_w'], h=anim['cameras'][cam_number]['res_h'])
                positions_2d = pos_2d_pixel_space.astype('float32')
            output_2d_poses[subject][action] = np.array(positions_2d)
            output_3d_poses[subject][action] = np.array(positions_3d)

    return output_2d_poses, output_3d_poses

def main():
    print("Extracting Ground Truth")
    gt_2d, gt_3d = load_ground_truth("../data/data_3d_h36m", cam_number = 0) # There are 4 caliberated camera use 4 if you want to use all cameras
        
    #splitting training, testing
    for subject in gt_3d.keys():
        for action in gt_3d[subject].keys():
            temp = gt_3d[subject][action]
            temp1 = gt_2d[subject][action]

            gt_3d[subject][action] = {}
            gt_3d[subject][action]["training"], gt_3d[subject][action]["testing"] = np.split(temp, [int(temp.shape[0]*0.9)]) #90% in training and rest in testing
            
            gt_2d[subject][action] = {}
            gt_2d[subject][action]["training"], gt_2d[subject][action]["testing"] = np.split(temp1, [int(temp1.shape[0]*0.9)]) #90% in training and rest in testing
    
    print("Extraction successful")
    print(gt_3d["S1"]["Directions"]["training"].shape)
    
    gan = GAN()
    gan.train(gt_3d, epochs=100)
    
def test():
    gt_2d, gt_3d = load_ground_truth("../data/data_3d_h36m", cam_number = 0) # There are 4 caliberated camera use 4 if you want to use all cameras
        
    #splitting training, testing
    for subject in gt_3d.keys():
        for action in gt_3d[subject].keys():
            temp = gt_3d[subject][action]
            temp1 = gt_2d[subject][action]

            gt_3d[subject][action] = {}
            gt_3d[subject][action]["training"], gt_3d[subject][action]["testing"] = np.split(temp, [int(temp.shape[0]*0.9)]) #90% in training and rest in testing
            
            gt_2d[subject][action] = {}
            gt_2d[subject][action]["training"], gt_2d[subject][action]["testing"] = np.split(temp1, [int(temp1.shape[0]*0.9)]) #90% in training and rest in testing
    
    kp_path = r"D:\Research\ViTPose\ViTPose\vis_results\Keypoints_1.json"
    keypoints=json.load(open(kp_path))
    gan = GAN()
    gan.test(gt_3d, keypoints)

main()
#test()