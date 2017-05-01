from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import os
import numpy as np
from skimage import io
from skimage.transform import resize
from keras.callbacks import ModelCheckpoint
from PIL import Image
import tensorflow as tf

class Segment(object):

    path = ""

    def __init__(self, data_dir):
        """
            data_directory : path like /home/rajat/nnproj/dataset/
                             includes the dataset folder with '/'
            Initialize all your variables here
        """
        self.img_size = 128
        self.path = data_dir
        inputs = keras.layers.Input((self.img_size, self.img_size, 3))
        c_layer_1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        c_layer_1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c_layer_1)
        p_layer_1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(c_layer_1)
        c_layer_2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p_layer_1)
        c_layer_2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c_layer_2)
        p_layer_2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(c_layer_2)
        c_layer_3 = keras.layers.Conv2D(96, (3, 3), activation='relu', padding='same')(p_layer_2)
        c_layer_3 = keras.layers.Conv2D(96, (3, 3), activation='relu', padding='same')(c_layer_3)
        p_layer_3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(c_layer_3)
        c_layer_4 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p_layer_3)
        c_layer_4 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c_layer_4)
        p_layer_4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(c_layer_4)
        c_layer_5 = keras.layers.Conv2D(160, (3, 3), activation='relu', padding='same')(p_layer_4)
        c_layer_5 = keras.layers.Conv2D(160, (3, 3), activation='relu', padding='same')(c_layer_5)
        u_layer_6 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(c_layer_5), c_layer_4], axis=3)
        d_layer_6 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u_layer_6)
        d_layer_6 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(d_layer_6)
        u_layer_7 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(d_layer_6), c_layer_3], axis=3)
        d_layer_7 = keras.layers.Conv2D(96, (3, 3), activation='relu', padding='same')(u_layer_7)
        d_layer_7 = keras.layers.Conv2D(96, (3, 3), activation='relu', padding='same')(d_layer_7)
        u_layer_8 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(d_layer_7), c_layer_2], axis=3)
        d_layer_8 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u_layer_8)
        d_layer_8 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(d_layer_8)
        u_layer_9 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(d_layer_8), c_layer_1], axis=3)
        d_layer_9 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u_layer_9)
        d_layer_9 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(d_layer_9)
        u_layer_10 = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(d_layer_9)

        self.model = keras.models.Model(inputs=[inputs], outputs=[u_layer_10])
        self.model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss=self.loss_func, metrics=[self.metric_func])

    def metric_func(self,val_true, val_pred):
        val_t = keras.backend.flatten(val_true)
        val_p = keras.backend.flatten(val_pred)
        acc = keras.backend.sum(val_t * val_p)
        return (acc/(keras.backend.sum(val_t) + keras.backend.sum(val_p) - acc))

    def loss_func(self,true_val, pred_val):
        return keras.losses.binary_crossentropy(y_true=keras.backend.flatten(true_val), y_pred=keras.backend.flatten(pred_val))

    def preprocess_img(self, imgs, flag=True):
        if flag:
            imgs_p = np.ndarray((imgs.shape[0], self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            imgs_p = np.ndarray((imgs.shape[0], self.img_size, self.img_size), dtype=np.uint8)

        for i in range(imgs.shape[0]):
            if flag:
                imgs_p[i] = resize(imgs[i], (self.img_size, self.img_size, 3), preserve_range=True, mode='constant')
            else:
                imgs_p[i] = resize(imgs[i], (self.img_size, self.img_size), preserve_range=True, mode='constant')
        imgs_p = imgs_p[..., np.newaxis]
        return imgs_p

    def load_images_and_masks(self):
        image_files = []
        for i in os.listdir('traindata'):
            image_files.append(i)
        input_images = np.ndarray((len(image_files), self.img_size, self.img_size, 3), dtype=np.uint8)
        mask_images = np.ndarray((len(image_files), self.img_size, self.img_size), dtype=np.uint8)
        # input_images = np.ndarray((164, self.img_size, self.img_size, 3), dtype=np.uint8)
        # mask_images = np.ndarray((164, self.img_size, self.img_size), dtype=np.uint8)
        image_files.sort()
        files_masks = [image_files[i] for i in range(1, len(image_files), 2)]
        files_images = [image_files[i] for i in range(0, len(image_files), 2)]
        for i in range(len(files_images)):
            temp = io.imread('./traindata/'+files_images[i])
            input_images[i] = np.asarray([temp])
            mask_images[i] = np.array([io.imread('./traindata/'+files_masks[i], as_grey=True)])
        return input_images, mask_images

    def train(self):
        """
            Trains the model on data given in path/train.csv
            No return expected
        """
        imgs, imgs_mask = self.load_images_and_masks()
        imgs_mask = self.preprocess_img(imgs_mask, False)
        imgs = imgs.astype('float32')
        imgs -= np.mean(imgs)
        imgs /= np.std(imgs)
        imgs_mask = imgs_mask.astype('float32')
        imgs_mask /= 255
        model_checkpoint = ModelCheckpoint('avyav_weights.h5', monitor='loss', save_best_only=True)
        self.model.fit(imgs, imgs_mask, verbose=1, shuffle=True,epochs=20,validation_split=0.1, callbacks=[model_checkpoint])

    def compute_prediction(self,img):
        height = np.int(np.ceil(np.shape(img)[0]/128))
        width = np.int(np.ceil(np.shape(img)[1]/128))
        mat_with_pad = np.zeros([height*128, width*128, 3])
        # mat_with_pad_t = np.zeros([1,np.shape(img)[1],3])
        # x = img
        # for i in range(height*128-np.shape(img)[0]):
        #     x = np.vstack((x,mat_with_pad_t))
        # mat_with_pad_t = np.zeros([height*128,1,3])
        # for i in range(width*128-np.shape(img)[1]):
        #     x = np.hstack((x,mat_with_pad_t))
        # mat_with_pad = x
        mat_with_pad[0:np.shape(img)[0], 0:np.shape(img)[1], :] = img[:,:,:]
        image_set = np.ndarray([height*width, 128, 128, 3])
        index=0
        for i in range(height):
            for j in range(width):
                image_set[index,:,:,:] = mat_with_pad[i*128:(i+1)*128,j*128:(j+1)*128,:]
                index+=1

        final_image = np.zeros([height*128, width*128])
        index=0
        for i in range(height):
            for j in range(width):
                temp = self.predict_this_image(image_set[index])
                final_image[i*128:(i+1)*128,j*128:(j+1)*128] = temp
                index+=1

        return final_image[0:np.shape(img)[0], 0:np.shape(img)[1]].astype(np.uint8)

    def predict_this_image(self,img):
        imgs_t = np.array(np.resize(img,(1,128,128,3)))
        imgs_t = imgs_t.astype('float32')
        imgs_t -= np.mean(imgs_t)
        imgs_t /= np.std(imgs_t)
        np.resize(imgs_t, (1,128,128,3))
        img_mask_test = self.model.predict(imgs_t, verbose=0)
        image_t = (img_mask_test[0][:,:,0] * 255.0).astype(np.uint8)
        image_t[image_t < 128] = 0
        image_t[image_t >= 128] = 1
        return image_t

    def get_mask(self, image):
        """
            image : a variable resolution RGB image in the form of a numpy array
            return: A list of lists with the same 2d size as of the input image with either 0 or 1 as each entry
        """
        return self.compute_prediction(image)

    def save_model(self, **params):
        """
            saves model on the disk
            no return expected
        """
        self.model.save(params['name'])
        # file_name = params['name']
        # pickle.dump(self, gzip.open(file_name, 'wb'))

    @staticmethod
    def load_model(**params):
        """
            returns a pre-trained instance of Segment class
        """
        seg_class=Segment('traindata')
        seg_class.model.load_weights('avyav_weights.h5')
        return seg_class
        # file_name = params['name']
        # return pickle.load(gzip.open(file_name, 'rb'))

# if __name__ == "__main__":
#     # pass
#     obj = Segment('traindata')
#     # obj.train()
#     obj = Segment.load_model(name='avyav_weights.h5')
#     temp = io.imread('./validationdata/valid-4.jpg')
#     temp = np.asarray([temp]).astype('float32')
#     t = obj.get_mask(temp[0])
#     m=Image.fromarray(t)
#     m.save('predvf.jpg')
#     # obj.train()
#     # obj.save_model(name="segment.gz")
