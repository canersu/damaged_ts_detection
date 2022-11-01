import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn, reshape
# from keras.utils import to_categorical

import os


class autoEncoder:
    def __init__(self):
        self.data = []
        self.img_train = []
        self.img_val = []
        self.img_shape = ()
        
        
    def dataset(self, dsPath, imgHeight, imgWidth, imgChannel, valSize):
        self.img_shape = (imgHeight, imgWidth, imgChannel)
        ts_db = os.path.join(dsPath)
        for img in os.listdir(ts_db):
            image = Image.open(os.path.join(ts_db, img))
            image = image.resize((imgHeight,imgWidth))
            image = np.array(image)
            image = image/255.0
            self.data.append(image)

        self.data = np.array(self.data)
        print("#All images: ", len(self.data))

        self.img_train, self.img_val = train_test_split(self.data,test_size=valSize, random_state=16)
        print("Train image set shape: ", self.img_train.shape, "Validation image set shape: ", self.img_val.shape)

    def model1(self, printSummary=False):
        ae = tf.keras.models.Sequential()
        ae.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), activation='relu', input_shape=self.img_shape))
        ae.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
        ae.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        ae.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
        ae.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        ae.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        ae.add(tf.keras.layers.Flatten())
        ae.add(tf.keras.layers.Dense(1024, activation='relu'))
        ae.add(tf.keras.layers.Dense(256, activation='relu'))
        ae.add(tf.keras.layers.Dense(1024, activation='relu'))
        ae.add(tf.keras.layers.Dense(5184, activation='relu'))
        ae.add(tf.keras.layers.Reshape((9, 9, 64)))
        ae.add(tf.keras.layers.UpSampling2D(size=(2, 2)))
        ae.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), activation='relu'))
        ae.add(tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), activation='relu'))
        ae.add(tf.keras.layers.UpSampling2D(size=(2, 2)))
        ae.add(tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(3,3), activation='relu'))
        ae.add(tf.keras.layers.Conv2DTranspose(filters=3, kernel_size = (3,3), activation = 'relu'))
        if printSummary:
            ae.summary()
        return ae

    def model2(self, printSummary=False):
        ae = tf.keras.models.Sequential()
        ae.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), activation='relu', input_shape=self.img_shape))
        ae.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
        ae.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        ae.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
        ae.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        ae.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        ae.add(tf.keras.layers.Flatten())
        ae.add(tf.keras.layers.Dense(1024, activation='relu'))
        ae.add(tf.keras.layers.Dense(256, activation='relu'))
        ae.add(tf.keras.layers.Dense(1024, activation='relu'))
        ae.add(tf.keras.layers.Dense(5184, activation='relu'))
        ae.add(tf.keras.layers.Reshape((9, 9, 64)))
        ae.add(tf.keras.layers.UpSampling2D(size=(2, 2)))
        ae.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), activation='relu'))
        ae.add(tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), activation='relu'))
        ae.add(tf.keras.layers.UpSampling2D(size=(2, 2)))
        ae.add(tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(3,3), activation='relu'))
        ae.add(tf.keras.layers.Conv2DTranspose(filters=3, kernel_size = (3,3), activation = 'relu'))
        if printSummary:
            ae.summary()
        return ae
        
    def SSIMLoss(self, y_true, y_pred):
        return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred,1.0))
       
    
    def MSEComparison(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred)))
    
    def RMSEComparison(self, y_true, y_pred):
        #return tf.keras.metrics.RootMeanSquaredError(y_true, y_pred)
        return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred))))

    def train(self, model, lr, lossFunc, batchSize, epochs):
        self.ae_model = model
        optimizer = tf.keras.optimizers.Adam(lr = lr)
        self.ae_model.compile(loss=lossFunc, optimizer=optimizer, metrics=['accuracy'])
        history = model.fit(self.img_train, self.img_train, batch_size=batchSize, epochs=epochs, validation_data=(self.img_val, self.img_val))
        
    def saveModel(self, dst_dir, dsName, modelName, lossType, dsType):
        self.ae_model.save(dst_dir+dsName+dsType+modelName+lossType+".h5")
        
    def loadModel(self, src_file):
        model = tf.keras.models.load_model(src_file)
        return model
    
    def compMetric(self, img1, img2, method):
        if method == "SSIM":
            return 1 - float(tf.image.ssim(img1, img2, 1.0))
        elif method == "PSNR":
            return float(tf.image.psnr(img1, img2, max_val=255))
        elif method == "MSE":
            m = tf.keras.metrics.MeanSquaredError()
        elif method == "RMSE":
            m = tf.keras.metrics.RootMeanSquaredError()
        elif method == "MRE":
            m = tf.keras.metrics.MeanRelativeError(normalizer=img1)
        else:
            print("Invalid Metric !!")
            return 0
            
        m.update_state(img1, img2)
        return m.result().numpy()
    
    
    def showResults(self, n_images, test_path, model, comp_metric):
    
        test_data = []
        label = ""
        test_ts = os.path.join(test_path)
        for img in os.listdir(test_ts):
            image = Image.open(os.path.join(test_ts, img)).convert('RGB')
            image = image.resize((48,48))
            image = np.array(image)
            image = image/255.0
            test_data.append(image)

        test_data = np.array(test_data)            
        gen = model.predict(test_data)
        
        tensor_test = tf.convert_to_tensor(test_data, dtype=tf.float32)
        plt.figure(figsize=(20, 14), dpi=100)
        plt.subplots_adjust(wspace=0.1, hspace=0.5)
        plt_a=1
        for i in range(n_images):
            # Original training dataset vs Original training
            ax = plt.subplot(3, n_images, plt_a   )
            plt.imshow(test_data[i].reshape(48,48,3))
            ax.get_xaxis().set_visible(True)
            ax.get_yaxis().set_visible(False)
            ax.set_title("Original Image")

            # Reconstructed good data  vs Original training data
            ax = plt.subplot(3, n_images, plt_a + n_images )
            plt.imshow(gen[i].reshape(48,48,3))
            ax.get_xaxis().set_visible(True)
            ax.get_yaxis().set_visible(False)    
            # value_a = self.MSEComparison(tensor_test[i], gen[i]) # self.SSIMLoss(tensor_test[i], gen[i])
            ax.set_title("Reconstructed Image")
            if comp_metric == "ALL":
                label = ""
                methods = ["SSIM", "PSNR", "MSE", "RMSE", "MRE"]
                values = []
                for j in range(len(methods)):
                    value = self.compMetric(tensor_test[i], gen[i], methods[j])
                    # values.append(value)
                    label += (methods[j] + ' Loss value: ' + str(round(value,3)) + '\n')
                ax.set_xlabel(label)
            else:
                value_a = self.compMetric(tensor_test[i], gen[i], comp_metric)
                label = comp_metric + ' Loss value: {:.3f}'
                ax.set_xlabel(label.format(value_a))

            plt_a+=1
        plt.show()
        
        
    def __del__(self):
        print('Destructor called, Employee deleted.')