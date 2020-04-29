from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import os.path

class DataProcessor:
    def __init__(self):

        self.im_size = (64,64)
        # Validation split done during training
        self.data_split = (0.8,0.2)
        #self.data_split = (0.6, 0.8, 1) # training, validation, test

        parasitized = self.load_images('./cell_images/Parasitized')
        uninfected = self.load_images('./cell_images/Uninfected')

        parasitized_length = parasitized.shape[0]
        uninfected_length = uninfected.shape[0]

        shuffle_order = [i for i in range(parasitized_length)]
        np.random.shuffle(shuffle_order)
        parasitized = parasitized[shuffle_order,]
        shuffle_order = [i for i in range(uninfected_length)]
        np.random.shuffle(shuffle_order)
        uninfected = uninfected[shuffle_order,]

        length = parasitized_length+uninfected_length
        data = np.append(parasitized, uninfected, axis=0)
        data = data/255
        labels = np.zeros([length, 2])
        labels[0:parasitized_length, 0] = 1
        labels[parasitized_length:,1] = 1

        shuffle_order = [i for i in range(length)]
        np.random.shuffle(shuffle_order)
        data = data[shuffle_order,]
        labels = labels[shuffle_order,]

        data_training = data[0:int(self.data_split[0]*length),]
        labels_training = labels[0:int(self.data_split[0]*length),]

        #data_validation = data[int(self.data_split[0]*length)+1:int(self.data_split[1]*length),]
        #labels_validation = labels[int(self.data_split[0]*length)+1:int(self.data_split[1]*length),]

        data_test = data[int(self.data_split[0]*length):,]
        labels_test = labels[int(self.data_split[0]*length):,]

        np.save('data_set/x_train.npy', data_training)
        np.save('data_set/y_train.npy', labels_training)

        #np.save('data_set/x_val.npy', data_validation)
        #np.save('data_set/y_val.npy', labels_validation)

        np.save('data_set/x_test.npy', data_test)
        np.save('data_set/y_test.npy', labels_test)



    def pad_and_resize_image(self, image):
        image_size = image.size
        max_size = np.max(image_size)
        new_im = Image.new('RGB', (max_size, max_size))
        new_im.paste(image, ( int((max_size - image_size[0])/2), int((max_size - image_size[1])/2)))

        return new_im.resize(self.im_size, Image.ANTIALIAS)


    def load_images(self, directory):
        counter = 0

        num_files = len([name for name in os.listdir(directory)])
        print('Number of files: ' + str(num_files))
        images = np.zeros((num_files,) + self.im_size + (3,))
        for filename in os.listdir(directory):
            if filename.endswith('.png'):
                im = Image.open(directory+'/'+filename)
                im_np = np.asarray(self.pad_and_resize_image(im))
                images[counter,] = im_np.reshape((1,) + im_np.shape)
                counter += 1
                if counter % 500 == 0:
                    print('Loaded images = ' + str(counter))

                #if counter == 3000:
                #    break

        return images[:counter,]


if __name__ == "__main__":
    DataProcessor()
    print('Data has been successfully processed')



     #   data_training = np.array([])
     #   data_validation = np.array([])
     #   data_test = np.array([])
'''
        data_training = np.append(parasitized[0:int(self.data_split[0]*parasitized_length),],
                                  uninfected[0:int(self.data_split[0]*uninfected_length),], axis=0)
        data_validation = np.append(parasitized[int(self.data_split[0]*parasitized_length)+1:
                                                int(self.data_split[1]*parasitized_length),],
                          uninfected[int(self.data_split[0]*uninfected_length)+1:
                                     int(self.data_split[1]*uninfected_length),], axis=0)
        data_test = np.append(parasitized[int(self.data_split[1]*parasitized_length)+1:,],
                          uninfected[int(self.data_split[1]*uninfected_length)+1:,], axis=0)

        shuffle_order = [i for i in range(data_training.shape[0])]
        np.random.shuffle(shuffle_order)
        data_training = data_training[shuffle_order,]

        shuffle_order = [i for i in range(data_validation.shape[0])]
        np.random.shuffle(shuffle_order)
        data_validation = data_validation[shuffle_order,]

        shuffle_order = [i for i in range(data_test.shape[0])]
        np.random.shuffle(shuffle_order)
        data_test = data_test[shuffle_order,]
'''
