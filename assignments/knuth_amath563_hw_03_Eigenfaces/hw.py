import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import seaborn as sns


'''
Uncropped Class
'''
class LoadUncropped:
    def __init__(self, data_dir):
        self.data_dir       = data_dir
        self.all_file_names = sorted(glob.glob(data_dir+'*'))
        
        get_sample_number(self)
        get_image_shape(self)
        get_cropped_sample_labels(self)
        
        self.X = load_images(self.all_file_names,flatten=True)
        
        self.n_features = self.X.shape[1]
        self.n_classes  = len(list(set(self.labels)))

'''
Uncropped Class Functions
'''
def get_cropped_sample_labels(self):
    c = 0
    labels = []
    y = []
    for i in self.all_file_names:
        label = i.split('.')[-2]
        labels.append(label)
        y.append(c)
        c = c+1
        if c == 11:
            c = 0
    self.labels = np.array(labels)
    self.y      = np.array(y)
    
    
    
'''
Cropped Class
'''
class LoadCropped:
    def __init__(self, data_dir):
        self.data_dir       = data_dir
        self.all_file_names = sorted(glob.glob(self.data_dir+'*/*.pgm'))
        self.face_dirs      = sorted(glob.glob(self.data_dir+'*'))
        
        get_cropped_data(self)
        get_cropped_sample_names(self)
        get_sample_number(self)
        get_image_shape(self)
        
        self.n_features = self.X.shape[1]
        self.n_classes  = len(self.target_names)


'''
Cropped Class Functions
'''
def get_cropped_data(self):
    faces = []
    face_ids = []
    for i,v in enumerate(self.face_dirs):
        images_fn = sorted(glob.glob(v+'/*.pgm'))
        for j in images_fn:
            face_ids.append(i)
        faces_subject = load_images(images_fn,flatten=True).T
        faces.append(faces_subject)

    self.y = np.array(face_ids)
    self.X = np.hstack(faces).T
    self.faces = faces 

def get_cropped_sample_names(self):
    names = []
    for i in self.face_dirs:
        name = os.path.split(i)[-1]
        names.append(name)
    self.target_names = np.array(names)

def get_first_face_per_dir(face_dirs):
    face_images_list = []
    for i, v in enumerate(face_dirs):
        images_fn = sorted(glob.glob(face_dirs[i]+'/*.pgm'))
        face_images_list.append(images_fn[0])
        
    return face_images_list




'''
Generic functions
'''

def get_sample_number(self):
    self.n_samples= len(self.all_file_names)

def get_image_shape(self):
    img = mpimg.imread(self.all_file_names[0])
    self.h, self.w = img.shape
    
def load_images(image_file_name_list,
                flatten=False):
    images_list = []

    for i, v in enumerate(image_file_name_list):
        img = mpimg.imread(image_file_name_list[i])
        if flatten:
            images_list.append(img.flatten())
        else:
            images_list.append(img)

    image_arrays = np.array(images_list)
    
    return image_arrays

def plot_images_subset(image_arrays,
                       rows = 5,
                       columns = 5,
                       reshape=None,
                       figsize=(10, 10),
                       cmap = 'gray',
                       title=None,
                       labels=None):

    plt.figure(figsize=figsize)

    for i in range(rows*columns):
        ax = plt.subplot(rows, columns, i + 1)
        
        try:
            if reshape:
                image = image_arrays[i].reshape(reshape)
            else:
                image = image_arrays[i]
            ax.imshow(image, cmap = cmap)
            ax.set_xticks(())
            ax.set_yticks(())
            
            if not isinstance(labels, type(None)):
                ax.set_title(labels[i])
        except:
            ax.axis('off')
            pass
        
    if not isinstance(title, type(None)):
        plt.suptitle(title, fontsize=15)
#         plt.tight_layout()
        plt.subplots_adjust(top=0.95);