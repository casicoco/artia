import matplotlib.pyplot as plt
import tensorflow as tf
import PIL.Image
import numpy as np

def load_img():
    #data_path to raw_data to extract content & style images
    content_data_path = '/content/drive/My Drive/Artia on essecdrive/charles_style_transfer/dataset_style_transfer/content_charles.png'
    style_data_path = '/content/drive/My Drive/Artia on essecdrive/charles_style_transfer/dataset_style_transfer/style_vangogh.jpeg'

    return {'content':content_data_path,'style':style_data_path}

def img_to_tensor(path_to_img,max_dim=672):
    img = tf.io.read_file(path_to_img)                                        #load data into tensor object
    img = tf.image.decode_image(img, channels=3)                              #convert type of image (png,jpeg) into a single one with 3 channels (RGB)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim                                                #compute scale for tensor object

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)                                     #resizing
    img = img[tf.newaxis, :]
    return img                                                                #return image of shape (number of image, height in px, width in px, RGB)

def VGG_preprocessing(image_tensor):
    x = tf.keras.applications.vgg19.preprocess_input(image_tensor*255)        #Calling VGG19 model on content_tensor
    x = tf.image.resize(x, (224, 224))                                        #Resizing to 224 images used by the VGG model
    return x

def vgg_layers():
    #Initialization of VGG19 model
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    #Generate a list of layers outputs
    outputs = [layer_.output for layer_ in vgg.layers]

    #Generate a Model that will take an input (None,None,None,3) and generates 5 differents outputs
    model = tf.keras.Model(vgg.input, outputs)
    return model

def output_extraction(model,image_processed,image_layers=None):
    if image_layers:
        return {model.layers[index].name:output for index,output in enumerate(model(image_processed)) if model.layers[index].name in image_layers}    #Extract all {name:output} if not
    return {model.layers[index].name:output for index,output in enumerate(model(image_processed))}

def content_loss(content_output, image_output):
    return tf.reduce_mean(tf.square(tf.add(content_output,-image_output)))

def gram_matrix(image_output):
    """
    Computation of the gram matrix from the outputs of the model.
    Initial object is of shape (1,px,px,channels), we convert it into
    a transposable matrix of dim (1*px*px,channels) and finally
    we compute the X.X^T
    """
    n_channels = image_output.shape[-1]
    n_size=image_output.shape[1]**2
    transposable_matrix = tf.reshape(image_output, [-1, n_channels])
    style_matrix = tf.matmul(transposable_matrix, transposable_matrix, transpose_a=True)
    return style_matrix

def style_loss(style_output,image_output):
    gram_style=gram_matrix(style_output)
    gram_image=gram_matrix(image_output)
    N=style_output.shape[-1]**2                                 #squared number of layers
    M=(style_output.shape[1]**2)**2                             #squared filter dimension
    return tf.reduce_sum(tf.square(tf.add(gram_style,-gram_image)))/(4*N*M)

def compute_loss(style_output,content_output,image_output,style_weight=1e-2,content_weight=1e4):

    content_l=tf.reduce_sum([content_loss(content_output[key], image_output[key])/len(content_output.keys()) for key in content_output.keys()])

    style_l=tf.reduce_sum([style_loss(style_output[key],image_output[key])/len(style_output.keys()) for key in style_output.keys()])

    total_l=style_weight*style_l+content_weight*content_l

    return {'style_loss':style_l,'content_loss':content_l,'total_loss':total_l}

def denoising_loss(image,denoising_weight=1):
    x_var = tf.reduce_sum(tf.abs(image[:, 1:, :, :] - image[:, :-1, :, :]))
    y_var = tf.reduce_sum(tf.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))

    return denoising_weight*0.5*(x_var+y_var)

model=vgg_layers()
@tf.function()

def train_step(image):
    with tf.GradientTape() as tape:

        image_processed=VGG_preprocessing(image)
        image_output=output_extraction(model,image_processed,style_layers+content_layers)
        loss = compute_loss(style_targets,content_targets,image_output,style_weight=1.8e-2,content_weight=2e4)['total_loss']
        loss+=denoising_loss(image_processed,12)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image,0.0,1.0))

def training(image,epochs,steps_per_epoch):
    epochs = 30
    steps_per_epoch = 50

    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image)
    return

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

if __name__ == "__main__":
    #defining main layers
    content_layers = ['block5_conv2']

    style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    #defining tensor/processed
    style_tensor=img_to_tensor(load_img()['style'])
    content_tensor=img_to_tensor(load_img()['content'])

    content_processed = VGG_preprocessing(content_tensor)
    style_processed = VGG_preprocessing(style_tensor)

    style_outputs=output_extraction(vgg_layers(),style_processed,style_layers)
    content_outputs=output_extraction(vgg_layers(),content_processed,content_layers)

    content_targets = content_outputs.copy()
    style_targets=style_outputs.copy()

    #running the model
