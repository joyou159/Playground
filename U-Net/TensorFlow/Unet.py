import tensorflow as tf
import tensorflow.keras.layers as tfl

def build_Unet(width, height, channels):
    input = tfl.Input((width, height, channels))
    convert = tfl.Lambda(lambda x: x/255)(input)

    # contracting path 
    c1  = tfl.Conv2D(filters = 16,kernel_size = (3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(convert)
    c1 = tfl.Dropout(0.1)(c1) 
    c1  = tfl.Conv2D(filters = 16,kernel_size = (3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(c1)
    p1 = tfl.MaxPooling2D(pool_size = (2,2))(c1)

    c2  = tfl.Conv2D(filters = 32,kernel_size = (3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(p1)
    c2 = tfl.Dropout(0.1)(c2) 
    c2  = tfl.Conv2D(filters = 32,kernel_size = (3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(c2)
    p2 = tfl.MaxPooling2D(pool_size = (2,2))(c2)

    c3  = tfl.Conv2D(filters = 64,kernel_size = (3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(p2)
    c3 = tfl.Dropout(0.2)(c3) 
    c3  = tfl.Conv2D(filters = 64,kernel_size = (3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(c3)
    p3 = tfl.MaxPooling2D(pool_size = (2,2))(c3)

    c4  = tfl.Conv2D(filters = 128,kernel_size = (3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(p3)
    c4 = tfl.Dropout(0.2)(c4) 
    c4  = tfl.Conv2D(filters = 128,kernel_size = (3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(c4)
    p4 = tfl.MaxPooling2D(pool_size = (2,2))(c4)

    c5  = tfl.Conv2D(filters = 256,kernel_size = (3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(p4)
    c5 = tfl.Dropout(0.3)(c5) 
    c5  = tfl.Conv2D(filters = 256,kernel_size = (3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(c5)
    u5 = tfl.Conv2DTranspose(filters = 128, kernel_size = (2,2), strides = (2,2), padding = "same")(c5)

    # expanding path 
    u6 = tfl.concatenate([u5, c4], axis = 3)
    c6  = tfl.Conv2D(filters = 128,kernel_size = (3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(u6)
    c6 = tfl.Dropout(0.2)(c6) 
    c6  = tfl.Conv2D(filters = 128,kernel_size = (3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(c6)
    u6 = tfl.Conv2DTranspose(filters = 64, kernel_size = (2,2), strides = (2,2), padding = "same")(c6)

    u7 = tfl.concatenate([u6, c3], axis = 3)
    c7  = tfl.Conv2D(filters = 64,kernel_size = (3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(u7)
    c7 = tfl.Dropout(0.2)(c7) 
    c7  = tfl.Conv2D(filters = 64,kernel_size = (3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(c7)
    u7 = tfl.Conv2DTranspose(filters = 32, kernel_size = (2,2), strides = (2,2), padding = "same")(c7)

    u8 = tfl.concatenate([u7, c2], axis = 3)
    c8  = tfl.Conv2D(filters = 32,kernel_size = (3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(u8)
    c8 = tfl.Dropout(0.1)(c8)
    c8  = tfl.Conv2D(filters = 32,kernel_size = (3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(c8)
    u8 = tfl.Conv2DTranspose(filters = 16, kernel_size = (2,2), strides = (2,2), padding = "same")(c8)

    u9 = tfl.concatenate([u8, c1], axis = 3)
    c9  = tfl.Conv2D(filters = 16,kernel_size = (3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(u9)
    c9 = tfl.Dropout(0.1)(c9)
    c9  = tfl.Conv2D(filters = 16,kernel_size = (3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(c9)

    output = tfl.Conv2D(1, (1,1), activation = "sigmoid")(c9) # 2 class segmentation

    model = tf.keras.Model(inputs = [input], outputs = [output])

    return model  


    

    











    