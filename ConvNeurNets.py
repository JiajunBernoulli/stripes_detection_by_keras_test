import keras
from keras import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
    Activation, ZeroPadding2D, LeakyReLU, Convolution2D, GlobalAveragePooling2D, Lambda
from keras.layers import add, Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D

from config import NUM_CLASS, WIDTH, HEIGHT
import keras.backend as K

from custom_layers import Scale


class LeNet:
    def CreateModel(self):
        model = Sequential()
        model.add(Conv2D(32, (5, 5), strides=(1, 1), input_shape=(WIDTH, HEIGHT, 3), padding='valid', activation='relu',
                         kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='valid', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(NUM_CLASS, activation='softmax'))
        return model

class AlexNet:
    def CreateModel(self):
        model = Sequential()
        model.add(
            Conv2D(96, (11, 11), strides=(4, 4), input_shape=(WIDTH, HEIGHT, 3), padding='valid', activation='relu',
                   kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASS, activation='softmax'))
        return model

class ZF_Net:
    def CreateModel(self):
        model = Sequential()
        model.add(Conv2D(96, (7, 7), strides=(2, 2), input_shape=(WIDTH, HEIGHT, 3), padding='valid', activation='relu',
                         kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(256, (5, 5), strides=(2, 2), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASS, activation='softmax'))
        return model

class VGG13:
    def CreateModel(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(224, 224, 3), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 2), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASS, activation='softmax'))
        return model

class VGG16:
    def CreateModel(self):
        model = Sequential()

        model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(WIDTH, HEIGHT, 3), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 2), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASS, activation='softmax'))
        return model

class MiniGoogLeNet:
    @staticmethod
    def conv_module(x, K, kX, kY, stride, chanDim, padding="same"):#x表示输入数据，K表示conv的filter的数量，KX,KY表示kernel_size
        #define a CONV => BN => RELU pattern,我们严格按照原论文的说法，使用CONV => BN => RELU的顺序，但是实际上，CONV => Relu => BN的效果会更好一些
        x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation("relu")(x)
        # return the block
        return x

    @staticmethod
    def inception_module(x,numK1_1,numK3_3,chanDim):#x表示输入数据,numK1_1,numK3_3表示kernel的filter的数量，chanDim：first_channel or last_channel
        conv1_1=MiniGoogLeNet.conv_module(x,numK1_1,1,1,(1,1),chanDim)
        conv3_3=MiniGoogLeNet.conv_module(x,numK3_3,3,3,(1,1),chanDim)
        x=concatenate([conv1_1,conv3_3],axis=chanDim)#将conv1_1和conv3_3串联到一起
        return x

    @staticmethod
    def downsample_module(x,K,chanDim):#K表示conv的filter的数量
        conv3_3=MiniGoogLeNet.conv_module(x,K,3,3,(2,2),chanDim,padding='valid')#padding=same表示：出输入和输出的size是相同的，由于加入了padding，如果是padding=valid，那么padding=0
        pool=MaxPooling2D((3,3),strides=(2,2))(x)
        x=concatenate([conv3_3,pool],axis=chanDim)#将conv3_3和maxPooling串到一起
        return x

    @staticmethod
    def CreateModel(width=WIDTH, height=HEIGHT, depth=3, classes=NUM_CLASS):
        inputShape = (height, width, depth)#keras默认channel last，tf作为backend
        chanDim = -1
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        # define the model input and first CONV module
        inputs = Input(shape=inputShape)


        x = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1),chanDim)
        # two Inception modules followed by a downsample module

        x = MiniGoogLeNet.inception_module(x, 32, 32, chanDim)#第一个分叉
        x = MiniGoogLeNet.inception_module(x, 32, 48, chanDim)#第二个分叉
        x = MiniGoogLeNet.downsample_module(x, 80, chanDim)#第三个分叉，含有maxpooling



        # four Inception modules followed by a downsample module
        x = MiniGoogLeNet.inception_module(x, 112, 48, chanDim)
        x = MiniGoogLeNet.inception_module(x, 96, 64, chanDim)
        x = MiniGoogLeNet.inception_module(x, 80, 80, chanDim)
        x = MiniGoogLeNet.inception_module(x, 48, 96, chanDim)
        x = MiniGoogLeNet.downsample_module(x, 96, chanDim)

        # two Inception modules followed by global POOL and dropout
        x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)
        x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)#输出是（7×7×（160+176））
        x = AveragePooling2D((7, 7))(x)#经过平均池化之后变成了（1*1*376）
        x = Dropout(0.5)(x)

        # softmax classifier
        x = Flatten()(x)#特征扁平化
        x = Dense(classes)(x)#全连接层，进行多分类,形成最终的10分类
        x = Activation("softmax")(x)
        # create the model
        model = Model(inputs, x, name="googlenet")
        # return the constructed network architecture
        return model

class InceptionV4:
    global CONV_BLOCK_COUNT, INCEPTION_A_COUNT, INCEPTION_B_COUNT, INCEPTION_C_COUNT
    CONV_BLOCK_COUNT = 0  # 用来命名计数卷积编号
    INCEPTION_A_COUNT = 0
    INCEPTION_B_COUNT = 0
    INCEPTION_C_COUNT = 0


    def conv_block(x, nb_filters, nb_row, nb_col, strides=(1, 1), padding='same', use_bias=False):
        global CONV_BLOCK_COUNT
        CONV_BLOCK_COUNT += 1
        with K.name_scope('conv_block_' + str(CONV_BLOCK_COUNT)):
            x = Conv2D(filters=nb_filters,
                       kernel_size=(nb_row, nb_col),
                       strides=strides,
                       padding=padding,
                       use_bias=use_bias)(x)
            x = BatchNormalization(axis=-1, momentum=0.9997, scale=False)(x)
            x = Activation("relu")(x)
        return x


    def stem(x_input):
        with K.name_scope('stem'):
            x = InceptionV4.conv_block(x_input, 32, 3, 3, strides=(2, 2), padding='valid')
            x = InceptionV4.conv_block(x, 32, 3, 3, padding='valid')
            x = InceptionV4.conv_block(x, 64, 3, 3)

            x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
            x2 = InceptionV4.conv_block(x, 96, 3, 3, strides=(2, 2), padding='valid')

            x = concatenate([x1, x2], axis=-1)

            x1 = InceptionV4.conv_block(x, 64, 1, 1)
            x1 = InceptionV4.conv_block(x1, 96, 3, 3, padding='valid')

            x2 = InceptionV4.conv_block(x, 64, 1, 1)
            x2 = InceptionV4.conv_block(x2, 64, 7, 1)
            x2 = InceptionV4.conv_block(x2, 64, 1, 7)
            x2 = InceptionV4.conv_block(x2, 96, 3, 3, padding='valid')

            x = concatenate([x1, x2], axis=-1)

            x1 = InceptionV4.conv_block(x, 192, 3, 3, strides=(2, 2), padding='valid')
            x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

            merged_vector = concatenate([x1, x2], axis=-1)
        return merged_vector


    def inception_A(x_input):
        """35*35 卷积块"""
        global INCEPTION_A_COUNT
        INCEPTION_A_COUNT += 1
        with K.name_scope('inception_A' + str(INCEPTION_A_COUNT)):
            averagepooling_conv1x1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(
                x_input)  # 35 * 35 * 192
            averagepooling_conv1x1 = InceptionV4.conv_block(averagepooling_conv1x1, 96, 1, 1)  # 35 * 35 * 96

            conv1x1 = InceptionV4.conv_block(x_input, 96, 1, 1)  # 35 * 35 * 96

            conv1x1_3x3 = InceptionV4.conv_block(x_input, 64, 1, 1)  # 35 * 35 * 64
            conv1x1_3x3 = InceptionV4.conv_block(conv1x1_3x3, 96, 3, 3)  # 35 * 35 * 96

            conv3x3_3x3 = InceptionV4.conv_block(x_input, 64, 1, 1)  # 35 * 35 * 64
            conv3x3_3x3 = InceptionV4.conv_block(conv3x3_3x3, 96, 3, 3)  # 35 * 35 * 96
            conv3x3_3x3 = InceptionV4.conv_block(conv3x3_3x3, 96, 3, 3)  # 35 * 35 * 96

            merged_vector = concatenate([averagepooling_conv1x1, conv1x1, conv1x1_3x3, conv3x3_3x3],
                                        axis=-1)  # 35 * 35 * 384
        return merged_vector


    def inception_B(x_input):
        """17*17 卷积块"""
        global INCEPTION_B_COUNT
        INCEPTION_B_COUNT += 1
        with K.name_scope('inception_B' + str(INCEPTION_B_COUNT)):
            averagepooling_conv1x1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x_input)
            averagepooling_conv1x1 = InceptionV4.conv_block(averagepooling_conv1x1, 128, 1, 1)

            conv1x1 = InceptionV4.conv_block(x_input, 384, 1, 1)

            conv1x7_1x7 = InceptionV4.conv_block(x_input, 192, 1, 1)
            conv1x7_1x7 = InceptionV4.conv_block(conv1x7_1x7, 224, 1, 7)
            conv1x7_1x7 = InceptionV4.conv_block(conv1x7_1x7, 256, 1, 7)

            conv2_1x7_7x1 = InceptionV4.conv_block(x_input, 192, 1, 1)
            conv2_1x7_7x1 = InceptionV4.conv_block(conv2_1x7_7x1, 192, 1, 7)
            conv2_1x7_7x1 = InceptionV4.conv_block(conv2_1x7_7x1, 224, 7, 1)
            conv2_1x7_7x1 = InceptionV4.conv_block(conv2_1x7_7x1, 224, 1, 7)
            conv2_1x7_7x1 = InceptionV4.conv_block(conv2_1x7_7x1, 256, 7, 1)

            merged_vector = concatenate([averagepooling_conv1x1, conv1x1, conv1x7_1x7, conv2_1x7_7x1], axis=-1)
        return merged_vector


    def inception_C(x_input):
        """8*8 卷积块"""
        global INCEPTION_C_COUNT
        INCEPTION_C_COUNT += 1
        with K.name_scope('Inception_C' + str(INCEPTION_C_COUNT)):
            averagepooling_conv1x1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x_input)
            averagepooling_conv1x1 = InceptionV4.conv_block(averagepooling_conv1x1, 256, 1, 1)

            conv1x1 = InceptionV4.conv_block(x_input, 256, 1, 1)

            # 用 1x3 和 3x1 替代 3x3
            conv3x3_1x1 = InceptionV4.conv_block(x_input, 384, 1, 1)
            conv3x3_1 = InceptionV4.conv_block(conv3x3_1x1, 256, 1, 3)
            conv3x3_2 = InceptionV4.conv_block(conv3x3_1x1, 256, 3, 1)

            conv2_3x3_1x1 = InceptionV4.conv_block(x_input, 384, 1, 1)
            conv2_3x3_1x1 = InceptionV4.conv_block(conv2_3x3_1x1, 448, 1, 3)
            conv2_3x3_1x1 = InceptionV4.conv_block(conv2_3x3_1x1, 512, 3, 1)
            conv2_3x3_1x1_1 = InceptionV4.conv_block(conv2_3x3_1x1, 256, 3, 1)
            conv2_3x3_1x1_2 = InceptionV4.conv_block(conv2_3x3_1x1, 256, 1, 3)

            merged_vector = concatenate(
                [averagepooling_conv1x1, conv1x1, conv3x3_1, conv3x3_2, conv2_3x3_1x1_1, conv2_3x3_1x1_2], axis=-1)
        return merged_vector


    def reduction_A(x_input, k=192, l=224, m=256, n=384):
        with K.name_scope('Reduction_A'):
            """Architecture of a 35 * 35 to 17 * 17 Reduction_A block."""
            maxpool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x_input)

            conv3x3 = InceptionV4.conv_block(x_input, n, 3, 3, strides=(2, 2), padding='valid')

            conv2_3x3 = InceptionV4.conv_block(x_input, k, 1, 1)
            conv2_3x3 = InceptionV4.conv_block(conv2_3x3, l, 3, 3)
            conv2_3x3 = InceptionV4.conv_block(conv2_3x3, m, 3, 3, strides=(2, 2), padding='valid')

            merged_vector = concatenate([maxpool, conv3x3, conv2_3x3], axis=-1)
        return merged_vector


    def reduction_B(x_input):
        """Architecture of a 17 * 17 to 8 * 8 Reduction_B block."""
        with K.name_scope('Reduction_B'):
            maxpool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x_input)

            conv3x3 = InceptionV4.conv_block(x_input, 192, 1, 1)
            conv3x3 = InceptionV4.conv_block(conv3x3, 192, 3, 3, strides=(2, 2), padding='valid')

            conv1x7_7x1_3x3 = InceptionV4.conv_block(x_input, 256, 1, 1)
            conv1x7_7x1_3x3 = InceptionV4.conv_block(conv1x7_7x1_3x3, 256, 1, 7)
            conv1x7_7x1_3x3 = InceptionV4.conv_block(conv1x7_7x1_3x3, 320, 7, 1)
            conv1x7_7x1_3x3 = InceptionV4.conv_block(conv1x7_7x1_3x3, 320, 3, 3, strides=(2, 2), padding='valid')

            merged_vector = concatenate([maxpool, conv3x3, conv1x7_7x1_3x3], axis=-1)
        return merged_vector


    def CreateModel(self, nb_classes=NUM_CLASS, load_weights=True):
        x_input = Input(shape=(WIDTH, HEIGHT, 3))
        # Stem
        x = InceptionV4.stem(x_input)  # 35 x 35 x 384
        # 4 x Inception_A
        for i in range(4):
            x = InceptionV4.inception_A(x)  # 35 x 35 x 384
        # Reduction_A
        x = InceptionV4.reduction_A(x, k=192, l=224, m=256, n=384)  # 17 x 17 x 1024
        # 7 x Inception_B
        for i in range(7):
            x = InceptionV4.inception_B(x)  # 17 x 17 x1024
        # Reduction_B
        x = InceptionV4.reduction_B(x)  # 8 x 8 x 1536
        # Average Pooling
        x = AveragePooling2D(pool_size=(5, 5))(x)  # 1536
        # dropout
        x = Dropout(0.2)(x)
        x = Flatten()(x)  # 1536
        # 全连接层
        x = Dense(units=nb_classes, activation='softmax')(x)
        model = Model(inputs=x_input, outputs=x, name='Inception-V4')
        return model

class ResNet34:
    def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x

    def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):

        x = ResNet34.Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
        x = ResNet34.Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
        if with_conv_shortcut:
            shortcut = ResNet34.Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x

    def CreateModel(self):
        inpt = Input(shape=(WIDTH, HEIGHT, 3))
        x = ZeroPadding2D((3, 3))(inpt)

        x = ResNet34.Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # (56,56,64)

        x = ResNet34.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        # (28,28,128)
        x = ResNet34.Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        # (14,14,256)
        x = ResNet34.Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        # (7,7,512)
        x = ResNet34.Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = ResNet34.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Flatten()(x)
        x = Dense(NUM_CLASS, activation='softmax')(x)

        model = Model(inputs=inpt, outputs=x)
        return model

class ResNet34_0_1:
    def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x

    def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):

        x = ResNet34_0_1.Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
        x = ResNet34_0_1.Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
        if with_conv_shortcut:
            shortcut = ResNet34_0_1.Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x

    def CreateModel(self):
        inpt = Input(shape=(WIDTH, HEIGHT, 3))
        x = ZeroPadding2D((3, 3))(inpt)

        x = ResNet34_0_1.Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # (56,56,64)

        x = ResNet34_0_1.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34_0_1.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34_0_1.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        # (28,28,128)
        x = ResNet34_0_1.Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_0_1.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34_0_1.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34_0_1.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        # (14,14,256)
        x = ResNet34_0_1.Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_0_1.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_0_1.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_0_1.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_0_1.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_0_1.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        # (7,7,512)
        x = ResNet34_0_1.Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_0_1.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = ResNet34_0_1.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Flatten()(x)
        x = Dropout(0.1)(x)
        x = Dense(NUM_CLASS, activation='softmax')(x)

        model = Model(inputs=inpt, outputs=x)
        return model

class ResNet34_0_2:
    def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x
    def Conv2d_Test(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x

    def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):

        x = ResNet34_0_2.Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
        x = ResNet34_0_2.Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
        if with_conv_shortcut:
            shortcut = ResNet34_0_2.Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x

    def CreateModel(self):
        inpt = Input(shape=(WIDTH, HEIGHT, 3))
        x = ZeroPadding2D((3, 3))(inpt)

        # x = ResNet34_0_2.Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
        x = ResNet34_0_2.Conv2d_Test(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # (56,56,64)

        x = ResNet34_0_2.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34_0_2.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34_0_2.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        # (28,28,128)
        x = ResNet34_0_2.Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_0_2.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34_0_2.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34_0_2.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        # (14,14,256)
        x = ResNet34_0_2.Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_0_2.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_0_2.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_0_2.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_0_2.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_0_2.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        # (7,7,512)
        x = ResNet34_0_2.Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_0_2.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = ResNet34_0_2.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Flatten()(x)
        # x = Dropout(0.2)(x)
        x = Dense(NUM_CLASS, activation='softmax')(x)

        model = Model(inputs=inpt, outputs=x)
        return model

class ResNet34_0_3:
    def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x

    def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):

        x = ResNet34_0_3.Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
        x = ResNet34_0_3.Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
        if with_conv_shortcut:
            shortcut = ResNet34_0_3.Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x

    def CreateModel(self):
        inpt = Input(shape=(WIDTH, HEIGHT, 3))
        x = ZeroPadding2D((3, 3))(inpt)

        x = ResNet34_0_3.Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # (56,56,64)

        x = ResNet34_0_3.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34_0_3.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34_0_3.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        # (28,28,128)
        x = ResNet34_0_3.Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_0_3.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34_0_3.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34_0_3.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        # (14,14,256)
        x = ResNet34_0_3.Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_0_3.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_0_3.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_0_3.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_0_3.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_0_3.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        # (7,7,512)
        x = ResNet34_0_3.Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_0_3.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = ResNet34_0_3.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Flatten()(x)
        x = Dropout(0.3)(x)
        x = Dense(NUM_CLASS, activation='softmax')(x)

        model = Model(inputs=inpt, outputs=x)
        return model
    
class ResNet34_0_4:
    def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x

    def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):

        x = ResNet34_0_4.Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
        x = ResNet34_0_4.Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
        if with_conv_shortcut:
            shortcut = ResNet34_0_4.Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x

    def CreateModel(self):
        inpt = Input(shape=(WIDTH, HEIGHT, 3))
        x = ZeroPadding2D((3, 3))(inpt)

        x = ResNet34_0_4.Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # (56,56,64)

        x = ResNet34_0_4.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34_0_4.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34_0_4.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        # (28,28,128)
        x = ResNet34_0_4.Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_0_4.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34_0_4.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34_0_4.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        # (14,14,256)
        x = ResNet34_0_4.Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_0_4.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_0_4.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_0_4.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_0_4.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_0_4.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        # (7,7,512)
        x = ResNet34_0_4.Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_0_4.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = ResNet34_0_4.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Flatten()(x)
        x = Dropout(0.4)(x)
        x = Dense(NUM_CLASS, activation='softmax')(x)

        model = Model(inputs=inpt, outputs=x)
        return model
    
class ResNet34_1:
    def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x

    def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):

        x = ResNet34_1.Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
        x = ResNet34_1.Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
        if with_conv_shortcut:
            shortcut = ResNet34_1.Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x

    def CreateModel(self):
        inpt = Input(shape=(WIDTH, HEIGHT, 3))
        x = ZeroPadding2D((3, 3))(inpt)

        x = ResNet34_1.Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
        x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # (56,56,64)

        x = ResNet34_1.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34_1.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34_1.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        # (28,28,128)
        x = ResNet34_1.Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_1.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34_1.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34_1.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        # (14,14,256)
        x = ResNet34_1.Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_1.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_1.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_1.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_1.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_1.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        # (7,7,512)
        x = ResNet34_1.Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_1.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = ResNet34_1.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Flatten()(x)
        x = Dropout(0.1)(x)
        x = Dense(NUM_CLASS, activation='softmax')(x)

        model = Model(inputs=inpt, outputs=x)
        return model

class ResNet34_1_1:
    def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x

    def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):

        x = ResNet34_1_1.Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
        x = ResNet34_1_1.Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
        if with_conv_shortcut:
            shortcut = ResNet34_1_1.Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x

    def CreateModel(self):
        inpt = Input(shape=(WIDTH, HEIGHT, 3))
        x = ZeroPadding2D((3, 3))(inpt)

        x = ResNet34_1_1.Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
        x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # (56,56,64)

        x = ResNet34_1_1.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34_1_1.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34_1_1.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        # (28,28,128)
        x = ResNet34_1_1.Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_1_1.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34_1_1.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34_1_1.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        # (14,14,256)
        x = ResNet34_1_1.Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_1_1.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_1_1.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_1_1.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_1_1.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_1_1.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        # (7,7,512)
        x = ResNet34_1_1.Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_1_1.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = ResNet34_1_1.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Flatten()(x)
        x = Dropout(0.1)(x)
        x = Dense(NUM_CLASS, activation='softmax')(x)

        model = Model(inputs=inpt, outputs=x)
        return model
    
class ResNet34_1_2:
    def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x

    def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):

        x = ResNet34_1_2.Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
        x = ResNet34_1_2.Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
        if with_conv_shortcut:
            shortcut = ResNet34_1_2.Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x

    def CreateModel(self):
        inpt = Input(shape=(WIDTH, HEIGHT, 3))
        x = ZeroPadding2D((3, 3))(inpt)

        x = ResNet34_1_2.Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
        x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # (56,56,64)

        x = ResNet34_1_2.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34_1_2.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34_1_2.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        # (28,28,128)
        x = ResNet34_1_2.Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_1_2.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34_1_2.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34_1_2.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        # (14,14,256)
        x = ResNet34_1_2.Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_1_2.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_1_2.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_1_2.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_1_2.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_1_2.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        # (7,7,512)
        x = ResNet34_1_2.Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_1_2.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = ResNet34_1_2.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(NUM_CLASS, activation='softmax')(x)

        model = Model(inputs=inpt, outputs=x)
        return model

class ResNet34_1_3:
    def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x

    def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):

        x = ResNet34_1_3.Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
        x = ResNet34_1_3.Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
        if with_conv_shortcut:
            shortcut = ResNet34_1_3.Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x

    def CreateModel(self):
        inpt = Input(shape=(WIDTH, HEIGHT, 3))
        x = ZeroPadding2D((3, 3))(inpt)

        x = ResNet34_1_3.Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
        x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # (56,56,64)

        x = ResNet34_1_3.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34_1_3.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34_1_3.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        # (28,28,128)
        x = ResNet34_1_3.Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_1_3.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34_1_3.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34_1_3.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        # (14,14,256)
        x = ResNet34_1_3.Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_1_3.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_1_3.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_1_3.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_1_3.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_1_3.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        # (7,7,512)
        x = ResNet34_1_3.Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_1_3.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = ResNet34_1_3.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Flatten()(x)
        x = Dropout(0.3)(x)
        x = Dense(NUM_CLASS, activation='softmax')(x)

        model = Model(inputs=inpt, outputs=x)
        return model

class ResNet34_1_4:
    def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x

    def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):

        x = ResNet34_1_4.Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
        x = ResNet34_1_4.Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
        if with_conv_shortcut:
            shortcut = ResNet34_1_4.Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x

    def CreateModel(self):
        inpt = Input(shape=(WIDTH, HEIGHT, 3))
        x = ZeroPadding2D((3, 3))(inpt)

        x = ResNet34_1_4.Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
        x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # (56,56,64)

        x = ResNet34_1_4.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34_1_4.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34_1_4.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        # (28,28,128)
        x = ResNet34_1_4.Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_1_4.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34_1_4.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34_1_4.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        # (14,14,256)
        x = ResNet34_1_4.Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_1_4.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_1_4.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_1_4.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_1_4.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_1_4.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        # (7,7,512)
        x = ResNet34_1_4.Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_1_4.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = ResNet34_1_4.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Flatten()(x)
        x = Dropout(0.4)(x)
        x = Dense(NUM_CLASS, activation='softmax')(x)

        model = Model(inputs=inpt, outputs=x)
        return model
    
    
class ResNet34_2:
    def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x

    def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):

        x = ResNet34_2.Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
        x = ResNet34_2.Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
        if with_conv_shortcut:
            shortcut = ResNet34_2.Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x

    def CreateModel(self):
        inpt = Input(shape=(WIDTH, HEIGHT, 3))
        x = ZeroPadding2D((3, 3))(inpt)

        x = ResNet34_2.Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
        x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # (56,56,64)

        x = ResNet34_2.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34_2.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34_2.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        # (28,28,128)
        x = ResNet34_2.Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_2.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34_2.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34_2.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        # (14,14,256)
        x = ResNet34_2.Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_2.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_2.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_2.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_2.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_2.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        # (7,7,512)
        x = ResNet34_2.Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_2.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = ResNet34_2.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = MaxPooling2D(pool_size=(7, 7))(x)
        x = Flatten()(x)
        x = Dense(NUM_CLASS, activation='softmax')(x)

        model = Model(inputs=inpt, outputs=x)
        return model
    
class ResNet34_3:
    def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x

    def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):

        x = ResNet34_3.Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
        x = ResNet34_3.Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
        if with_conv_shortcut:
            shortcut = ResNet34_3.Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x

    def CreateModel(self):
        inpt = Input(shape=(WIDTH, HEIGHT, 3))
        x = ZeroPadding2D((3, 3))(inpt)

        x = ResNet34_3.Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # (56,56,64)

        x = ResNet34_3.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34_3.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = ResNet34_3.Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        # (28,28,128)
        x = ResNet34_3.Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_3.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34_3.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = ResNet34_3.Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        # (14,14,256)
        x = ResNet34_3.Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_3.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_3.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_3.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_3.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = ResNet34_3.Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        # (7,7,512)
        x = ResNet34_3.Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet34_3.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = ResNet34_3.Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = MaxPooling2D(pool_size=(7, 7))(x)
        x = Flatten()(x)
        x = Dense(NUM_CLASS, activation='softmax')(x)

        model = Model(inputs=inpt, outputs=x)
        return model
    
    
class ResNet50:
    def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x

    def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
        x = ResNet50.Conv2d_BN(inpt, nb_filter=nb_filter[0], kernel_size=(1, 1), strides=strides, padding='same')
        x = ResNet50.Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(3, 3), padding='same')
        x = ResNet50.Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1, 1), padding='same')
        if with_conv_shortcut:
            shortcut = ResNet50.Conv2d_BN(inpt, nb_filter=nb_filter[2], strides=strides, kernel_size=kernel_size)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x

    def CreateModel(self):
        inpt = Input(shape=(WIDTH, HEIGHT, 3))
        x = ZeroPadding2D((3, 3))(inpt)
        x = ResNet50.Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        x = ResNet50.Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3), strides=(1, 1), with_conv_shortcut=True)
        x = ResNet50.Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3))
        x = ResNet50.Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3))

        x = ResNet50.Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet50.Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))
        x = ResNet50.Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))
        x = ResNet50.Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))

        x = ResNet50.Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet50.Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
        x = ResNet50.Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
        x = ResNet50.Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
        x = ResNet50.Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
        x = ResNet50.Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))

        x = ResNet50.Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = ResNet50.Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3))
        x = ResNet50.Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3))
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Flatten()(x)
        x = Dense(NUM_CLASS, activation='softmax')(x)

        model = Model(inputs=inpt, outputs=x)
        return model

class DenseNet121:
    import warnings
    warnings.filterwarnings("ignore")
    def CreateModel(self, nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4,
                    classes=NUM_CLASS, weights_path=None):
        '''Instantiate the DenseNet 121 architecture,
            # Arguments
                nb_dense_block: number of dense blocks to add to end
                growth_rate: number of filters to add per dense block
                nb_filter: initial number of filters
                reduction: reduction factor of transition blocks.
                dropout_rate: dropout rate
                weight_decay: weight decay factor
                classes: optional number of classes to classify images
                weights_path: path to pre-trained weights
            # Returns
                A Keras model instance.
        '''
        eps = 1.1e-5

        # compute compression factor
        compression = 1.0 - reduction

        # Handle Dimension Ordering for different backends
        global concat_axis
        if K.image_dim_ordering() == 'tf':
            concat_axis = 3
            img_input = Input(shape=(224, 224, 3), name='data')
        else:
            concat_axis = 1
            img_input = Input(shape=(3, 224, 224), name='data')

        # From architecture for ImageNet (Table 1 in the paper)
        nb_filter = 64
        nb_layers = [6, 12, 24, 16]  # For DenseNet-121

        # Initial convolution
        x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
        x = Convolution2D(nb_filter, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
        x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
        x = Scale(axis=concat_axis, name='conv1_scale')(x)
        x = Activation('relu', name='relu1')(x)
        x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

        # Add dense blocks
        for block_idx in range(nb_dense_block - 1):
            stage = block_idx + 2
            x, nb_filter = DenseNet121.dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate,
                                       dropout_rate=dropout_rate, weight_decay=weight_decay)

            # Add transition_block
            x = DenseNet121.transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate,
                                 weight_decay=weight_decay)
            nb_filter = int(nb_filter * compression)

        final_stage = stage + 1
        x, nb_filter = DenseNet121.dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)

        x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv' + str(final_stage) + '_blk_bn')(x)
        x = Scale(axis=concat_axis, name='conv' + str(final_stage) + '_blk_scale')(x)
        x = Activation('relu', name='relu' + str(final_stage) + '_blk')(x)
        x = GlobalAveragePooling2D(name='pool' + str(final_stage))(x)

        x = Dense(classes, name='fc6')(x)
        x = Activation('softmax', name='prob')(x)

        model = Model(img_input, x, name='densenet')

        if weights_path is not None:
            model.load_weights(weights_path)

        return model

    def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
        '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
            # Arguments
                x: input tensor
                stage: index for dense block
                branch: layer index within each dense block
                nb_filter: number of filters
                dropout_rate: dropout rate
                weight_decay: weight decay factor
        '''
        eps = 1.1e-5
        conv_name_base = 'conv' + str(stage) + '_' + str(branch)
        relu_name_base = 'relu' + str(stage) + '_' + str(branch)

        # 1x1 Convolution (Bottleneck layer)
        inter_channel = nb_filter * 4
        x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x1_bn')(x)
        x = Scale(axis=concat_axis, name=conv_name_base + '_x1_scale')(x)
        x = Activation('relu', name=relu_name_base + '_x1')(x)
        x = Convolution2D(inter_channel, 1, 1, name=conv_name_base + '_x1', bias=False)(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        # 3x3 Convolution
        x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x2_bn')(x)
        x = Scale(axis=concat_axis, name=conv_name_base + '_x2_scale')(x)
        x = Activation('relu', name=relu_name_base + '_x2')(x)
        x = ZeroPadding2D((1, 1), name=conv_name_base + '_x2_zeropadding')(x)
        x = Convolution2D(nb_filter, 3, 3, name=conv_name_base + '_x2', bias=False)(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        return x

    def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
        ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
            # Arguments
                x: input tensor
                stage: index for dense block
                nb_filter: number of filters
                compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
                dropout_rate: dropout rate
                weight_decay: weight decay factor
        '''

        eps = 1.1e-5
        conv_name_base = 'conv' + str(stage) + '_blk'
        relu_name_base = 'relu' + str(stage) + '_blk'
        pool_name_base = 'pool' + str(stage)

        x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_bn')(x)
        x = Scale(axis=concat_axis, name=conv_name_base + '_scale')(x)
        x = Activation('relu', name=relu_name_base)(x)
        x = Convolution2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False)(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

        return x

    def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4,
                    grow_nb_filters=True):
        ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
            # Arguments
                x: input tensor
                stage: index for dense block
                nb_layers: the number of layers of conv_block to append to the model.
                nb_filter: number of filters
                growth_rate: growth rate
                dropout_rate: dropout rate
                weight_decay: weight decay factor
                grow_nb_filters: flag to decide to allow number of filters to grow
        '''

        eps = 1.1e-5
        concat_feat = x

        for i in range(nb_layers):
            branch = i + 1
            x = DenseNet121.conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
            concat_feat = concatenate([concat_feat, x], axis=concat_axis,
                                      name='concat_' + str(stage) + '_' + str(branch))

            if grow_nb_filters:
                nb_filter += growth_rate

        return concat_feat, nb_filter

class DenseNet161:
    def CreateModel(self, nb_dense_block=4, growth_rate=48, nb_filter=96, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4,
                 classes=NUM_CLASS, weights_path=None):
        '''Instantiate the DenseNet 161 architecture,
            # Arguments
                nb_dense_block: number of dense blocks to add to end
                growth_rate: number of filters to add per dense block
                nb_filter: initial number of filters
                reduction: reduction factor of transition blocks.
                dropout_rate: dropout rate
                weight_decay: weight decay factor
                classes: optional number of classes to classify images
                weights_path: path to pre-trained weights
            # Returns
                A Keras model instance.
        '''
        eps = 1.1e-5

        # compute compression factor
        compression = 1.0 - reduction

        # Handle Dimension Ordering for different backends
        global concat_axis
        if K.image_dim_ordering() == 'tf':
            concat_axis = 3
            img_input = Input(shape=(224, 224, 3), name='data')
        else:
            concat_axis = 1
            img_input = Input(shape=(3, 224, 224), name='data')

        # From architecture for ImageNet (Table 1 in the paper)
        nb_filter = 96
        nb_layers = [6, 12, 36, 24]  # For DenseNet-161

        # Initial convolution
        x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
        x = Convolution2D(nb_filter, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
        x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
        x = Scale(axis=concat_axis, name='conv1_scale')(x)
        x = Activation('relu', name='relu1')(x)
        x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

        # Add dense blocks
        for block_idx in range(nb_dense_block - 1):
            stage = block_idx + 2
            x, nb_filter = DenseNet161.dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate,
                                       dropout_rate=dropout_rate, weight_decay=weight_decay)

            # Add transition_block
            x = DenseNet161.transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate,
                                 weight_decay=weight_decay)
            nb_filter = int(nb_filter * compression)

        final_stage = stage + 1
        x, nb_filter = DenseNet161.dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)

        x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv' + str(final_stage) + '_blk_bn')(x)
        x = Scale(axis=concat_axis, name='conv' + str(final_stage) + '_blk_scale')(x)
        x = Activation('relu', name='relu' + str(final_stage) + '_blk')(x)
        x = GlobalAveragePooling2D(name='pool' + str(final_stage))(x)

        x = Dense(classes, name='fc6')(x)
        x = Activation('softmax', name='prob')(x)

        model = Model(img_input, x, name='densenet')

        if weights_path is not None:
            model.load_weights(weights_path)

        return model

    def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
        '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
            # Arguments
                x: input tensor
                stage: index for dense block
                branch: layer index within each dense block
                nb_filter: number of filters
                dropout_rate: dropout rate
                weight_decay: weight decay factor
        '''
        eps = 1.1e-5
        conv_name_base = 'conv' + str(stage) + '_' + str(branch)
        relu_name_base = 'relu' + str(stage) + '_' + str(branch)

        # 1x1 Convolution (Bottleneck layer)
        inter_channel = nb_filter * 4
        x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x1_bn')(x)
        x = Scale(axis=concat_axis, name=conv_name_base + '_x1_scale')(x)
        x = Activation('relu', name=relu_name_base + '_x1')(x)
        x = Convolution2D(inter_channel, 1, 1, name=conv_name_base + '_x1', bias=False)(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        # 3x3 Convolution
        x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x2_bn')(x)
        x = Scale(axis=concat_axis, name=conv_name_base + '_x2_scale')(x)
        x = Activation('relu', name=relu_name_base + '_x2')(x)
        x = ZeroPadding2D((1, 1), name=conv_name_base + '_x2_zeropadding')(x)
        x = Convolution2D(nb_filter, 3, 3, name=conv_name_base + '_x2', bias=False)(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        return x

    def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
        ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
            # Arguments
                x: input tensor
                stage: index for dense block
                nb_filter: number of filters
                compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
                dropout_rate: dropout rate
                weight_decay: weight decay factor
        '''

        eps = 1.1e-5
        conv_name_base = 'conv' + str(stage) + '_blk'
        relu_name_base = 'relu' + str(stage) + '_blk'
        pool_name_base = 'pool' + str(stage)

        x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_bn')(x)
        x = Scale(axis=concat_axis, name=conv_name_base + '_scale')(x)
        x = Activation('relu', name=relu_name_base)(x)
        x = Convolution2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False)(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

        return x

    def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4,
                    grow_nb_filters=True):
        ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
            # Arguments
                x: input tensor
                stage: index for dense block
                nb_layers: the number of layers of conv_block to append to the model.
                nb_filter: number of filters
                growth_rate: growth rate
                dropout_rate: dropout rate
                weight_decay: weight decay factor
                grow_nb_filters: flag to decide to allow number of filters to grow
        '''

        eps = 1.1e-5
        concat_feat = x

        for i in range(nb_layers):
            branch = i + 1
            x = DenseNet161.conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
            concat_feat = concatenate([concat_feat, x], axis=concat_axis,
                                name='concat_' + str(stage) + '_' + str(branch))

            if grow_nb_filters:
                nb_filter += growth_rate

        return concat_feat, nb_filter

class DenseNet169:
    def CreateModel(self, nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4,
                 classes=NUM_CLASS, weights_path=None):
        '''Instantiate the DenseNet architecture,
            # Arguments
                nb_dense_block: number of dense blocks to add to end
                growth_rate: number of filters to add per dense block
                nb_filter: initial number of filters
                reduction: reduction factor of transition blocks.
                dropout_rate: dropout rate
                weight_decay: weight decay factor
                classes: optional number of classes to classify images
                weights_path: path to pre-trained weights
            # Returns
                A Keras model instance.
        '''
        eps = 1.1e-5

        # compute compression factor
        compression = 1.0 - reduction

        # Handle Dimension Ordering for different backends
        global concat_axis
        if K.image_dim_ordering() == 'tf':
            concat_axis = 3
            img_input = Input(shape=(224, 224, 3), name='data')
        else:
            concat_axis = 1
            img_input = Input(shape=(3, 224, 224), name='data')

        # From architecture for ImageNet (Table 1 in the paper)
        nb_filter = 64
        nb_layers = [6, 12, 32, 32]  # For DenseNet-169

        # Initial convolution
        x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
        x = Convolution2D(nb_filter, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
        x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
        x = Scale(axis=concat_axis, name='conv1_scale')(x)
        x = Activation('relu', name='relu1')(x)
        x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

        # Add dense blocks
        for block_idx in range(nb_dense_block - 1):
            stage = block_idx + 2
            x, nb_filter = DenseNet169.dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate,
                                       dropout_rate=dropout_rate, weight_decay=weight_decay)

            # Add transition_block
            x = DenseNet169.transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate,
                                 weight_decay=weight_decay)
            nb_filter = int(nb_filter * compression)

        final_stage = stage + 1
        x, nb_filter = DenseNet169.dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)

        x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv' + str(final_stage) + '_blk_bn')(x)
        x = Scale(axis=concat_axis, name='conv' + str(final_stage) + '_blk_scale')(x)
        x = Activation('relu', name='relu' + str(final_stage) + '_blk')(x)
        x = GlobalAveragePooling2D(name='pool' + str(final_stage))(x)

        x = Dense(classes, name='fc6')(x)
        x = Activation('softmax', name='prob')(x)

        model = Model(img_input, x, name='densenet')

        if weights_path is not None:
            model.load_weights(weights_path)

        return model

    def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
        '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
            # Arguments
                x: input tensor
                stage: index for dense block
                branch: layer index within each dense block
                nb_filter: number of filters
                dropout_rate: dropout rate
                weight_decay: weight decay factor
        '''
        eps = 1.1e-5
        conv_name_base = 'conv' + str(stage) + '_' + str(branch)
        relu_name_base = 'relu' + str(stage) + '_' + str(branch)

        # 1x1 Convolution (Bottleneck layer)
        inter_channel = nb_filter * 4
        x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x1_bn')(x)
        x = Scale(axis=concat_axis, name=conv_name_base + '_x1_scale')(x)
        x = Activation('relu', name=relu_name_base + '_x1')(x)
        x = Convolution2D(inter_channel, 1, 1, name=conv_name_base + '_x1', bias=False)(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        # 3x3 Convolution
        x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x2_bn')(x)
        x = Scale(axis=concat_axis, name=conv_name_base + '_x2_scale')(x)
        x = Activation('relu', name=relu_name_base + '_x2')(x)
        x = ZeroPadding2D((1, 1), name=conv_name_base + '_x2_zeropadding')(x)
        x = Convolution2D(nb_filter, 3, 3, name=conv_name_base + '_x2', bias=False)(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        return x

    def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
        ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
            # Arguments
                x: input tensor
                stage: index for dense block
                nb_filter: number of filters
                compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
                dropout_rate: dropout rate
                weight_decay: weight decay factor
        '''

        eps = 1.1e-5
        conv_name_base = 'conv' + str(stage) + '_blk'
        relu_name_base = 'relu' + str(stage) + '_blk'
        pool_name_base = 'pool' + str(stage)

        x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_bn')(x)
        x = Scale(axis=concat_axis, name=conv_name_base + '_scale')(x)
        x = Activation('relu', name=relu_name_base)(x)
        x = Convolution2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False)(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

        return x

    def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4,
                    grow_nb_filters=True):
        ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
            # Arguments
                x: input tensor
                stage: index for dense block
                nb_layers: the number of layers of conv_block to append to the model.
                nb_filter: number of filters
                growth_rate: growth rate
                dropout_rate: dropout rate
                weight_decay: weight decay factor
                grow_nb_filters: flag to decide to allow number of filters to grow
        '''

        eps = 1.1e-5
        concat_feat = x

        for i in range(nb_layers):
            branch = i + 1
            x = DenseNet169.conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
            concat_feat = concatenate([concat_feat, x], axis=concat_axis,
                                name='concat_' + str(stage) + '_' + str(branch))

            if grow_nb_filters:
                nb_filter += growth_rate

        return concat_feat, nb_filter