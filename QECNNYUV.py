import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from YUV_RGB import yuv2rgb
import tensorflow
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.layers import Conv2D,UpSampling2D
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D,Activation, concatenate
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, AdamW
from tensorflow.keras.callbacks import ModelCheckpoint

#Frame size of training data
w=480
h=320
#patch size and petch step for training
patchsize = 40
patchstep = 20

#test folders for raw and compressed in yuv and png formats
testfolderRawYuv = './testrawyuv/'
testfolderRawPng = './testrawpng/'
testfolderCompYuv = './testcompyuv/'
testfolderCompPng = './testcomppng/'

#train folders for raw and compressed in yuv and png formats
trainfolderRawYuv = './trainrawyuv/'
trainfolderRawPng = './trainrawpng/'
trainfolderCompYuv = './traincompyuv/'
trainfolderCompPng = './traincomppng/'

for p in [testfolderRawPng, testfolderCompPng, trainfolderRawPng, trainfolderCompPng]:
    if not os.path.exists(p): os.makedirs(p)
        

def cal_psnr(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

def yuv2rgb (Y,U,V,fw,fh):
    U_new = cv2.resize(U, (fw, fh),cv2.INTER_CUBIC)
    V_new = cv2.resize(V, (fw, fh), cv2.INTER_CUBIC)
    U = U_new
    V = V_new
    Y = Y
    rf = Y + 1.4075 * (V - 128.0)
    gf = Y - 0.3455 * (U - 128.0) - 0.7169 * (V - 128.0)
    bf = Y + 1.7790 * (U - 128.0)

    for m in range(fh):
        for n in range(fw):
            if (rf[m, n] > 255):
                rf[m, n] = 255
            if (gf[m, n] > 255):
                gf[m, n] = 255
            if (bf[m, n] > 255):
                bf[m, n] = 255
            if (rf[m, n] < 0):
                rf[m, n] = 0
            if (gf[m, n] < 0):
                gf[m, n] = 0
            if (bf[m, n] < 0):
                bf[m, n] = 0
    r = rf
    g = gf
    b = bf
    return r, g, b

def FromFolderYuvToFolderPNG (folderyuv,folderpng,fw,fh):
    dir_list = os.listdir(folderpng)
    for name in dir_list:
        os.remove(folderpng+name)
    fwuv = fw // 2
    fhuv = fh // 2
    Y = np.zeros((fh, fw), np.uint8, 'C')
    U = np.zeros((fhuv, fwuv), np.uint8, 'C')
    V = np.zeros((fhuv, fwuv), np.uint8, 'C')
    #list of patch left-top coordinates
    numdx = (fw-patchsize)//patchstep
    dx = np.zeros(numdx)
    numdy = (fh - patchsize) // patchstep
    dy = np.zeros(numdy)
    for i in range(numdx):
        dx[i]=i*patchstep
    for i in range(numdy):
        dy[i]=i*patchstep
    dx = dx.astype(int)
    dy = dy.astype(int)
    Im = np.zeros((patchsize, patchsize,3))
    dir_list = os.listdir(folderyuv)
    pngframenum = 0
    for name in dir_list:
        fullname = folderyuv + name
        if fullname.endswith('.yuv'):
            fp = open(fullname, 'rb')
            fp.seek(0, 2)  # move the cursor to the end of the file
            size= fp.tell()
            fp.close()
            fp = open(fullname, 'rb')
            frames = (2*size)//(fw*fh*3)
            frames=100
            print(fullname,frames)
            for f in range(frames):
                for m in range(fh):
                    for n in range(fw):
                        Y[m, n] = ord(fp.read(1))
                for m in range(fhuv):
                    for n in range(fwuv):
                        U[m, n] = ord(fp.read(1))
                for m in range(fhuv):
                    for n in range(fwuv):
                        V[m, n] = ord(fp.read(1))
                r,g,b = yuv2rgb (Y,U,V,fw,fh)
                for i in range(numdx):
                    for j in range(numdy):
                        Im[:, :, 0] = b[dy[j]:dy[j]+patchsize,dx[i]:dx[i]+patchsize]
                        Im[:, :, 1] = g[dy[j]:dy[j]+patchsize,dx[i]:dx[i]+patchsize]
                        Im[:, :, 2] = r[dy[j]:dy[j]+patchsize,dx[i]:dx[i]+patchsize]
                        pngfilename = "%s/%i.png" % (folderpng,pngframenum)
                        cv2.imwrite(pngfilename, Im)
                        pngframenum = pngframenum + 1
            fp.close()
    return (pngframenum-1)

#reads all images from folder and puts them into x array
def LoadImagesFromFolder (foldername):
    dir_list = os.listdir(foldername)
    N = 0
    Nmax = 0
    for name in dir_list:
        fullname = foldername + name
        Nmax = Nmax + 1

    x = np.zeros([Nmax, patchsize, patchsize, 3])
    N = 0
    for name in dir_list:
        fullname = foldername + name
        I1 = cv2.imread(fullname)
        x[N, :, :, 0] = I1[:, :, 2]
        x[N, :, :, 1] = I1[:, :, 1]
        x[N, :, :, 2] = I1[:, :, 0]
        N = N + 1
    return x

def psnr(y_true, y_pred):
    # Вычисляем MSE (Mean Squared Error)
    mse = tensorflow.reduce_mean(tensorflow.square(y_true - y_pred))
    # Задаем максимальное значение пикселя (например, для изображений с нормализацией от 0 до 1 это 1.0)
    max_pixel_value = 1.0
    # Вычисляем PSNR
    psnr = 10.0 * tensorflow.math.log((max_pixel_value ** 2) / mse) / tensorflow.math.log(10.0)
    return psnr


def EnhancerModel(fw, fh):
    comp_tensor = layers.Input(shape=(fh, fw, 3))  # 输入图像

    # 第一个卷积块
    conv_1 = layers.Conv2D(filters=128, kernel_size=(9, 9), padding="same", name='conv_1')(comp_tensor)
    conv_1 = layers.PReLU(shared_axes=[1, 2], name='prelu_1')(conv_1)
    
    # 第二个卷积块
    conv_2 = layers.Conv2D(filters=64, kernel_size=(7, 7), padding="same", name='conv_2')(conv_1)
    conv_2 = layers.PReLU(shared_axes=[1, 2], name='prelu_2')(conv_2)

    # 使用空洞卷积来增大感受野
    conv_3 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", dilation_rate=2, name='conv_3')(conv_2)
    conv_3 = layers.PReLU(shared_axes=[1, 2], name='prelu_3')(conv_3)
    
    # 1x1卷积降维
    conv_4 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding="same", name='conv_4')(conv_3)
    conv_4 = layers.PReLU(shared_axes=[1, 2], name='prelu_4')(conv_4)

    # 通过使用残差学习来改善性能
    conv_11 = layers.Conv2D(filters=128, kernel_size=(9, 9), padding="same", name='conv_11')(comp_tensor)
    conv_11 = layers.PReLU(shared_axes=[1, 2], name='prelu_11')(conv_11)
    
    # 多尺度特征融合
    feat_11 = layers.concatenate([conv_1, conv_11], axis=-1)
    conv_22 = layers.Conv2D(filters=64, kernel_size=(7, 7), padding="same", name='conv_22')(feat_11)
    conv_22 = layers.PReLU(shared_axes=[1, 2], name='prelu_22')(conv_22)
    
    feat_22 = layers.concatenate([conv_2, conv_22], axis=-1)
    conv_33 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", name='conv_33')(feat_22)
    conv_33 = layers.PReLU(shared_axes=[1, 2], name='prelu_33')(conv_33)
    
    feat_33 = layers.concatenate([conv_3, conv_33], axis=-1)
    conv_44 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding="same", name='conv_44')(feat_33)
    conv_44 = layers.PReLU(shared_axes=[1, 2], name='prelu_44')(conv_44)
    
    feat_44 = layers.concatenate([conv_4, conv_44], axis=-1)

    # 输出层
    conv_10 = layers.Conv2D(filters=3, kernel_size=(5, 5), padding="same", name='conv_out')(feat_44)

    # 使用残差连接
    output_tensor = layers.Add()([comp_tensor, conv_10])

    # 创建模型
    enhancer = Model(inputs=comp_tensor, outputs=output_tensor)
    return enhancer


def TrainImageEnhancementModel (folderRaw,folderComp,folderRawVal,folderCompVal):
    print('Loading raw train images...')
    Xraw = LoadImagesFromFolder(folderRaw)
    print('Loading compressed train images...')
    Xcomp = LoadImagesFromFolder(folderComp)
    Xraw = Xraw/255.0
    Xcomp = Xcomp/255.0

    print('Loading raw validiation images...')
    XrawVal = LoadImagesFromFolder(folderRawVal)
    print('Loading compressed validiation images...')
    XcompVal = LoadImagesFromFolder(folderCompVal)
    XrawVal = XrawVal / 255.0
    XcompVal = XcompVal / 255.0
    enhancer = EnhancerModel (patchsize,patchsize)
    #learning_rate_schedule = tensorflow.keras.optimizers.schedules.ExponentialDecay(
    #    initial_learning_rate=0.001,
    #    decay_steps=300,
    #    decay_rate=0.96)
    #optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
    optimizer = tensorflow.keras.optimizers.Adam()
    print(optimizer.get_config())
    #{'name': 'adam', 'learning_rate': 0.0010000000474974513, 'weight_decay': None, 'clipnorm': None,
    # 'global_clipnorm': None, 'clipvalue': None, 'use_ema': False, 'ema_momentum': 0.99,
    # 'ema_overwrite_frequency': None, 'loss_scale_factor': None, 'gradient_accumulation_steps': None, 'beta_1': 0.9,
    # 'beta_2': 0.999, 'epsilon': 1e-07, 'amsgrad': False}

    optimizer = tensorflow.keras.optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        weight_decay=None,
        clipnorm=None,
        global_clipnorm=None,
        clipvalue=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        amsgrad=False)
    print(optimizer.get_config())
    enhancer.compile(loss='mean_squared_error',optimizer=optimizer,metrics=[psnr])
    #enhancer.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[psnr])

    # Путь для сохранения модели
    checkpoint_filepath = 'best_model.weights.h5'

    # 检查是否有预训练权重
    if os.path.exists(checkpoint_filepath):
        print(f"Loading pretrained weights from {checkpoint_filepath}...")
        enhancer.load_weights(checkpoint_filepath)
    else:
        print("No pretrained weights found. Training from scratch...")

    # Определяем колбэк для сохранения наилучшей модели
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,  # куда сохраняем веса
        monitor='val_loss',  # метрика для отслеживания
        save_best_only=True,  # сохраняем только наилучшие веса
        save_weights_only=True,  # сохраняем только веса (не архитектуру)
        mode='min',  # минимизируем значение метрики
        verbose=1  # вывод информации в процессе обучения
    )

    NumEpochs=200
    #enhancer.load_weights('enhancer.weights.h5')
    #with tensorflow.device('gpu'):
    hist = enhancer.fit(Xcomp, Xraw, epochs=NumEpochs, batch_size=128, verbose=1,
                            validation_data=(XcompVal, XrawVal),callbacks=[checkpoint_callback])
    enhancer.save_weights('enhancer.weights.h5')

    return enhancer

def InferenceImageEnhancementModel (fw,fh, model_path):
    enhancer = EnhancerModel (fw,fh)
    enhancer.compile(loss='mean_squared_error',optimizer='Adam',metrics=[psnr])
    enhancer.load_weights(model_path)

    return enhancer


def GetRGBFrame (folderyuv,VideoNumber,FrameNumber,fw,fh):
    fwuv = fw // 2
    fhuv = fh // 2
    Y = np.zeros((fh, fw), np.uint8, 'C')
    U = np.zeros((fhuv, fwuv), np.uint8, 'C')
    V = np.zeros((fhuv, fwuv), np.uint8, 'C')

    dir_list = os.listdir(folderyuv)
    v=0
    for name in dir_list:
        fullname = folderyuv + name
        if v!=VideoNumber:
            v = v + 1
            continue
        if fullname.endswith('.yuv'):
            fp = open(fullname, 'rb')
            fp.seek(0, 2)  # move the cursor to the end of the file
            size = fp.tell()
            fp.close()
            fp = open(fullname, 'rb')
            frames = (2 * size) // (fw * fh * 3)
            for f in range(frames):
                for m in range(fh):
                    for n in range(fw):
                        Y[m, n] = ord(fp.read(1))
                for m in range(fhuv):
                    for n in range(fwuv):
                        U[m, n] = ord(fp.read(1))
                for m in range(fhuv):
                    for n in range(fwuv):
                        V[m, n] = ord(fp.read(1))
                if f==FrameNumber:
                    r, g, b = yuv2rgb(Y, U, V, fw, fh)
                    return r,g,b

def GetEngancedRGB (RGBin,fw,fh):
    RGBin = np.expand_dims(RGBin, axis=0)
    EnhancedPatches = enhancer.predict(RGBin)
    EnhancedPatches=np.squeeze(EnhancedPatches, axis=0)
    return EnhancedPatches

def ShowOneFrameEnhancement(folderyuvraw,foldercomp,VideoIndex,FrameIndex):
    r1, g1, b1 = GetRGBFrame(folderyuvraw,VideoIndex, FrameIndex, w, h)
    RGBRAW = np.zeros((h, w, 3))
    RGBRAW[:, :, 0] = r1
    RGBRAW[:, :, 1] = g1
    RGBRAW[:, :, 2] = b1

    r2, g2, b2 = GetRGBFrame(foldercomp, VideoIndex, FrameIndex, w, h)
    RGBCOMP = np.zeros((h, w, 3))
    RGBCOMP[:, :, 0] = r2
    RGBCOMP[:, :, 1] = g2
    RGBCOMP[:, :, 2] = b2

    RGBENH = GetEngancedRGB(RGBCOMP, w, h)

    plt.grid(False)
    plt.gray()
    plt.axis('off')
    plt.subplot(1, 3, 1)
    plt.imshow(RGBRAW / 255.0)
    psnr1 = cal_psnr(RGBRAW / 255.0, RGBCOMP / 255.0)
    psnr2 = cal_psnr(RGBRAW / 255.0, RGBENH / 255.0)

    tit = "%.2f, %.2f" % (psnr1, psnr2)
    plt.title(tit)

    plt.grid(False)
    plt.gray()
    plt.axis('off')
    plt.subplot(1, 3, 2)

    plt.imshow(RGBCOMP / 255.0)

    plt.grid(False)
    plt.gray()
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(RGBENH / 255.0)
    plt.show()

def ShowFramePSNRPerformance (folderyuv,foldercomp,VideoIndex,framesmax,fw,fh):
    RGBRAW = np.zeros((h, w, 3))
    RGBCOMP = np.zeros((h, w, 3))
    dir_list = os.listdir(folderyuv)
    v = 0
    for name in dir_list:
        fullname = folderyuv + name
        print(name)
        if v != VideoIndex:
            v = v + 1
            continue
        if fullname.endswith('.yuv'):
            fp = open(fullname, 'rb')
            fp.seek(0, 2)  # move the cursor to the end of the file
            size = fp.tell()
            fp.close()
            frames = (2 * size) // (fw * fh * 3)
            if frames>framesmax:
                frames = framesmax

            PSNRCOMP = np.zeros((frames))
            PSNRENH = np.zeros((frames))
            for f in range(frames):
                print(f,frames)
                r, g, b = GetRGBFrame(folderyuv, VideoIndex, f, w, h)
                RGBRAW[:, :, 0] = r
                RGBRAW[:, :, 1] = g
                RGBRAW[:, :, 2] = b
                r, g, b = GetRGBFrame(foldercomp, VideoIndex, f, w, h)
                RGBCOMP[:, :, 0] = r
                RGBCOMP[:, :, 1] = g
                RGBCOMP[:, :, 2] = b
                PSNRCOMP[f] = cal_psnr(RGBRAW / 255.0, RGBCOMP / 255.0)
                RGBENH = GetEngancedRGB(RGBCOMP, w, h)
                PSNRENH[f] = cal_psnr(RGBRAW / 255.0, RGBENH / 255.0)
        break

    ind = np.argsort(PSNRCOMP)

    plt.plot(PSNRCOMP[ind], label='Compressed')
    plt.plot(PSNRENH[ind], label='Enhanced')
    plt.xlabel('Frame index')
    plt.ylabel('PSNR, dB')
    plt.grid()
    plt.legend()
    tit = "%s PSNR = [%.2f, %.2f] dB" % (name,np.mean(PSNRCOMP), np.mean(PSNRENH))
    plt.title(tit)
    plt.show()



TrainMode = 0
PrepareDataSetFromYUV=1

if TrainMode==1:
    if PrepareDataSetFromYUV==0:
        FromFolderYuvToFolderPNG (testfolderRawYuv,testfolderRawPng,w,h)
        FromFolderYuvToFolderPNG (testfolderCompYuv,testfolderCompPng,w,h)
        FromFolderYuvToFolderPNG (trainfolderRawYuv,trainfolderRawPng,w,h)
        FromFolderYuvToFolderPNG (trainfolderCompYuv,trainfolderCompPng,w,h)
    TrainImageEnhancementModel(trainfolderRawPng,trainfolderCompPng,testfolderRawPng,testfolderCompPng)



if 1:
    enhancer = InferenceImageEnhancementModel (w,h, "best_model.weights.h5")
    # enhancer = InferenceImageEnhancementModel (w,h, "enhancer.weights.h5")
    ShowOneFrameEnhancement(trainfolderRawYuv,trainfolderCompYuv,0,2)
    ShowOneFrameEnhancement(testfolderRawYuv,testfolderCompYuv,0,2)
    ShowOneFrameEnhancement(trainfolderRawYuv, trainfolderCompYuv, 0, 1)
    ShowOneFrameEnhancement(testfolderRawYuv, testfolderCompYuv, 0, 1)
    # ShowFramePSNRPerformance (trainfolderRawYuv,trainfolderCompYuv,0,20,w,h)
    # ShowFramePSNRPerformance (testfolderRawYuv,testfolderCompYuv,0,20,w,h)







