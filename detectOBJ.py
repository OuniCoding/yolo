'''
以下是一些步驟和程式碼範例，說明如何處理自己的影像資料：

1. 資料準備
    收集影像：
        收集大量的玻璃瓶正常影像。
        如果可能，收集一些包含不同瑕疵的影像。
        將影像整理到不同的資料夾中，例如 normal/ 和 defect/。
    影像預處理：
        將影像調整到相同的尺寸（例如 256x256 像素）。
        將影像轉換為灰階或 RGB 格式。
        將像素值縮放到 0 到 1 之間。
2. 使用 ImageDataGenerator 載入影像
    Keras 的 ImageDataGenerator 是一個方便的工具，用於從資料夾中載入影像並進行預處理。
    
需要將 data/train、data/validation 和 data/test 替換為您自己的資料夾路徑
根據影像資料集，調整 img_width、img_height、batch_size、encoding_dim 和 threshold 等參數。
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# 1. 資料準備 (假設您的影像資料夾結構如下)
# data/train/normal/xxx.jpg
# data/validation/normal/xxx.jpg
# data/test/xxx.jpg

# 2. 使用 ImageDataGenerator 載入影像
img_width, img_height = 256, 256
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)

# flow_from_directory() 函式會自動從指定的資料夾中載入影像，並根據資料夾名稱設定標籤。
# color_mode='grayscale'代表灰階，如果圖片是彩色的，改為'rgb'
# class_mode='input'代表將輸入的圖直接作為標籤，因為我們再進行非監督式的學習
train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='input'
)

validation_generator = datagen.flow_from_directory(
    'data/validation',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='input'
)

# 3. 建立和訓練自編碼器模型
encoding_dim = 128

input_img = keras.Input(shape=(img_width, img_height, 1))

# 編碼器部分
encoded = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
encoded = layers.MaxPooling2D((2, 2))(encoded)
encoded = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
encoded = layers.MaxPooling2D((2, 2))(encoded)
# 扁平化特徵圖
flat_encoded = layers.Flatten()(encoded)
# 瓶頸層
bottleneck = layers.Dense(encoding_dim, activation='relu')(flat_encoded)

# 解碼器部分
# 將瓶頸層的輸出 reshape 回到卷積層所需的形狀
reshaped_bottleneck = layers.Reshape((img_width // 4, img_height // 4, 64))(bottleneck) # 根據MaxPooling2D調整
decoded = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(reshaped_bottleneck)
decoded = layers.UpSampling2D((2, 2))(decoded)
decoded = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(decoded)
decoded = layers.UpSampling2D((2, 2))(decoded)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# 4. 異常檢測
test_datagen = ImageDataGenerator(rescale=1./255)
# flow_from_directory() 函式會自動從指定的資料夾中載入影像，並根據資料夾名稱設定標籤。
test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(img_width, img_height),
    batch_size=1, # 測試時，batch_size 設為 1
    color_mode='grayscale',
    class_mode='input',
    shuffle=False # 確保測試結果的順序與檔案列表一致
)

decoded_imgs = autoencoder.predict(test_generator)
loss = np.mean(np.abs(test_generator[0][0] - decoded_imgs), axis=(1, 2, 3)) # 計算每個樣本的平均絕對誤差

# 設定閾值並找出異常,當重建誤差大於該閾值時，則認為該影像為異常影像
threshold = 0.1 # 根據實際情況調整
anomalies = np.where(loss > threshold)

# 5. 結果視覺化
n = min(10, len(anomalies[0])) # 顯示最多10張異常影像
plt.figure(figsize=(20, 4))
for i in range(n):
    index = anomalies[0][i]
    
    # 顯示原始影像
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_generator[0][0][index].reshape(img_width, img_height), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 顯示重建後的影像
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[index].reshape(img_width, img_height), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# 顯示每個像素的異常分數(以第一張異常圖片為例)
if n > 0:
  plt.figure(figsize=(10, 10))
  plt.imshow(np.mean(np.abs(test_generator[0][0][anomalies[0][0]] - decoded_imgs[anomalies[0][0]]), axis=2).reshape(img_width, img_height), cmap='jet')
  plt.colorbar()
  plt.show()

  # 在原始影像上標記異常區域
  image = test_generator[0][0][anomalies[0][0]].reshape(img_width, img_height)
  heatmap = np.mean(np.abs(test_generator[0][0][anomalies[0][0]] - decoded_imgs[anomalies[0][0]]), axis=2).reshape(img_width, img_height)
  heatmap = cv2.resize(heatmap, (img_width, img_height))
  heatmap = heatmap / np.max(heatmap)
  heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
  superimposed_img = heatmap * 0.4 + image * 0.6
  cv2.imwrite('anomaly.jpg', superimposed_img)
  plt.imshow(superimposed_img)
  plt.show()