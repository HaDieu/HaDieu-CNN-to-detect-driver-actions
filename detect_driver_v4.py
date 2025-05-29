import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ẩn cảnh báo không cần thiết
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Định nghĩa ánh xạ hành vi
activity_map = {'c0': 'Safe driving',
                'c1': 'Texting - right',
                'c2': 'Talking on the phone - right',
                'c3': 'Texting - left',
                'c4': 'Talking on the phone - left',
                'c5': 'Operating the radio',
                'c6': 'Drinking',
                'c7': 'Reaching behind',
                'c8': 'Hair and makeup',
                'c9': 'Talking to passenger'}

# Kiểm tra thư mục dữ liệu
BASE_URL = r"D:\hoc phan\DATN\state-farm-distracted-driver-detection\imgs\train"
if not os.path.exists(BASE_URL):
    raise FileNotFoundError(f"Thư mục dữ liệu không tồn tại: {BASE_URL}")

# Hiển thị một số hình ảnh mẫu
plt.figure(figsize=(12, 20))
image_count = 1
for directory in os.listdir(BASE_URL):
    if directory in activity_map:
        file_list = os.listdir(os.path.join(BASE_URL, directory))
        if file_list:
            plt.subplot(5, 2, image_count)
            image_count += 1
            image = mpimg.imread(os.path.join(BASE_URL, directory, file_list[0]))
            plt.imshow(image)
            plt.title(activity_map[directory])
plt.show()


# Khởi tạo mô hình CNN
classifier = Sequential([
    Conv2D(128, (3, 3), activation='relu', input_shape=(240, 240, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(1024, activation='relu'),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.summary()

# Chuẩn bị dữ liệu huấn luyện
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)

training_set = train_datagen.flow_from_directory(
    BASE_URL,
    target_size=(240, 240),
    batch_size=32,
    subset='training'
)

validation_set = train_datagen.flow_from_directory(
    BASE_URL,
    target_size=(240, 240),
    batch_size=32,
    subset='validation'
)

# Huấn luyện mô hình
history = classifier.fit(training_set,
               steps_per_epoch=len(training_set),
               epochs=10,
               validation_data=validation_set,
               validation_steps=len(validation_set))

# Vẽ biểu đồ Accuracy và Loss qua từng epoch
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Biểu đồ Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Biểu đồ Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()



