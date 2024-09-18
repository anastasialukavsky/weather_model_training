import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam

# Loading rescaled dataset with ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'dataset',  # Path to dataset folder
    target_size=(299, 299),  # Xception expects 299x299 images
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Explicitly define the input layer for the Xception model
input_layer = Input(shape=(299, 299, 3), name='input_layer')

base_model = Xception(weights='imagenet', include_top=False, input_tensor=input_layer)

# Add new layers for weather classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers to avoid retraining
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Save the model
model.save('xception_weather_model_saved', save_format='tf')
