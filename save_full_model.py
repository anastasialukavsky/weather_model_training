from tensorflow.keras.models import load_model
import tensorflow as tf 

# End up not using the save script but saving it for later
model = load_model('xception_weather_model.h5')

print(model.summary())

# Replace SeparableConv2D layers' incorrect attributes
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.SeparableConv2D):
        layer.depthwise_initializer = layer.kernel_initializer
        layer.pointwise_initializer = layer.kernel_initializer
        layer.depthwise_regularizer = layer.kernel_regularizer
        layer.pointwise_regularizer = layer.kernel_regularizer
        layer.depthwise_constraint = layer.kernel_constraint
        layer.kernel_initializer = None
        layer.kernel_regularizer = None
        layer.kernel_constraint = None

model.save('xception_weather_model_updated.h5')











