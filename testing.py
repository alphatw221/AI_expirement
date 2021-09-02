import tensorflow as tf
import tensorflow_addons as tfa

from nfnets.nfnet import NFNet
 
 
model = NFNet(num_classes=2)
model.build((1, 128,128,3))
optimizer = tfa.optimizers.SGDW(learning_rate=0.001, weight_decay=2e-5, momentum=0.9)
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer,loss=loss,)
model.summary()