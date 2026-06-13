from tensorflow import keras   # tensorflow and keras (which is built on tensorflow) it's the simpler, friendlier layer that lets you build neural networks without writing complex math yourself.

#  Load & preprocess data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()     # MNIST comes pre-split inside Keras 60k for training and 10k for testing


X_train = X_train / 255.0
X_test = X_test/ 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# CNN — replaces it entirely
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),  # input_shape changes here
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation='softmax')
])

# Compile & Train
model.compile(optimizer='adam',
              loss= 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

#  Before training you have to compile — this sets up how the training process works.

# Loss — measures how wrong the model is after each prediction
# Optimizer — uses that wrongness to adjust weights and make the model less wrong next time
# Metrics — shows you human-readable progress while training runs

print(" Training model, please wait...")
model.fit(X_train, y_train, epochs=15, batch_size=32,
          validation_split=0.1, verbose=1)
# epochs=15 — the model sees the entire 60,000 training images 15 times. Each pass it gets a little better.
# batch_size=32 — doesn't process all 60,000 at once. Processes 32 images, adjusts weights, processes next 32, and so on. More efficient and actually trains better than doing all at once.
# validation_split=0.1 — holds back 10% of training data (6,000 images) and checks accuracy on them after each epoch. This data is not learned from — just used to monitor if the model is actually improving on unseen data during training.
# verbose=1 — print progress to terminal while training.

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n Test Accuracy: {test_acc * 100:.2f}%")
# After training is done, runs the model on the 10,000 test images it has never seen at all — not even for validation. This is the real accuracy number.


model.save('digit_recognizer.h5')
print(" Model saved!!!")
