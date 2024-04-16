import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import opening_dataset


class DigDefinition():
    def __init__(self, amount_epochs=None, learning_rate=None, metric=None, sgd_sample=None):
        self.weights_input_to_2 = np.random.uniform(-0.5, 0.5, (28, 784))
        self.weights_2_to_output = np.random.uniform(-0.5, 0.5, (10, 28))
        self.biases_input_to_2 = np.zeros((28, 1))
        self.biases_2_to_output = np.zeros((10, 1))
        self.epochs = 3 if amount_epochs == None else amount_epochs

        self.learning_rate = 0.01 if learning_rate == None else learning_rate
        self.metric = "mse" if metric == None else metric
        self.sgd_sample = sgd_sample
        self.random_state = 42
        self.noise = 5

    def training(self):
        # Loading training data
        images, answers = opening_dataset.load_dataset()

        loss = 0
        correct = 0
        for epoch in range(self.epochs):
            for image, answer in zip(images, answers):
                image = np.reshape(image, (-1, 1))  # First layer
                answer = np.reshape(answer, (-1, 1))
                learning_rate = self.get_new_rate(epoch)

                # Transfer into 2-d layer
                z_2 = np.dot(self.weights_input_to_2, image) + self.biases_input_to_2
                layer_2 = 1 / (1 + np.exp(-z_2))  # Normalisation using sigmoid

                # Transfer into output layer
                z_output = np.dot(self.weights_2_to_output, layer_2) + self.biases_2_to_output
                output_layer = 1 / (1 + np.exp(-z_output))  # Normalisation using sigmoid

                # Loss / Error calculation using selected metric (MSE - default)
                temp_loss = getattr(self, '_' + str(self.metric))(output_layer, answer)
                loss += temp_loss
                correct += int(np.argmax(output_layer) == np.argmax(answer))

                # Back propagation algorithms
                # For output layer
                nabla_output = 2 * (output_layer - answer)
                self.weights_2_to_output -= self.dev_weights(layer_2, nabla_output, learning_rate)
                self.biases_2_to_output -= self.dev_biases(nabla_output, learning_rate)

                # For 2-d layer
                dev_layer_2 = layer_2 * (1 - layer_2)
                nabla_layer_2 = np.dot(np.transpose(self.weights_2_to_output), nabla_output) * dev_layer_2
                self.weights_input_to_2 -= self.dev_weights(image, nabla_layer_2, learning_rate)
                self.biases_input_to_2 -= self.dev_biases(nabla_layer_2, learning_rate)

            print(f"Epoch {epoch + 1}")
            print(f"Loss: {round((loss[0] / images.shape[0]) * 100, 2)}%")
            print(f"Accuracy: {round((correct / images.shape[0]) * 100, 2)}%")

            loss = 0
            correct = 0

    # Predict for custom number
    def get_predict(self, test_image):
        self.training()
        # Test image processing
        grey = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        test_image = 1 - grey(test_image).astype("float32") / 255
        test_image_reshape = np.reshape(test_image, (-1, 1))

        # Forward propagation (to 2 layer)
        z_2 = np.dot(self.weights_input_to_2, test_image_reshape) + self.biases_input_to_2
        layer_2 = 1 / (1 + np.exp(-z_2))  # Normalization by sigmoid

        # Froward propagation (to output)
        z_output = np.dot(self.weights_2_to_output, layer_2) + self.biases_2_to_output
        output_layer = 1 / (1 + np.exp(-z_output))  # Normalization by sigmoid

        # Output
        plt.imshow(test_image.reshape(28, 28), cmap="Greys")
        plt.title(f"Suggests the custom number is: {output_layer.argmax()}")
        plt.show()

    #### DINAMIC LEARNING RATE
    def get_new_rate(self, epoch):
        if type(self.learning_rate) == float:
            return self.learning_rate
        else:
            # Can be set the change function
            new_learning_rate = self.learning_rate(epoch + 1)
            return new_learning_rate

    #### METRICS
    # Mean Squared Error
    def _mse(self, output, answer):
        error = output - answer
        MSE = (1 / len(output)) * np.sum(error ** 2, axis=0)
        return MSE

    # Mean Absolute Error
    def _mae(self, output, answer):
        error = output - answer
        MAE = (1 / len(output)) * np.sum(abs(error), axis=0)
        return MAE

    # Root Mean Squared Error
    def _rmse(self, output, answer):
        error = output - answer
        RMSE = ((1 / len(output)) * np.sum(error ** 2, axis=0)) ** 0.5
        return RMSE

    # Coefficient R^2
    def _r2(self, output, answer):
        error = output - answer
        mean_answer = np.mean(answer)
        R2 = 1 - (np.sum(error ** 2, axis=0)) / (np.sum(answer - mean_answer, axis=0) ** 2)
        return R2

    # Mean Absolute Percentage Error
    def _mape(self, output, answer):
        error = output - answer
        MAPE = (100 / len(output)) * (abs(error / answer))
        return MAPE

    #### DERIVATIVE FUNCTIONS
    # Derivative of the weights
    def dev_weights(self, before_layer, nabla, learning_rate):
        nabla_weights = learning_rate * np.dot(nabla, np.transpose(before_layer))
        return nabla_weights

    # Derivative of the biases
    def dev_biases(self, nabla, learning_rate):
        nabla_biases = learning_rate * nabla
        return nabla_biases


if __name__ == "__main__":
    image = Image.open("digits_images/digit_0.jpg")

    image_resized = image.resize((28, 28))
    image_resized.save("digits_images\\digit_0_28.jpg")

    test_image = plt.imread("digits_images\\digit_0_28.jpg")
    DigDefinition(learning_rate=lambda iter: 0.5 * (0.05 ** iter)).get_predict(test_image)