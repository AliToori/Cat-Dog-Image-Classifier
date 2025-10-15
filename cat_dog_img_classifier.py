import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class CatDogClassifier:
    """A class to classify cat and dog images using a CNN."""

    def __init__(self, data_dir='cats_and_dogs', batch_size=128, img_height=150, img_width=150, epochs=15):
        """Initialize paths and hyperparameters."""
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, 'train')
        self.validation_dir = os.path.join(data_dir, 'validation')
        self.test_dir = os.path.join(data_dir, 'test')
        self.batch_size = batch_size
        self.test_batch_size = 50  # For test set (50 images)
        self.img_height = img_height
        self.img_width = img_width
        self.epochs = epochs
        self.model = None
        self.history = None
        self.train_data_gen = None
        self.validation_data_gen = None
        self.test_data_gen = None
        self.total_train = 0
        self.total_val = 0
        self.total_test = 0

    def verify_dataset(self):
        """Verify dataset directories and count images."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"Dataset directory '{self.data_dir}' not found. "
                "Please download and unzip 'cats_and_dogs.zip' from "
                "https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip"
            )
        self.total_train = sum(len(files) for r, d, files in os.walk(self.train_dir))
        self.total_val = sum(len(files) for r, d, files in os.walk(self.validation_dir))
        self.total_test = len(os.listdir(self.test_dir))
        print(f"Found {self.total_train} training images, {self.total_val} validation images, "
              f"and {self.total_test} test images.")

    def setup_data_generators(self):
        """Create data generators for train, validation, and test sets."""
        # Training generator with augmentation
        train_image_generator = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=45,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.2,
            zoom_range=0.5,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        self.train_data_gen = train_image_generator.flow_from_directory(
            batch_size=self.batch_size,
            directory=self.train_dir,
            target_size=(self.img_height, self.img_width),
            class_mode='binary'
        )

        # Validation generator (only rescaling)
        validation_image_generator = ImageDataGenerator(rescale=1. / 255)
        self.validation_data_gen = validation_image_generator.flow_from_directory(
            batch_size=self.batch_size,
            directory=self.validation_dir,
            target_size=(self.img_height, self.img_width),
            class_mode='binary'
        )

        # Test generator (no labels, flat directory)
        test_image_generator = ImageDataGenerator(rescale=1. / 255)
        self.test_data_gen = test_image_generator.flow_from_directory(
            batch_size=self.test_batch_size,
            directory=self.test_dir,
            target_size=(self.img_height, self.img_width),
            class_mode=None,
            shuffle=False,
            classes=['.']
        )

    def plot_images(self, images_arr, probabilities=None, output_file=None):
        """Plot images with optional probability labels."""
        fig, axes = plt.subplots(len(images_arr), 1, figsize=(5, len(images_arr) * 3))
        if probabilities is None:
            for img, ax in zip(images_arr, axes):
                ax.imshow(img)
                ax.axis('off')
        else:
            for img, probability, ax in zip(images_arr, probabilities, axes):
                ax.imshow(img)
                ax.axis('off')
                if probability > 50:
                    ax.set_title(f"{probability:.2f}% dog")
                else:
                    ax.set_title(f"{100 - probability:.2f}% cat")
        if output_file:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()

    def build_model(self):
        """Create and compile the CNN model."""
        self.model = Sequential([
            Conv2D(16, 3, padding='same', activation='relu', input_shape=(self.img_height, self.img_width, 3)),
            MaxPooling2D(),
            Dropout(0.2),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Dropout(0.2),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def train_model(self):
        """Train the model and save training history."""
        self.history = self.model.fit(
            self.train_data_gen,
            steps_per_epoch=self.total_train // self.batch_size,
            epochs=self.epochs,
            validation_data=self.validation_data_gen,
            validation_steps=self.total_val // self.batch_size,
            verbose=1
        )

    def plot_training_metrics(self, output_file='training_metrics.png'):
        """Plot training and validation accuracy/loss."""
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        plt.savefig(output_file)
        plt.close()

    def predict_and_evaluate(self, output_file='test_predictions.png'):
        """Predict on test set and evaluate accuracy."""
        # Hardcoded ground truth labels
        answers = [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
                   1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0,
                   1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
                   1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1,
                   0, 0, 0, 0, 0, 0]

        # Predict probabilities
        self.test_data_gen.reset()
        probabilities = (self.model.predict(self.test_data_gen, steps=1, verbose=1).flatten() * 100).astype(int)

        # Evaluate accuracy
        correct = sum(1 for prob, ans in zip(probabilities, answers) if round(prob / 100) == ans)
        percentage_identified = (correct / len(answers)) * 100
        passed_challenge = percentage_identified >= 63

        print(f"Your model correctly identified {percentage_identified:.2f}% of the images of cats and dogs.")
        print("You passed the challenge!" if passed_challenge else
              "You haven't passed yet. Your model should identify at least 63% of the images. Keep trying.")

        # Plot test images with predictions
        self.test_data_gen.reset()
        images = next(self.test_data_gen)
        self.plot_images(images, probabilities, output_file=output_file)

    def run(self):
        """Run the full pipeline."""
        print("Verifying dataset...")
        self.verify_dataset()
        print("Setting up data generators...")
        self.setup_data_generators()
        print("Visualizing sample training images...")
        sample_images, _ = next(self.train_data_gen)
        self.plot_images(sample_images[:5], output_file='sample_images.png')
        print("Visualizing augmented images...")
        augmented_images = [self.train_data_gen[0][0][0] for _ in range(5)]
        self.plot_images(augmented_images, output_file='augmented_images.png')
        print("Building model...")
        self.build_model()
        print("Training model...")
        self.train_model()
        print("Plotting training metrics...")
        self.plot_training_metrics()
        print("Predicting and evaluating on test set...")
        self.predict_and_evaluate()


if __name__ == '__main__':
    classifier = CatDogClassifier()
    classifier.run()