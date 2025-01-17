import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import glob
import os

class DIGITS_CLASSIFIER_LITE():
    def __init__(self, model_path):
        try:
            print(f"Loading model from {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            interpreter = tflite.Interpreter(model_path)
            interpreter.allocate_tensors()
            self.digits_classifier = interpreter.get_signature_runner('serving_default')
            print("Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TensorFlow Lite interpreter: {e}")

    def preprocess_image(self, images):
        print("Preprocessing images")
        gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
        resized_images = np.array([cv2.resize(gray_image, (32, 32)) for gray_image in gray_images])
        normalized_images = resized_images.astype('float32') / 255.0
        return np.expand_dims(normalized_images, axis=-1)
    
    def postprocess_predictions(self, predictions):
        print("Postprocessing predictions")
        predictions = predictions['tf.stack']
        predicted_labels = [''.join(map(str, row)) for row in np.argmax(predictions, axis=2)]
        predicted_confs = np.min(np.max(predictions, axis=2), axis=1)
        return predicted_labels, predicted_confs

    def predict(self, images):
        print("Predicting")
        preprocessed_images = self.preprocess_image(images)
        predictions = self.digits_classifier(input_1=preprocessed_images)
        return self.postprocess_predictions(predictions)
    
    def predict_without_preprocess(self, preprocessed_images):
        print("Predicting without preprocessing")
        predictions = self.digits_classifier(input_1=preprocessed_images)
        return self.postprocess_predictions(predictions)
    
    def _read_images_from_directory(self, directory):
        print(f"Reading images from directory: {directory}")
        pattern = os.path.join(directory, "*.*")
        png_files = glob.glob(pattern)
        images = []
        for file in png_files:
            print(f"Reading image: {file}")
            image = cv2.imread(file)
            if image is None:
                raise ValueError(f"Failed to read image: {file}")
            images.append(image)
        return images
    
    def predict_from_directory(self, directory):
        images = self._read_images_from_directory(directory)
        return self.predict(images)

# Example usage
if __name__ == "__main__":
    model_path = 'models/svhn_2digits_model.tflite'
    directory = 'custom_images'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    digits_classifier_lite = DIGITS_CLASSIFIER_LITE(model_path)
    # predicted_labels, predicted_confs = digits_classifier_lite.predict_from_directory(directory)
    predicted_labels, predicted_confs = digits_classifier_lite.predict('custom_images/n.png')
    print(predicted_labels, predicted_confs)