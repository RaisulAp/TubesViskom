import cv2
import numpy as np
from tensorflow.keras.models import load_model

def load_colorization_model(model_path):
    model = load_model(model_path)
    return model

def preprocess_input(image_path):
    # Load the black-and-white image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize the image to the required input size for the model
    img = cv2.resize(img, (256, 256))
    
    # Normalize pixel values to be in the range [0, 1]
    img = img / 255.0
    
    # Add a batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def postprocess_output(output):
    # Rescale the output to the range [0, 255]
    output = output * 255.0
    
    # Convert to uint8
    output = output.astype(np.uint8)
    
    return output

def colorize_image(model, image_path):
    # Preprocess the input image
    input_img = preprocess_input(image_path)
    
    # Get the colorized output from the model
    colorized_output = model.predict(input_img)
    
    # Postprocess the output
    colorized_output = postprocess_output(colorized_output)
    
    # Display the original and colorized images
    original_img = cv2.imread(image_path)
    cv2.imshow('Original Image', original_img)
    cv2.imshow('Colorized Image', colorized_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace 'path/to/your/model.h5' with the path to your pre-trained model
    model_path = 'path/to/your/model.h5'
    colorization_model = load_colorization_model(model_path)

    # Replace 'path/to/your/image.jpg' with the path to the black-and-white image you want to colorize
    image_path_to_colorize = 'path/to/your/image.jpg'
    colorize_image(colorization_model, image_path_to_colorize)
