<h1 align="center">VisionVerse</h1>
The project is an image captioning model using DenseNet201 and LSTM, generating descriptive captions for images from the Flickr8k dataset. Utilized TensorFlow, Keras, and custom data generators for feature extraction and sequence prediction.

## Execution Guide:
1. Run the following command line in the terminal:
   ```
   pip install numpy pandas tqdm matplotlib scikit-learn tensorflow keras
   ```
   
2. Download the dataset (link to the dataset: **https://www.kaggle.com/datasets/adityajn105/flickr8k**)

3. Upon running all the cells it outputs a file named `model.keras` (it stores the trained model)

4. Enter the path of the image whose caption you want to generate

5. The code then generates a caption according to the model
  
## Accuracy & Loss Over Epochs:

![image](https://github.com/user-attachments/assets/f29900b0-40fb-4ca6-a080-0fd0e421eeed)

![image](https://github.com/user-attachments/assets/7989c24f-adf9-4f2c-8512-aa4776a73947)

## Model Prediction:

![image](https://github.com/user-attachments/assets/fde528fa-b830-4097-902c-d9bb53bf49d4)

## Overview:
The provided code implements an image captioning model that uses a combination of Convolutional Neural Networks (CNN) for feature extraction and Recurrent Neural Networks (RNNs), specifically LSTMs, for generating captions. Here's a step-by-step breakdown of what each section does:

1. **Import Libraries:**
   - Essential libraries for data manipulation (`pandas`, `numpy`), visualization (`matplotlib`, `seaborn`), and deep learning (`tensorflow`, `keras`).
   - Libraries for handling image and text data, model building, and performance evaluation are included.

2. **Dataset Loading:**
   - The code uses the `flickr8k` dataset, a popular image-caption dataset, downloaded from Kaggle.
   - The dataset is unzipped, and images and captions are read from the `captions.txt` file. Each image has an associated caption describing its content.

3. **Image Display:**
   - A function `display_images` is defined to show a sample of images with their corresponding captions.
   - The captions are wrapped to make them readable when displayed along with the image.

4. **Text Preprocessing:**
   - The captions are preprocessed by converting them to lowercase, removing non-alphabetical characters, and tokenizing the text.
   - The captions are wrapped with the tokens `startseq` and `endseq` to signify the start and end of a sentence, which is essential for training the model.

5. **Tokenizer Setup:** A tokenizer is fitted on the processed captions to convert words into integers, and the vocabulary size and maximum caption length are determined.

6. **Data Splitting:**
   - The dataset is split into training and validation sets (80% for training and 20% for validation).
   - A custom data generator is created to yield batches of data, which include image features and caption sequences. The generator also performs the necessary padding for caption sequences.

7. **Custom Data Generator:**
   - `CustomDataGenerator` is a subclass of `keras.utils.Sequence` that handles the loading and preprocessing of images and captions in batches.
   - For each image, the corresponding feature vector is extracted using a pre-trained CNN (in this case, DenseNet201).
   - For each caption, sequences are generated and padded, with the target word being converted into a one-hot encoded vector.

8. **Feature Extraction:**
   - The pre-trained DenseNet201 model is used as a feature extractor for the images.
   - Features are extracted from the second-to-last layer of DenseNet201 and stored in a dictionary, where the key is the image filename and the value is the corresponding feature vector.

9. **Model Architecture:**
   - The model consists of two inputs:
     - **Image features:** Processed by a fully connected layer and reshaped.
     - **Caption sequence:** Embedding layer followed by an LSTM to process the sequential nature of the caption.
   - The features from both image and caption are merged and passed through a series of dense layers with Dropout to prevent overfitting, finally outputting a softmax layer that predicts the next word in the caption.

10. **Model Compilation:** The model is compiled with categorical cross-entropy loss and the Adam optimizer, which is suitable for multi-class classification tasks like word prediction in captions.

11. **Model Training:**
   - The model is trained with callbacks for early stopping, learning rate reduction, and saving the best model based on validation loss.
   - The training process is done for up to 50 epochs, using the custom data generator to feed the model with batches of images and captions.

12. **Model Evaluation:** The model's performance is evaluated by printing the best validation accuracy and plotting training/validation accuracy and loss curves over epochs.

13. **Model Prediction:** The `predict_caption` function generates a caption for a given image. It starts with the token `startseq` and predicts one word at a time using the trained model, stopping when the token `endseq` is predicted or when the maximum caption length is reached.

14. **Image and Caption Display:** Finally, a function `display_image_and_caption` is used to display a predicted caption alongside the corresponding image.

### Overview of Key Concepts:
- **Image Captioning:** The model combines CNNs and RNNs to generate captions for images. The CNN extracts image features, while the RNN (LSTM) generates the caption from these features.
- **Pretrained Model:** DenseNet201 is used for feature extraction from images, leveraging its power to understand visual content.
- **Data Generator:** Custom data generator handles large datasets by yielding data in batches, ensuring efficient memory usage.
- **Text Processing:** Tokenization and sequence padding are essential for preparing captions for training the RNN.
- **Model Evaluation:** The model's accuracy and loss are tracked and visualized for performance monitoring during training.

The code creates a fully functional image captioning system that can generate captions for new images based on a pre-trained model.
