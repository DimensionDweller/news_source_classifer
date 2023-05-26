# News Source Classifier

This project is about creating a classifier that can predict the source of a news article given the content of the article. The sources of news we are considering are CNN, Fox News, MSNBC, and Breitbart. I utilize a pre-trained BERT model to extract features from the articles and perform the classification. After 10 epochs of training, the model achieved an accuracy of 0.9514 with a loss of 0.1718.

## Dependencies

Before running this project, you need to install the dependencies. The dependencies for the project are listed in the `requirements.txt` file. You can install them using the following command:

```bash
pip install -r requirements.txt
```

The main dependencies are:

- torch
- transformers
- pandas
- numpy
- scikit-learn
- tqdm

## Pre-trained BERT Model

We are using the DistilBERT model, a transformer-based machine learning model, specifically designed for natural language processing tasks. The DistilBERT model is a smaller version of the BERT model. It has 40% less parameters than `bert-base-uncased` and runs 60% faster, while preserving over 95% of BERT's performances.

The model is trained in an unsupervised manner on a large corpus of text, which allows it to learn a representation of the language. We use the pre-trained model to convert our sentences into vectors (or embeddings). These vectors capture the semantic meaning of the sentences and can be used for various downstream tasks like classification, information extraction, etc.

We use the `transformers` library to load the pre-trained DistilBERT model.

## Project Structure

The project consists of the following files:

- `data_processing.py`: This file is responsible for loading and preprocessing the dataset. It loads the data from a csv file, cleans the text, splits the data into training and testing sets, and converts the data into a format that can be input into the DistilBERT model.

- `model.py`: This file defines the `SourceClassifier` class, which is our classification model. It uses the pre-trained DistilBERT model to extract features from our data and adds a few layers on top to perform the classification.

- `train.py`: This file contains the training loop. It uses the Adam optimizer and cross-entropy loss to train our model. After each epoch, it prints out the training loss and accuracy on the test set.

- `main.py`: This is the main entry point of the application. It loads the data, initializes the model, and starts the training process.

## How to Run

To run this project, follow these steps:

1. Install the dependencies using the command mentioned above.

2. Run the `main.py` script:

```bash
python main.py
```

This will start the training process. After each epoch, the training loss and test accuracy will be printed out.

3. The trained model will be saved in the specified path as defined in `main.py`. You can then use this model for predicting the source of news articles.

## Future Work

While the current model gives promising results, there are several improvements that could be made:

- Use a larger and more diverse dataset: The current dataset only contains articles from four sources. Including articles from more sources could make the model more general and robust.

- Use a larger pre-trained model: Although DistilBERT performs well and is fast to train, larger models like BERT or RoBERTa could potentially give better results.

- Fine-tuning: Instead of training only the top layers of our model, we could fine-tune the entire model on our task. This could potentially give better
