from data_processing import get_data, train_test_split, get_indices, create_tokenized_data_loaders
from model import SourceClassifier
from train import train
import transformers

def main():
    # get data
    df = get_data()

    # split data
    train_df, test_df = train_test_split(df)

    # get indices
    source_to_idx, idx_to_source = get_indices(df)

    # instantiate tokenizer
    tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # create dataloaders
    train_loader, test_loader = create_tokenized_data_loaders(train_df, test_df, tokenizer, source_to_idx)

    # instantiate model
    model = SourceClassifier(len(source_to_idx))

    # train the model and save model weights after each epoch
    model_save_path = "/path/to/save/model/weights"
    train(model, train_loader, test_loader, model_save_path)

if __name__ == "__main__":
    main()
