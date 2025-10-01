import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch.nn as nn

# --- Configuration ---
MODEL_NAME = 'distilbert-base-uncased'
TRAIN_FILE = "contest1_train.csv"
TEST_FILE = "contest1_test.csv"
OUTPUT_FILE = "test_pred.csv"
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
MAX_LEN = 128

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Loading and Preprocessing ---
def load_data(train_file, test_file):
    """Load train and test data from CSV files."""
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    return train_df, test_df

def create_label_maps(df):
    """Create mappings from labels to integers and vice-versa."""
    aspect_categories = df['aspectCategory'].unique()
    polarities = df['polarity'].unique()

    aspect_map = {label: i for i, label in enumerate(aspect_categories)}
    polarity_map = {label: i for i, label in enumerate(polarities)}

    inv_aspect_map = {i: label for label, i in aspect_map.items()}
    inv_polarity_map = {i: label for label, i in polarity_map.items()}

    return aspect_map, polarity_map, inv_aspect_map, inv_polarity_map


# --- Custom Dataset ---
class ABSADataset(Dataset):
    """Custom dataset for Aspect-Based Sentiment Analysis."""
    def __init__(self, df, tokenizer, aspect_map, polarity_map, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.aspect_map = aspect_map
        self.polarity_map = polarity_map
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row.text)
        aspect = row.aspectCategory
        polarity = row.polarity

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'aspect_label': torch.tensor(self.aspect_map[aspect], dtype=torch.long),
            'polarity_label': torch.tensor(self.polarity_map[polarity], dtype=torch.long)
        }


# --- Model Definition ---
class ABSAModel(nn.Module):
    """The multi-task model for ABSA."""
    def __init__(self, n_aspects, n_polarities):
        super(ABSAModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.aspect_classifier = nn.Linear(self.bert.config.dim, n_aspects)
        self.polarity_classifier = nn.Linear(self.bert.config.dim, n_polarities)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use the [CLS] token's hidden state for classification
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        
        aspect_logits = self.aspect_classifier(pooled_output)
        polarity_logits = self.polarity_classifier(pooled_output)
        
        return aspect_logits, polarity_logits


# --- Training and Evaluation ---
def train_epoch(model, data_loader, loss_fn_aspect, loss_fn_polarity, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        aspect_labels = d["aspect_label"].to(device)
        polarity_labels = d["polarity_label"].to(device)

        aspect_logits, polarity_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss_aspect = loss_fn_aspect(aspect_logits, aspect_labels)
        loss_polarity = loss_fn_polarity(polarity_logits, polarity_labels)
        loss = loss_aspect + loss_polarity

        _, aspect_preds = torch.max(aspect_logits, dim=1)
        _, polarity_preds = torch.max(polarity_logits, dim=1)

        correct_predictions += torch.sum((aspect_preds == aspect_labels) & (polarity_preds == polarity_labels))
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn_aspect, loss_fn_polarity, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            aspect_labels = d["aspect_label"].to(device)
            polarity_labels = d["polarity_label"].to(device)

            aspect_logits, polarity_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss_aspect = loss_fn_aspect(aspect_logits, aspect_labels)
            loss_polarity = loss_fn_polarity(polarity_logits, polarity_labels)
            loss = loss_aspect + loss_polarity

            _, aspect_preds = torch.max(aspect_logits, dim=1)
            _, polarity_preds = torch.max(polarity_logits, dim=1)

            correct_predictions += torch.sum((aspect_preds == aspect_labels) & (polarity_preds == polarity_labels))
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

def get_predictions(model, data_loader, device):
    model = model.eval()
    aspect_preds_list = []
    polarity_preds_list = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)

            aspect_logits, polarity_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, aspect_preds = torch.max(aspect_logits, dim=1)
            _, polarity_preds = torch.max(polarity_logits, dim=1)

            aspect_preds_list.extend(aspect_preds.cpu().numpy())
            polarity_preds_list.extend(polarity_preds.cpu().numpy())

    return aspect_preds_list, polarity_preds_list


# --- Main Execution ---
if __name__ == "__main__":
    train_df, test_df = load_data(TRAIN_FILE, TEST_FILE)
    
    # Create label maps from training data
    aspect_map, polarity_map, inv_aspect_map, inv_polarity_map = create_label_maps(train_df)

    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    # Split training data for validation
    df_train, df_val = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['aspectCategory'])

    train_dataset = ABSADataset(df_train, tokenizer, aspect_map, polarity_map, MAX_LEN)
    val_dataset = ABSADataset(df_val, tokenizer, aspect_map, polarity_map, MAX_LEN)
    test_dataset = ABSADataset(test_df, tokenizer, aspect_map, polarity_map, MAX_LEN)

    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    # For test_data_loader, we need to handle the dummy labels not being in the map
    # A simpler approach for prediction is to create a dataset without labels
    class ABSAPredictionDataset(Dataset):
        def __init__(self, texts, tokenizer, max_len):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_len = max_len
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            encoding = self.tokenizer.encode_plus(
                text, add_special_tokens=True, max_length=self.max_len, 
                return_token_type_ids=False, padding='max_length', 
                return_attention_mask=True, return_tensors='pt', truncation=True)
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }

    test_texts = test_df.text.values
    pred_dataset = ABSAPredictionDataset(test_texts, tokenizer, MAX_LEN)
    test_data_loader = DataLoader(pred_dataset, batch_size=BATCH_SIZE)


    # Initialize model
    model = ABSAModel(n_aspects=len(aspect_map), n_polarities=len(polarity_map))
    model = model.to(device)

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Loss Functions
    loss_fn_aspect = nn.CrossEntropyLoss().to(device)
    loss_fn_polarity = nn.CrossEntropyLoss().to(device)

    # Training loop
    best_accuracy = 0
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model, train_data_loader, loss_fn_aspect, loss_fn_polarity, optimizer, device, scheduler, len(df_train)
        )
        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model, val_data_loader, loss_fn_aspect, loss_fn_polarity, device, len(df_val)
        )
        print(f'Val   loss {val_loss} accuracy {val_acc}')

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc

    # Load best model
    model.load_state_dict(torch.load('best_model_state.bin'))

    # Get predictions on test set
    aspect_preds, polarity_preds = get_predictions(model, test_data_loader, device)

    # Map predictions back to labels
    pred_aspect_labels = [inv_aspect_map[pred] for pred in aspect_preds]
    pred_polarity_labels = [inv_polarity_map[pred] for pred in polarity_preds]

    # Create submission file
    submission_df = test_df.copy()
    submission_df['aspectCategory'] = pred_aspect_labels
    submission_df['polarity'] = pred_polarity_labels
    submission_df = submission_df[['id', 'text', 'aspectCategory', 'polarity']]
    submission_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSubmission file '{OUTPUT_FILE}' created successfully!")
    print("Use the following command to evaluate:")
    print(f"python evaluate.py {TRAIN_FILE} {OUTPUT_FILE}")
    print(f"python check_id.py {TEST_FILE} {OUTPUT_FILE}")

