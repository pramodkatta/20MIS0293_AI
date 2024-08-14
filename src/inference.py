import torch
from transformers import BertTokenizer
from model import TrademarkClassifier

def predict_class(description, model, tokenizer, device):
    encoding = tokenizer.encode_plus(
        description,
        truncation=True,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs, dim=1)

    return prediction.item() + 1  # Adding 1 to align with the class_id

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TrademarkClassifier(n_classes=45)
    model.load_state_dict(torch.load('../model/trademark_model.pt'))
    model = model.to(device)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    description = "Example product description for classification"
    class_id = predict_class(description, model, tokenizer, device)
    print(f"Predicted Class ID: {class_id}")
