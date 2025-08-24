import argparse
import torch
from preprocessing import preprocess_agnews, AGNewsDataset
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import accuracy_score
from tqdm import tqdm



def id_to_word(vocab):
    id2word = {v: k for k, v in vocab.items()}
    return id2word


# 10. 整理 batch 的資料 (定義 function)
def collate_batch(batch):
    # 抽每一個 batch 的第 0 個(注意順序)
    text = [i[0] for i in batch]
    # 進行 padding
    text = pad_sequence(text, batch_first=True)
    
    # 抽每一個 batch 的第 1 個(注意順序)
    label = [i[1] for i in batch]
    # 把每一個 batch 的答案疊成一個 tensor
    label = torch.stack(label)
    
    return text, label


def evaluate(dataloader, model, criterion):
    """定義驗證時的進行流程
    Arguments:
        - dataloader: 具備 mini-batches 的 dataset，由 PyTorch DataLoader 所建立
        - model: 要進行驗證的模型
    Returns:
        - loss: 模型在驗證/測試集的 loss
        - acc: 模型在驗證/測試集的正確率
    """
    # 設定模型的驗證模式
    # 此時 dropout 會自動關閉
    model.eval()
    
    # 設定現在不計算梯度
    with torch.no_grad():
        # 把每個 batch 的 label 儲存成一維 tensor
        y_true = torch.tensor([])
        y_pred = torch.tensor([])

        # 從 dataloader 一次一次抽
        for x, y in dataloader:
            # 把正確的 label concat 起來
            y_true = torch.cat([y_true, y])

            x = x.to(device)
            y = y.to(device)


            logits = model(x)
            # 預測的數值大於 0.5 則視為類別1，反之為類別0
            pred = torch.argmax(logits, dim=-1)
            # 把預測的 label concat 起來
            # 注意: 如果使用 gpu 計算的話，要先用 .cpu 把 tensor 轉回 cpu
            y_pred = torch.cat([y_pred, pred.cpu()])

    # 模型輸出的維度是 (B, 1)，使用.squeeze(-1)可以讓維度變 (B,)
    loss = criterion(y_pred.squeeze(-1), y_true)
    # 計算正確率
    acc = accuracy_score(y_true, y_pred.squeeze(-1))
            
    return loss, acc


parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="agnews")
parser.add_argument("--use_agnews_title", action="store_true")
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--num_epoch", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--dropout_rate", type=float, default=0.2)

args = parser.parse_args()


train_text, train_label, val_text, val_label = preprocess_agnews(
    args.data_name,
    data_type="train",
    use_agnews_title=args.use_agnews_title,
)
test_text, test_label = preprocess_agnews(
    args.data_name,
    data_type="test",
    use_agnews_title=args.use_agnews_title,
)
num_labels = len(set(train_label))
exit()
# build vocab
vocab = {'<pad>':0, '<unk>':1}
for text in train_text:
    for word in word_tokenize(text.lower()):
        if word not in vocab:
            vocab[word] = len(vocab)
print(f"Vocab size: {len(vocab)}")

train_dataset = AGNewsDataset(train_text, train_label, vocab, word_tokenize)
val_dataset = AGNewsDataset(val_text, val_label, vocab, word_tokenize)
test_dataset = AGNewsDataset(test_text, test_label, vocab, word_tokenize)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_batch,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.test_batch_size,
    shuffle=False,
    collate_fn=collate_batch,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=args.test_batch_size,
    shuffle=False,
    collate_fn=collate_batch,
)

class LSTMTextClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout, padding_idx):
        super(LSTMTextClassifier, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding_dropout = torch.nn.Dropout(dropout)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch_size, seq_len]
        E = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        E = self.embedding_dropout(E)
        # packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        H_out, H_n = self.lstm(E)
        tmp = self.fc1(H_out[:, -1, :]+H_out[:, 0, :])
        return self.fc2(tmp)
        # return self.fc(H_out[:, -1, :])  # [batch_size, output_dim]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMTextClassifier(
    vocab_size=len(vocab),
    embedding_dim=300,
    hidden_dim=args.hidden_size,
    output_dim=num_labels,
    dropout=args.dropout_rate,
    padding_idx=vocab['<pad>'],
).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
train_loss = 0
model.train()
for epoch in range(args.num_epoch):
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epoch}")
    for x, y in progress_bar:
        x = x.to(device)
        y = y.to(device)

        # 重新設定模型的梯度
        optimizer.zero_grad()
        
        # 1. 前向傳遞 (Forward Pass)
        logits = model(x)
        # 2. 計算 loss
        loss = criterion(logits, y)
        # 3. 計算反向傳播的梯度
        loss.backward()
        # 4. "更新"模型的權重
        optimizer.step()

        # 一個 epoch 會抽很多次 batch，所以每個 batch 計算完都要加起來
        # .item() 在 PyTorch 中可以獲得該 tensor 的數值
        train_loss += loss.item()
        # print(f"Loss: {loss.item()}")
        # 更新 tqdm 的右側顯示
        progress_bar.set_postfix(loss=loss.item())

    # 計算驗證集的 loss 和正確率
    val_loss, val_acc = evaluate(val_loader, model, criterion)
    print(f"Epoch: {epoch+1}, Val Loss: {val_loss}, Val Acc: {val_acc}")

# 計算測試集的正確率
test_loss, test_acc = evaluate(test_loader, model, criterion)
print(f"Test Loss: {test_loss}, Test Acc: {test_acc}")
