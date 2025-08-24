import torch
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from torchmetrics import SpearmanCorrCoef, Accuracy, F1Score
from dataset import SemevalDataset
from model import MultiLabelModel

from peft import get_peft_model, LoraConfig, TaskType
from peft.utils.other import prepare_model_for_kbit_training

# 將資料打包成tensor
# complete_text: 輸入模型的資料，text+summary
# infer_text: 用於validation的資料，只有text
def get_tensor(sample):
    input_text = [(d['premise'], d['hypothesis']) for d in sample]
    labels1 = torch.tensor([d['relatedness_score'] for d in sample]).to(device)
    labels2 = torch.tensor([d['entailment_judgment'] for d in sample]).to(device)
    input_text = tokenizer.batch_encode_plus(input_text, padding=True, truncation=True, return_tensors="pt")
    input_text = {k: input_text[k].to(device) for k in input_text}
    return input_text, labels1, labels2

# 設定參數
MODEL_NAME = "microsoft/deberta-large" # "bert-base-uncased"
LR = 3e-5
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 4
SAVE_DIR = "./saved_models/"
# Create the directory if it doesn't exist
if not Path(SAVE_DIR).exists():
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=False)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="./cache/")
data_sample = SemevalDataset(split="train").data[:3]
print(f"Dataset example: \n{data_sample[0]} \n{data_sample[1]} \n{data_sample[2]}")


train_loader = DataLoader(
    SemevalDataset(split="train"),
    collate_fn=get_tensor,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
)
val_loader = DataLoader(
    SemevalDataset(split="validation"),
    collate_fn=get_tensor,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
)
loss_fn1 = torch.nn.MSELoss()
loss_fn2 = torch.nn.CrossEntropyLoss()
spc = SpearmanCorrCoef()
acc = Accuracy(task="multiclass", num_classes=3)
f1 = F1Score(task="multiclass", num_classes=3, average='macro')

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = MultiLabelModel(MODEL_NAME).to(device)

# 配置LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,                          # LoRA矩陣的秩
    lora_alpha=32,                 # LoRA的縮放參數
    lora_dropout=0.1,              # LoRA層的dropout率
    bias="none",                   # 是否包含偏置參數
    target_modules=["query_proj", "key_proj", "value_proj", "output.dense"],  # 要應用LoRA的模塊
)

# 為主幹模型做準備
model = prepare_model_for_kbit_training(model)

# 將模型轉換為PEFT模型
model = get_peft_model(model, peft_config)

# 只訓練LoRA參數
for name, param in model.named_parameters():
    if "lora" not in name and "regression_head" not in name and "classification_head" not in name:
        param.requires_grad = False

# 打印可訓練參數的數量
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"可訓練參數數量: {trainable_params} ({trainable_params/total_params:.2%})")

optimizer = AdamW(model.parameters(), lr = 1e-5)

for ep in range(NUM_EPOCHS):
    pbar = tqdm(train_loader)
    pbar.set_description(f"Training epoch [{ep+1}/{NUM_EPOCHS}]")
    model.train()
    for input_text, labels1, labels2 in pbar:
        optimizer.zero_grad()
        outputs1, outputs2 = model(**input_text)
        loss = loss_fn1(outputs1, labels1) + loss_fn2(outputs2, labels2)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss = loss.item())

    pbar = tqdm(val_loader)
    pbar.set_description(f"Validation epoch [{ep+1}/{NUM_EPOCHS}]")
    model.eval()
    with torch.no_grad():
        pred1 = torch.tensor([])
        pred2 = torch.tensor([])
        gt1 = torch.tensor([])
        gt2 = torch.tensor([])
        for input_text, labels1, labels2 in pbar:
            outputs1, outputs2 = model(**input_text)
            outputs1 = outputs1.squeeze()
            outputs2 = torch.argmax(outputs2, dim=-1)
            pred1 = torch.cat((pred1, outputs1.to("cpu")), dim=-1)
            pred2 = torch.cat((pred2, outputs2.to("cpu")), dim=-1)
            gt1 = torch.cat((gt1, labels1.to("cpu")), dim=-1)
            gt2 = torch.cat((gt2, labels2.to("cpu")), dim=-1)

    print(f"Spearman Corr: {spc(pred1, gt1)} \nAccuracy: {acc(pred2, gt2)} \nF1 Score: {f1(pred2, gt2)}")
    torch.save(model, f'./saved_models/ep{ep}.ckpt')
    print(f"Model saved to {SAVE_DIR}ep{ep}.ckpt!")