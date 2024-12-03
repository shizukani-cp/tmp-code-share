import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import MllamaForConditionalGeneration, AutoProcessor
from logging import getLogger, Formatter, StreamHandler, DEBUG

# ロガーの設定
logger = getLogger(__name__)
logger.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# WikipediaDatasetの定義
class WikipediaDataset(Dataset):
    def __init__(self, file_path, max_length=512):
        self.file_path = file_path
        self.max_length = max_length
        self.data = self.load_annotations()

    def load_annotations(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][:self.max_length]

# SmallModelの定義
class SmallModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SmallModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ローカルのLlama 3.2モデルのロード
model_path = "path/to/your/local/llama3.2/model"
teacher_model = MllamaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

# SmallModelの初期化
input_size = 768  # 入力サイズ（実際のタスクに合わせて調整）
hidden_size = 256
output_size = teacher_model.config.vocab_size
student_model = SmallModel(input_size, hidden_size, output_size).to("cuda")

# データセットとDataLoaderの準備
dataset = WikipediaDataset("path/to/wiki.txt")
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 損失関数とオプティマイザの設定
criterion = nn.KLDivLoss(reduction='batchmean')
optimizer = optim.Adam(student_model.parameters())

# 学習ループ
num_epochs = 10
temperature = 2.0

for epoch in range(num_epochs):
    for batch in dataloader:
        # 入力の処理
        inputs = processor(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
        
        # 教師モデルの出力を取得
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs).logits
        
        # 生徒モデルの出力を取得
        student_outputs = student_model(inputs.input_ids)
        
        # 知識蒸留損失の計算
        loss = criterion(
            nn.functional.log_softmax(student_outputs / temperature, dim=-1),
            nn.functional.softmax(teacher_outputs / temperature, dim=-1)
        )
        
        # 逆伝播と最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    logger.debug(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# モデルの保存
torch.save(student_model.state_dict(), "small_model.pth")

logger.debug("モデルが保存されました。")
