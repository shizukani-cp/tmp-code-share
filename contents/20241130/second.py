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

# ローカルのLlama 3.2モデルのロード（教師モデル）
model_path = "path/to/your/local/llama3.2/model"
teacher_model = MllamaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

# 既存の生徒モデルの読み込み
student_model = SmallModel(input_size=768, hidden_size=256, output_size=teacher_model.config.vocab_size)
student_model.load_state_dict(torch.load("small_model.pth"))
student_model.eval()  # 教師モデルとして評価モードに設定

# 新しい生徒モデルの初期化
new_student_model = SmallModel(input_size=768, hidden_size=256, output_size=teacher_model.config.vocab_size).to("cuda")

# データセットとDataLoaderの準備
dataset = WikipediaDataset("path/to/wiki.txt")
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 損失関数とオプティマイザの設定
criterion_soft = nn.KLDivLoss(reduction='batchmean')
criterion_hard = nn.CrossEntropyLoss()
optimizer = optim.Adam(new_student_model.parameters())

# 学習ループ
num_epochs = 10
temperature = 2.0
alpha = 0.5  # ソフトターゲットとハードターゲットのバランス

for epoch in range(num_epochs):
    for batch in dataloader:
        # 入力の処理
        inputs = processor(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
        
        # 教師モデル（元の生徒モデル）の出力を取得
        with torch.no_grad():
            teacher_outputs = student_model(inputs.input_ids)

        # 新しい生徒モデルの出力を取得
        new_student_outputs = new_student_model(inputs.input_ids)
        
        # 知識蒸留損失の計算
        loss_soft = criterion_soft(
            nn.functional.log_softmax(new_student_outputs / temperature, dim=-1),
            nn.functional.softmax(teacher_outputs / temperature, dim=-1)
        )
        
        loss_hard = criterion_hard(new_student_outputs.view(-1, teacher_model.config.vocab_size), inputs.input_ids.view(-1))
        
        # 総合損失の計算
        loss = alpha * loss_soft + (1 - alpha) * loss_hard
        
        # 逆伝播と最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    logger.debug(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# 新しい生徒モデルの保存
torch.save(new_student_model.state_dict(), "new_small_model.pth")

logger.debug("新しい生徒モデルが保存されました。")
