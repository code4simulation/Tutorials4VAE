import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# 1. 하이퍼파라미터 설정
latent_dim = 20
num_classes = 10  # MNIST의 라벨 수
batch_size = 128
epochs = 10
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 데이터셋 로드 및 전처리
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 3. Conditional VAE 모델 정의 (MLP 기반)
class MLP_CVAE(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(MLP_CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder: 이미지 + 라벨
        self.encoder = nn.Sequential(
            nn.Linear(784 + num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder: 잠재 벡터 + 라벨
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )
    
    def encode(self, x, label):
        x = x.view(-1, 784)
        label = torch.nn.functional.one_hot(label, num_classes=self.num_classes).float()
        x = torch.cat([x, label], dim=1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, label):
        label = torch.nn.functional.one_hot(label, num_classes=self.num_classes).float()
        z = torch.cat([z, label], dim=1)
        return self.decoder(z).view(-1, 1, 28, 28)
    
    def forward(self, x, label):
        mu, logvar = self.encode(x, label)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, label), mu, logvar

# 4. Conditional VAE 모델 정의 (CNN 기반)
class CNN_CVAE(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(CNN_CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder: 이미지 + 라벨 특징
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.label_projector = nn.Linear(num_classes, 32 * 7 * 7)  # 라벨을 7x7 특징으로 확장
        self.fc_mu = nn.Linear(32 * 7 * 7 + 32 * 7 * 7, latent_dim)  # 64x7x7 특징
        self.fc_logvar = nn.Linear(32 * 7 * 7 + 32 * 7 * 7, latent_dim)
        
        # Decoder: 잠재 벡터 + 라벨 특징
        self.z_projector = nn.Linear(latent_dim, 32 * 7 * 7)  # z를 7x7 특징으로 확장
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 3, stride=2, padding=1, output_padding=1),  # padding=1 추가
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),    # padding=1 추가
            nn.Sigmoid()
        )
    
    def encode(self, x, label):
        h = self.encoder(x)
        h = h.view(x.size(0), -1)  # 32x7x7 → 1568
        label = self.label_projector(label).view(h.size(0), -1)  # 10 → 1568
        h = torch.cat([h, label], dim=1)  # 3136
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, label):
        z_features = self.z_projector(z).view(-1, 32, 7, 7)  # z → 32x7x7
        label_features = self.label_projector(label).view(-1, 32, 7, 7)  # label → 32x7x7
        combined = torch.cat([z_features, label_features], dim=1)  # 64x7x7
        return self.decoder(combined)
    
    def forward(self, x, label):
        mu, logvar = self.encode(x, label)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, label), mu, logvar

# 5. 손실 함수 정의 (VAE와 동일)
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.BCELoss(reduction='sum')(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 6. 학습 함수
def train(model, train_loader, epochs=10, learning_rate=1e-3, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            recon_data, mu, logvar = model(data, label)
            loss = loss_function(recon_data, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'{model.__class__.__name__} Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_dataset):.4f}')

# 7. 특정 라벨로 이미지 생성
def generate_images(model, num_samples=16, label=5, save_path='generated_cva.png'):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        label_tensor = torch.tensor([label] * num_samples).to(device)
        generated = model.decode(z, label_tensor)
        save_image(generated.cpu(), save_path, nrow=4, normalize=True)
        print(f"생성된 이미지가 저장되었습니다: {save_path}")

# 8. 모델 학습 및 생성
mlp_cva = MLP_CVAE(latent_dim=latent_dim, num_classes=num_classes)
cnn_cva = CNN_CVAE(latent_dim=latent_dim, num_classes=num_classes)

print("## MLP 기반 CVAE 학습 ##")
train(mlp_cva, train_loader, epochs=epochs, learning_rate=learning_rate, device=device)

print("\n## CNN 기반 CVAE 학습 ##")
train(cnn_cva, train_loader, epochs=epochs, learning_rate=learning_rate, device=device)

# 9. 특정 라벨로 생성 (예: 숫자 5 생성)
generate_images(mlp_cva, label=5, save_path='generated_mnist_mlp_cva.png')
generate_images(cnn_cva, label=5, save_path='generated_mnist_cnn_cva.png')
