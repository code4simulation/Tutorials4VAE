import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# 1. 하이퍼파라미터 설정
latent_dim = 20
batch_size = 128
epochs = 10
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 데이터셋 로드 및 전처리
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 3. MLP 기반 VAE 정의
class MLP_VAE(nn.Module):
    def __init__(self, latent_dim):
        super(MLP_VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = x.view(-1, 784)  # 입력 이미지를 1D 벡터로 변환
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)  # 출력을 28x28 이미지로 reshape
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# 4. CNN 기반 VAE 정의
class CNN_VAE(nn.Module):
    def __init__(self, latent_dim):
        super(CNN_VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(32 * 7 * 7, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 32 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 32, 7, 7)  # 7x7 크기로 reshape
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# 5. 손실 함수 정의
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
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon_data, mu, logvar = model(data)
            loss = loss_function(recon_data, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'{model.__class__.__name__} Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_dataset):.4f}')

# 7. 생성 함수
def generate_images(model, num_samples=16, save_path='generated.png'):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        generated = model.decode(z)
        save_image(generated.cpu(), save_path, nrow=4, normalize=True)
        print(f"생성된 이미지가 저장되었습니다: {save_path}")

# 8. 모델 학습 및 생성
#mlp_vae = MLP_VAE(latent_dim=latent_dim)
cnn_vae = CNN_VAE(latent_dim=latent_dim)

#print("## MLP 기반 VAE 학습 ##")
#train(mlp_vae, train_loader, epochs=epochs, learning_rate=learning_rate, device=device)

print("\n## CNN 기반 VAE 학습 ##")
train(cnn_vae, train_loader, epochs=epochs, learning_rate=learning_rate, device=device)

# 9. 생성 결과 비교
#generate_images(mlp_vae, save_path='generated_mnist_mlp.png')
generate_images(cnn_vae, save_path='generated_mnist_cnn.png')
