import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# 1. Definisi Arsitektur Decoder (Sesuai dengan kodingan Kaggle)
class Decoder(nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 2048),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)

# 2. Definisi Model VAE Lengkap untuk me-load state_dict
class EncoderVAE(nn.Module):
    def __init__(self, z_dim):
        super(EncoderVAE, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, z_dim)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar

class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.encoder = EncoderVAE(z_dim)
        self.decoder = Decoder(z_dim)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        out = self.decoder(z)
        return out, mu, logvar

# 3. Pengaturan Gambar Input
image_path = "baju_test.jpg"  

if not os.path.exists(image_path):
    print(f"Error: File gambar '{image_path}' tidak ditemukan di folder ini!")
    exit()

# Transformasi gambar ke Grayscale, ukuran 32x32, dan Tensor (Sesuai setelan training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Load gambar asli
img = Image.open(image_path)
img_tensor = transform(img).unsqueeze(0) # Tambah dimensi batch (1, 1, 32, 32)

# Persiapan Plot Gambar (1 Asli + 3 Rekonstruksi)
plt.figure(figsize=(12, 4))

# Tampilkan Gambar Asli
plt.subplot(1, 4, 1)
plt.imshow(img_tensor.squeeze(), cmap='gray')
plt.title("Gambar Asli")
plt.axis("off")

# 4. Lakukan Rekonstruksi untuk Setiap Dimensi (2, 8, 32)
dimensions = [2, 8, 32]

for idx, z_dim in enumerate(dimensions):
    model_name = f"autoencoder_{z_dim}.pth"
    
    if os.path.exists(model_name):
        # Inisialisasi model sesuai dimensinya
        model = VAE(z_dim=z_dim)
        
        # Load bobot pelatihan
        model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu'), weights_only=True))
        model.eval()
        
        with torch.no_grad():
            # Jalankan forward pass untuk rekonstruksi gambar
            recon_img, _, _ = model(img_tensor)
            
        # Plot hasil rekonstruksi
        plt.subplot(1, 4, idx + 2)
        plt.imshow(recon_img.squeeze(), cmap='gray')
        plt.title(f"Dimensi {z_dim}")
        plt.axis("off")
    else:
        print(f"Peringatan: File model '{model_name}' tidak ditemukan, dilewati.")

plt.tight_layout()
print("Menampilkan hasil perbandingan rekonstruksi...")
plt.show()