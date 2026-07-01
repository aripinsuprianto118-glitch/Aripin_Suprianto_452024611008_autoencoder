import argparse
import torch
import torch.nn as nn
from torchvision.utils import save_image

# ==============================================================================
# 1. ARSITEKTUR DECODER 
# ==============================================================================
class Decoder(nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        
        # ---> PENTING: PASTIKAN BAGIAN INI SAMA PERSIS DENGAN KAGGLE KAMU <---
        # Berdasarkan error, kamu punya layer di index 0, 2, 4, 6, dan 8.
        # Susunannya kurang lebih seperti di bawah ini, TAPI COCOKKAN LAGI YA!
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 2048),                                                              # Index 0
            nn.Unflatten(1, (128, 4, 4)),                                                        # Index 1
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Index 2
            nn.ReLU(),                                                                           # Index 3
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # Index 4
            nn.ReLU(),                                                                           # Index 5
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),    # Index 6 (Tambahan 1)
            nn.ReLU(),                                                                           # Index 7
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),           # Index 8                      # Index 8 (Tambahan 2)
            nn.Sigmoid() 
        )

    def forward(self, z):
        return self.decoder(z)
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", type=str, required=True, help="Path ke model decoder .pth")
    parser.add_argument("--latent", type=str, required=True, help="Latent vector dipisah koma")
    args = parser.parse_args()

    latent_values = [float(x) for x in args.latent.split(',')]
    z_dim = len(latent_values)
    latent_tensor = torch.tensor(latent_values, dtype=torch.float32).unsqueeze(0)
    
    print(f"[*] Menyiapkan Decoder dengan Latent Dimension: {z_dim}")

    model = Decoder(z_dim=z_dim)
    
    try:
        # 1. Load FULL state dict (karena file .pth kamu isinya full model)
        full_state_dict = torch.load(args.decoder, map_location=torch.device('cpu'), weights_only=True)
        
        # 2. Filter! Ambil HANYA yang milik decoder dan buang encoder-nya
        decoder_only_dict = {}
        for key, value in full_state_dict.items():
            if key.startswith('decoder.decoder.'):
                # Hapus prefix 'decoder.decoder.' jadi 'decoder.' agar cocok dengan class
                new_key = key.replace('decoder.decoder.', 'decoder.')
                decoder_only_dict[new_key] = value
            elif key.startswith('decoder.'):
                decoder_only_dict[key] = value
                
        # 3. Load dictionary yang sudah difilter ke dalam model
        model.load_state_dict(decoder_only_dict)
        model.eval()
        print("[*] Berhasil menyaring dan memuat bobot Decoder!")
        
    except Exception as e:
        print(f"[!] Error saat memuat model: {e}")
        exit()

    print("[*] Melakukan inferensi...")
    with torch.no_grad():
        generated = model(latent_tensor)
        if generated.dim() == 2: 
            generated = generated.view(1, 1, 28, 28)
    
    save_image(generated, "generated_image.png")
    print("[*] Selesai! Gambar berhasil disimpan sebagai 'generated_image.png'")