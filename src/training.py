import torch 
import torch.nn as nn 
from models.autoencoder import Autoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model: Autoencoder, train_loader, val_loader, nb_epochs=20, learning_rate=1e-3):

    torch.cuda.empty_cache()

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    losses = []

    for epoch in range(nb_epochs):
        epoch_loss = 0.0

        for idx, (noisy_imgs, clean_imgs) in enumerate(train_loader):
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            optimizer.zero_grad()

            outputs = model(noisy_imgs)
            
            loss = criterion(outputs, clean_imgs)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            if (idx + 1) % 10 == 0:
                percent = (idx + 1) / len(train_loader) * 100

                bar = ('█' * int(percent // 2)) + ('-' * (50 - int(percent // 2)))

                print(f"\r|{bar}| Batch [{idx+1}/{len(train_loader)}]   |   Loss : {loss.item():.4f}\033[K", end = '', flush=True)
        
        losses.append(epoch_loss / len(train_loader))
        
        print(f"\n Epoch [{epoch}/{nb_epochs}]   |   Loss : {losses[-1]:.4f}",)




