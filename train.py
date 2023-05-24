import torch
from tqdm import tqdm

def train(model, train_loader, test_loader, model_save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    EPOCHS = 10
    LEARNING_RATE = 1e-05
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
      model.train()
      train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} - Training")
      for batch in train_bar:
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['targets'].to(device)

            optimizer.zero_grad()
            outputs = model(ids, mask)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            train_bar.set_postfix(loss=loss.item())

      model.eval()
      total_loss = 0
      total_correct = 0
      total_samples = 0

      test_bar = tqdm(test_loader, desc=f"Epoch {epoch+1} - Testing")
      for batch in test_bar:
          ids = batch['ids'].to(device)
          mask = batch['mask'].to(device)
          targets = batch['targets'].to(device)

          with torch.no_grad():
              outputs = model(ids, mask)
              loss = loss_function(outputs, targets)
              total_loss += loss.item()

              _, predicted = torch.max(outputs, 1)
              total_correct += (predicted == targets).sum().item()
              total_samples += targets.size(0)

              test_bar.set_postfix(loss=loss.item())

      avg_loss = total_loss / len(test_loader)
      accuracy = total_correct / total_samples

      print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

      # Save the model after each epoch
      torch.save(model.state_dict(), f"{model_save_path}/model_epoch{epoch+1}.pth")
