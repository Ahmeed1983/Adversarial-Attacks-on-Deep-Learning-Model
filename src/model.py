class DogBreedClassifier(pl.LightningModule):
    def __init__(self, num_classes=120):
        super().__init__()
        # Initialize the DenseNet model with the final layer adapted for 120 classes
        self.model = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

        # Freeze the pretrained layers
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Define loss function and metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(num_classes=num_classes, task = "multiclass")
        self.val_acc = Accuracy(num_classes=num_classes, task = "multiclass")
        self.f1 = F1Score(num_classes=num_classes, task = "multiclass", average='macro')
        self.precision = Precision(num_classes=num_classes, task = "multiclass", average='macro')
        self.confmat = ConfusionMatrix(num_classes=num_classes, task = "multiclass")
        self.confmat_matrices = []  # To store confusion matrices after each epoch

        # Variables to track validation metrics for each epoch
        self.validation_outputs = []  # To store outputs for the on_validation_epoch_end hook
        self.validation_loss_list = []
        self.validation_acc_list = []
        self.validation_f1_list = []
        self.validation_precision_list = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.train_acc(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Calculate metrics manually
        correct = preds.eq(y).sum().item()
        total = y.size(0)
        manual_acc = correct / total

        # Update metrics
        self.val_acc.update(preds, y)
        self.f1.update(preds, y)
        self.precision.update(preds, y)
         # Update confusion matrix
        self.confmat.update(preds, y)


        # Store the outputs for use in the on_validation_epoch_end hook
        self.validation_outputs.append({'val_loss': loss.detach(), 'preds': preds, 'targets': y})
        return {'val_loss': loss.detach()}

    def on_validation_epoch_end(self):
        # Calculate the average of the validation loss using stored outputs
        if self.validation_outputs:
            avg_val_loss = torch.stack([x['val_loss'] for x in self.validation_outputs]).mean()
            avg_val_acc = self.val_acc.compute()
            avg_val_f1 = self.f1.compute()
            avg_val_precision = self.precision.compute()

            # Append the average metrics to the tracking lists
            self.validation_loss_list.append(avg_val_loss.cpu().item())
            self.validation_acc_list.append(avg_val_acc.cpu().item())
            self.validation_f1_list.append(avg_val_f1.cpu().item())
            self.validation_precision_list.append(avg_val_precision.cpu().item())

            # Print the validation metrics
            print(f'Epoch {self.current_epoch + 1}: '
                  f'Validation Loss: {avg_val_loss}, '
                  f'Accuracy: {avg_val_acc}, '
                  f'F1 Score: {avg_val_f1}, '
                  f'Precision: {avg_val_precision}')

            # Log the validation metrics
            self.log('avg_val_loss', avg_val_loss)
            self.log('avg_val_acc', avg_val_acc)
            self.log('avg_val_f1', avg_val_f1)
            self.log('avg_val_precision', avg_val_precision)

            # Compute and store the confusion matrix for the current epoch
            confmat = self.confmat.compute()
            self.confmat_matrices.append(confmat.cpu().numpy())
            self.confmat.reset()

            # Reset metrics after each epoch
            self.val_acc.reset()
            self.f1.reset()
            self.precision.reset()

            # Clear the list of saved outputs for the next epoch
            self.validation_outputs.clear()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=1e-3)
        return optimizer

# Instantiate the model and trainer with CSVLogger
model = DogBreedClassifier(num_classes=120)
csv_logger = CSVLogger(save_dir='logs/', name='dog_breed_classifier_logs')
trainer = Trainer(
    max_epochs=5,
    logger=csv_logger,
    callbacks=[ModelCheckpoint(monitor='avg_val_loss'), EarlyStopping(monitor='avg_val_loss')]
)

# Assuming 'train_dataloader' and 'val_dataloader' are defined
try:
    trainer.fit(model, train_dataloader, val_dataloader)
except Exception as e:
    print(f"An error occurred during training: {e}")

torch.save(model.state_dict(), 'dog_breed_classifier.pth')

torch.save(model, 'dog_breed_classifier_full.pth')

#print("Validation Loss List:", model.validation_loss_list)
print("Validation Accuracy List:", model.validation_acc_list)

epochs = list(range(1, len(model.validation_loss_list) + 1))
plt.figure(figsize=(6, 5))
plt.plot(epochs, model.validation_acc_list, label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.xticks(epochs)
plt.legend()
plt.grid()
plt.show()

# Plot for loss
plt.figure(figsize=(6, 5))
plt.plot(epochs, model.validation_loss_list, 'b-', label='Validation Loss')
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
