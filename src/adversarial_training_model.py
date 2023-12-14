class DogBreedClassifier(pl.LightningModule):
    def __init__(self, num_classes=120, adv_training_start_epoch=5):
        super().__init__()
        self.automatic_optimization = False
        self.model = models.densenet161(pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

        # Freeze the pretrained layers
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Define loss function and metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(num_classes=num_classes, task = "multiclass")
        self.val_acc = Accuracy(num_classes=num_classes, task = "multiclass")
        self.f1 = F1Score(num_classes=num_classes, task = "multiclass")
        self.precision = Precision(num_classes=num_classes, task = "multiclass")
        self.confmat = ConfusionMatrix(num_classes=num_classes, task = "multiclass")

        # Adversarial training parameters
        self.adv_training_start_epoch = adv_training_start_epoch
        self.epsilon = 0.01  # Epsilon for FGSM
        self.alpha = 0.01  # Alpha for PGD
        self.iters = 10  # Iterations for PGD
        self.norm_type = 'Linf'  # Can be 'L1', 'L2', or 'Linf'

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.detach().requires_grad_(True)
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.train_acc(logits, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)

        if self.current_epoch >= self.adv_training_start_epoch:
            # Calculate gradients for the current step
            self.manual_backward(loss, retain_graph=True)

            # Adversarial training
            x_adv = fgsm_attack(x, self.epsilon, x.grad)  # FGSM attack
            logits_adv = self.forward(x_adv)
            loss_adv = self.loss_fn(logits_adv, y)
            self.log('train_adv_loss', loss_adv, on_step=True, on_epoch=True)

            # Combine losses and do a backward pass
            combined_loss = loss + loss_adv
            self.manual_backward(combined_loss)

            # Perform an optimizer step and zero the gradients
            optimizer = self.optimizers()
            optimizer.step()
            optimizer.zero_grad()

            return combined_loss

        else:
            self.manual_backward(loss)

            # Perform an optimizer step and zero the gradients
            optimizer = self.optimizers()
            optimizer.step()
            optimizer.zero_grad()

            return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.val_acc(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=1e-3)
        return optimizer

# Instantiate the model and trainer with CSVLogger
model = DogBreedClassifier(num_classes=120, adv_training_start_epoch=2)
csv_logger = CSVLogger(save_dir='logs/', name='dog_breed_classifier_logs')
trainer = Trainer(
    max_epochs=5,
    logger=csv_logger,
    callbacks=[ModelCheckpoint(monitor='val_loss'), EarlyStopping(monitor='val_loss')]
)


try:
    trainer.fit(model, train_dataloader, val_dataloader)
except Exception as e:
    print(f"An error occurred during training: {e}")

torch.save(model.state_dict(), 'dog_breed_classifier_state_dict_adversarial.pth')
torch.save(model, 'dog_breed_classifier_full_model_adversarial.pth')
