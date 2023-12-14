def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    return accuracy

def calculate_adversarial_accuracy(model, dataloader, device, attack_fn, epsilon, alpha=None, iters=None, norm='Linf'):
    model.eval()
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        data_grad = images.grad.data
        if attack_fn == fgsm_attack:
            adv_images = attack_fn(images, epsilon, data_grad)
        elif attack_fn == pgd_attack:
            adv_images = attack_fn(model, images, labels, epsilon, alpha, iters, norm)
        else:
            raise ValueError("Unknown attack function")
        outputs = model(adv_images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    accuracy = 100 * correct / total
    return accuracy


# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Evaluate clean accuracy
clean_accuracy = calculate_accuracy(model, val_dataloader, device)

# Norms and attack parameters
norms = ['L1', 'L2', 'Linf']
epsilon = 0.003
alpha = 0.01
iters = 10

# Evaluate FGSM and PGD attacks
fgsm_accuracies = {}
pgd_accuracies = {}

# FGSM is typically with Linf norm
fgsm_accuracies['Linf'] = calculate_adversarial_accuracy(
    model, val_dataloader, device, fgsm_attack, epsilon=epsilon
)

# Evaluate PGD for different norms
for norm in norms:
    pgd_accuracies[norm] = calculate_adversarial_accuracy(
        model, val_dataloader, device, pgd_attack, epsilon=epsilon, alpha=alpha, iters=iters, norm=norm
    )


# Print out the accuracies
print(f"Clean Accuracy: {clean_accuracy:.2f}%")
print(f"FGSM (Linf) Accuracy: {fgsm_accuracies['Linf']:.2f}%")
for norm in norms:
    print(f"PGD ({norm}) Accuracy: {pgd_accuracies[norm]:.2f}%")
