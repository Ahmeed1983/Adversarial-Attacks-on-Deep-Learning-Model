# Normalization factors used in your dataset
NORM_MEAN = torch.tensor([0.485, 0.456, 0.406])
NORM_STD = torch.tensor([0.229, 0.224, 0.225])

class_names = dataset.classes

# Functions to create adversarial examples
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def pgd_attack(model, images, labels, eps, alpha, iters, norm='Linf'):
    # Copy the original images to perturb
    perturbed_images = images.clone().detach().requires_grad_(True)

    for _ in range(iters):
        outputs = model(perturbed_images)
        model.zero_grad()
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        grad = perturbed_images.grad.data

        # Perform the attack step
        if norm == "Linf":
            step = alpha * grad.sign()
        elif norm == "L2":
            g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            step = alpha * grad / (g_norm + 1e-10)
        elif norm == "L1":
            step = alpha * grad / (grad.abs().sum(dim=[1,2,3], keepdim=True) + 1e-10)
        else:
            raise ValueError("Invalid norm choice.")

        # Apply the perturbation step while staying within the epsilon-ball and [0,1] range
        perturbed_images = perturbed_images + step
        perturbed_images = torch.min(torch.max(perturbed_images, images - eps), images + eps)
        perturbed_images = torch.clamp(perturbed_images, 0, 1).detach().requires_grad_(True)

    return perturbed_images


def tensor_to_img(tensor):
    # Move normalization tensors to the same device as the input tensor
    norm_mean = NORM_MEAN.to(tensor.device)
    norm_std = NORM_STD.to(tensor.device)

    tensor = tensor * norm_std[:, None, None] + norm_mean[:, None, None]
    tensor = torch.clamp(tensor, 0, 1)
    # Make sure to detach before calling numpy()
    img = tensor.detach().permute(1, 2, 0).cpu().numpy()
    return img

# Function to visualize adversarial examples
def visualize_adversarial_examples(model, dataloader, device, attack='fgsm', epsilon=0.003, alpha=0.01, iters=10, norm='L1'):
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    if attack == 'fgsm':
        images.requires_grad = True
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        data_grad = images.grad.data
        perturbed_images = fgsm_attack(images, epsilon, data_grad)
    elif attack == 'pgd':
        perturbed_images = pgd_attack(model, images, labels, epsilon, alpha, iters, norm)

    # Plot the results
    for i in range(min(len(images), 3)):
        original_img_np = tensor_to_img(images[i])
        perturbed_img_np = tensor_to_img(perturbed_images[i])
        noise_img_np = tensor_to_img(perturbed_images[i] - images[i])
        outputs = model(perturbed_images[i].unsqueeze(0))
        pred_probs, pred_classes = torch.topk(torch.softmax(outputs, dim=1).squeeze(), 5)

        fig, axes = plt.subplots(1, 4, figsize=(23, 3), gridspec_kw={'width_ratios': [2, 2, 2, 3.5], 'wspace': 1})

        axes[0].imshow(original_img_np)
        axes[0].set_title(f'{class_names[labels[i]]}')
        axes[0].axis('off')

        axes[1].imshow(perturbed_img_np)
        axes[1].set_title(f'Adversarial ({attack.upper()})')
        axes[1].axis('off')

        axes[2].imshow(noise_img_np)
        axes[2].set_title(f'Noise ({norm})')
        axes[2].axis('off')

        axes[3].barh(np.arange(5), pred_probs.detach().cpu().numpy() * 100,
                    color=['#55A868' if class_names[pred_classes[j]] == class_names[labels[i]] else '#4C72B0' for j in range(5)])
        axes[3].set_yticks(np.arange(5))
        axes[3].set_yticklabels([class_names[pred_classes[j]] for j in range(5)])
        axes[3].invert_yaxis()
        axes[3].set_xlabel('Confidence %')
        axes[3].set_title('Predictions')
        axes[3].set_xlim(0, 100)
        axes[3].set_facecolor('#ECECEC')
        axes[3].xaxis.grid(True, color='white', linestyle='-', linewidth=1.3, alpha=0.7)
        axes[3].yaxis.grid(True, color='white', linestyle='-', linewidth=1.3, alpha=0.7)
        axes[3].set_axisbelow(True)
        plt.tight_layout()
        plt.show()

# Check if a GPU is available and set PyTorch to use the GPU, otherwise use the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

visualize_adversarial_examples(model, val_dataloader, device, attack='fgsm', epsilon=0.003)
