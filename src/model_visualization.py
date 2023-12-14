# Normalization factors used in your dataset
NORM_MEAN = torch.tensor([0.485, 0.456, 0.406])
NORM_STD = torch.tensor([0.229, 0.224, 0.225])


class_names = dataset.classes  # This will give you the list of class names

def show_prediction(img, label, pred, K=5):
    # Convert normalization factors to numpy arrays
    norm_mean = NORM_MEAN.numpy()
    norm_std = NORM_STD.numpy()

    if isinstance(img, torch.Tensor):
        # Convert image to numpy array and normalize
        img = img.cpu().permute(1, 2, 0).numpy()
        img = (img * norm_std) + norm_mean
        img = np.clip(img, a_min=0.0, a_max=1.0)
        label = label.item()

    # Create a figure with a 1x2 grid layout
    fig = plt.figure(figsize=(10, 3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2])

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    # Display the image
    #ax0.imshow(img)
    ax0.imshow(img, aspect='auto')
    ax0.set_title(f'{class_names[label]}')
    ax0.axis('off')

    # Compute the softmax probabilities and find the top K predictions
    pred = torch.softmax(pred, dim=-1)
    topk_vals, topk_idx = pred.topk(K, dim=-1)
    topk_vals, topk_idx = topk_vals.cpu().numpy(), topk_idx.cpu().numpy()

    # Assign colors to the bars based on the correct label
    colors = ['#55A868' if i == label else '#4C72B0' for i in topk_idx]

    # Plot the top K predictions with their confidence percentages
    ax1.barh(np.arange(K), topk_vals * 100, color=colors, edgecolor='white', linewidth=1.3)
    ax1.set_yticks(np.arange(K))
    ax1.set_yticklabels([class_names[i] for i in topk_idx])
    ax1.invert_yaxis()  # Highest probabilities on top
    ax1.set_xlabel('Confidence %')
    ax1.set_xlim(0, 100)
    ax1.set_title('Predictions')
    # Set the background to a grid
    ax1.set_facecolor('#ECECEC')  # Light grey background
    # Add gridlines and put them behind the bars
    ax1.xaxis.grid(True, color='white', linestyle='-', linewidth=1.3, alpha=0.7)
    ax1.yaxis.grid(True, color='white', linestyle='-', linewidth=1.3, alpha=0.7)
    ax1.set_axisbelow(True)  # This line ensures the grid is behind the plot

    plt.tight_layout()
    plt.show()


# Check if a GPU is available and set PyTorch to use the GPU, otherwise use the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Now, move your model to the selected device
model.to(device)

# The rest of your code can stay the same
model.eval()
exmp_batch, label_batch = next(iter(val_dataloader))
with torch.no_grad():
    preds = model(exmp_batch.to(device))

# Visualize predictions for the first few images in the batch
for i in range(min(len(exmp_batch), 2)):
    show_prediction(exmp_batch[i], label_batch[i], preds[i])
