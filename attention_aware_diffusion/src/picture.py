import matplotlib.pyplot as plt


def show_picture(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['total_loss'], label='Total Loss')
    plt.plot(history['mse_loss'], label='MSE Loss')
    plt.plot(history['kl_loss'], label='KL Loss')
    plt.plot(history['sparse_loss'], label='Sparse Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=300)
