from matplotlib import pyplot as plt

from train.gpt_train import train_main, plot_losses


def train_gpt():
    GPT_CONFIG_62M = {
        "file_path": './dataset/shakespeare_larger.txt',
        "vocab_size": 50257,
        "context_length": 256,
        "d_model": 768,
        "num_heads": 12,
        "num_layers": 6,
        "dropout": 0.1,
        "d_ff": 2048
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "batch_size": 2,
        "weight_decay": 0.1,
        'eval_freq': 20
    }

    # Initiate training
    train_losses, val_losses, tokens_seen, model = train_main(GPT_CONFIG_62M, OTHER_SETTINGS)

    # After training, plot results
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    # Save model
    torch.save(model.state_dict(), "model.pth")


if __name__ == '__main__':
    train_gpt()
