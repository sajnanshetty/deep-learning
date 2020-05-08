import matplotlib.pyplot as plt


def display_all_plot(self, train_losses, train_acc, test_losses, test_acc):
    """Plots graph for train, validation accuracy and losses"""
    try:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs[0, 0].plot(train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(test_losses)
        axs[0, 1].set_title("Validation Loss")
        axs[1, 1].plot(test_acc)
        axs[1, 1].set_title("Validation Accuracy")
    except Exception as e:
        print("First train the model")


# def plot_validation_graph(self, data, plot_case):
#     """Plots single graph for validation losses/accuracy"""
#     try:
#         fig, axs = plt.subplots(2, 2, figsize=(15, 10))
#         axs[0, 0].plot(train_losses)
#         axs[0, 0].set_title("Training Loss")
#         axs[1, 0].plot(train_acc)
#         axs[1, 0].set_title("Training Accuracy")
#         axs[0, 1].plot(test_losses)
#         axs[0, 1].set_title("Test Loss")
#         axs[1, 1].plot(test_acc)
#         axs[1, 1].set_title("Test Accuracy")
#     except Exception as e:
#         print("First train the model")


def plot_misclassified(self, model_obj, image_count=25):
  fig = plt.figure(figsize = (15,15))
  for i in range(image_count):
    sub = fig.add_subplot(5, 5, i+1)
    plt.imshow(model_obj.misclassified_images[i][0].cpu().numpy().squeeze(),cmap='gray',interpolation='none')
    sub.set_title("Predicted={0}, Actual={1}".format(str(model_obj.misclassified_images[i][1].data.cpu().numpy()),str(model_obj.misclassified_images[i][2].data.cpu().numpy())))
  plt.tight_layout()
  plt.show()