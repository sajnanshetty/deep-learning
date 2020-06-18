import matplotlib.pyplot as plt
import numpy as np
import torchvision
from google.colab import files
from gradcam import GradCAM, visualize_cam
from torchvision.utils import make_grid, save_image


class Plot(object):

    def __init__(self, train_acc, train_losses, test_acc, test_losses):
        self.train_acc = train_acc
        self.train_losses = train_losses
        self.test_acc = test_acc
        self.test_losses = test_losses

    def display_all_plot(self):
        """Plots graph for train, validation accuracy and losses"""
        try:
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            axs[0, 0].plot(self.train_losses)
            axs[0, 0].set_title("Training Loss")
            axs[1, 0].plot(self.train_acc)
            axs[1, 0].set_title("Training Accuracy")
            axs[0, 1].plot(self.test_losses)
            axs[0, 1].set_title("Validation Loss")
            axs[1, 1].plot(self.test_acc)
            axs[1, 1].set_title("Validation Accuracy")
            fig.savefig('all_graph.png')
            files.download('all_graph.png')
        except Exception as err:
            raise err

    def plot_train_graph(self, plot_case="Accuracy"):
        try:
            fig = plt.figure(figsize=(9, 9))
            if plot_case == "Accuracy":
                train_data = self.train_acc
            else:
                train_data = self.train_losses
            plt.title("Training {0}".format(plot_case))
            plt.xlabel("Epochs")
            plt.ylabel(plot_case)
            plt.plot(train_data)
            plt.show()
            fig.savefig('train_{0}_graph.png'.format(plot_case.lower()))
            files.download('train_{0}_graph.png'.format(plot_case.lower()))
        except Exception as err:
            raise err

    def plot_graph(self, plot_case="Accuracy"):
        fig, ax = plt.subplots()
        if plot_case == "Accuracy":
            train_data = self.train_acc
            test_data = self.test_acc
        else:
            train_data = self.train_losses
            test_data = self.test_losses
        plt.title("Change in training and validation {0}".format(plot_case.lower()))
        plt.xlabel("Num of Epochs")
        plt.ylabel(plot_case)
        ax.plot(train_data, 'r', label='Train')
        ax.plot(test_data, 'b', label='Validation')
        ax.legend(loc='upper left')
        plt.show()
        fig.savefig('train_validation_{0}_change.png'.format(plot_case.lower()))
        files.download('train_validation_{0}_change.png'.format(plot_case.lower()))

    def plot_validation_graph(self, plot_case="Accuracy"):
        """Plots single graph for validation losses/accuracy"""
        try:
            fig = plt.figure(figsize=(9, 9))
            if plot_case == "Accuracy":
                test_data = self.test_acc
            else:
                test_data = self.test_losses
            plt.title("Validation {0}".format(plot_case))
            plt.xlabel("Epochs")
            plt.ylabel(plot_case)
            plt.plot(test_data)
            plt.show()
            fig.savefig('validation_{0}_graph.png'.format(plot_case.lower()))
            files.download('validation_{0}_graph.png'.format(plot_case.lower()))
        except Exception as err:
            raise err


    @staticmethod
    def plot_mnist_misclassified_images(misclassified_images, image_count=25):
        fig = plt.figure(figsize=(15, 15))
        for i in range(image_count):
            sub = fig.add_subplot(5, 5, i + 1)
            plt.imshow(misclassified_images[i][0].cpu().numpy().squeeze(), cmap='gray', interpolation='none')
            sub.set_title("Predicted={0}, Actual={1}".format(str(misclassified_images[i][1].data.cpu().numpy()),
                                                             str(misclassified_images[i][2].data.cpu().numpy())))
        plt.tight_layout()
        plt.show()
        fig.savefig("mnist_misclassified_images.png")
        files.download("mnist_misclassified_images.png")

    @staticmethod
    def plot_cifar_misclassified_images(classes, misclassified_images, image_count=25):
        fig = plt.figure(figsize=(10, 10))
        for i in range(image_count):
            sub = fig.add_subplot(5, 5, i + 1)
            img = misclassified_images[i][0].cpu()
            img = img / 2 + 0.5
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray', interpolation='nearest')
            sub.set_title("P={0}, A={1}".format(str(classes[misclassified_images[i][1].item()]),
                                                str(classes[misclassified_images[i][2].item()])))
        plt.xlabel("A={0} ".format(str(classes[misclassified_images[i][1].item()])))
        plt.ylabel("P={0} ".format(str(classes[misclassified_images[i][2].item()])))
        plt.tight_layout()
        plt.show()
        fig.savefig("cifar_misclassified_images.png")
        files.download("cifar_misclassified_images.png")

    @staticmethod
    def plot_transormed_images(train_loader, classes):
        """
        Displays 5 images from each class
        :param train_loader: train loader
        :param classes: class names
        :return: None
        """
        images, labels = iter(train_loader).next()
        for i in range(len(classes)):
            index = [j for j in range(len(labels)) if labels[j] == i]
            img = torchvision.utils.make_grid(images[index[0:5]], nrow=5, padding=2, scale_each=False)
            img = img / 2 + 0.5  # unnormalize
            npimg = img.numpy()
            fig = plt.figure(figsize=(5, 5))
            plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='none')
            plt.title(classes[i])

    @staticmethod
    def plot_transormed_random_images(train_loader, classes):
        """
        Displays random 5 images
        :param train_loader: train loader
        :param classes: class names
        :return: None
        """
        images, labels = iter(train_loader).next()
        class_names = [classes[x] for x in labels[0:5]]
        img = torchvision.utils.make_grid(images[0:5], nrow=5, padding=2, scale_each=False)
        img = img / 2 + 0.5
        npimg = img.numpy()
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='none')
        plt.title(class_names)

    @staticmethod
    def plot_classified_images(classes, classified_images, image_count=25, image_name=None):
        fig = plt.figure(figsize=(10, 10))
        for i in range(image_count):
            sub = fig.add_subplot(5, 5, i + 1)
            img = classified_images[i][0].cpu()
            img = img / 2 + 0.5
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray', interpolation='nearest')
            sub.set_title("P={0}, A={1}".format(str(classes[classified_images[i][1].item()]),
                                                str(classes[classified_images[i][2].item()])))
        plt.xlabel("A={0} ".format(str(classes[classified_images[i][1].item()])))
        plt.ylabel("P={0} ".format(str(classes[classified_images[i][2].item()])))
        plt.tight_layout()
        plt.show()
        if image_name:
            fig.savefig(image_name)
            files.download(image_name)

    @staticmethod
    def image_show(img, title=None, download_image=None):
        fig = plt.figure(figsize=(7, 7))
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='none')
        if title is not None:
            plt.title(title)
        plt.pause(0.001)
        if download_image:
            fig.savefig(download_image)
            files.download(download_image)

    @staticmethod
    def plot_gradcam_images(model, layers, image_list, classes, figsize=(23,33), sub_plot_rows=9, sub_plot_cols=3, image_count=25):
      fig = plt.figure(figsize=figsize)
      for i in range(image_count):
        heat_map_image = [image_list[i][0].cpu()/2+0.5]
        result_image = [image_list[i][0].cpu()/2+0.5]
        for model_layer in layers:
          grad_cam = GradCAM(model, model_layer)
          mask, _ = grad_cam(image_list[i][0].clone().unsqueeze_(0))
          heatmap, result = visualize_cam(mask, image_list[i][0].clone().unsqueeze_(0)/2+0.5)
          heat_map_image.extend([heatmap])
          result_image.extend([result])
        grid_image = make_grid(heat_map_image+result_image, nrow=len(layers)+1,pad_value=1)
        npimg = grid_image.numpy()
        sub = fig.add_subplot(sub_plot_rows, sub_plot_cols, i+1)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        sub.set_title('P = '+classes[int(image_list[i][1])]+" A = "+classes[int(image_list[i][2])], fontweight="bold", fontsize=18)
        sub.axis("off")
        plt.tight_layout()
        fig.subplots_adjust(wspace=0)

    @staticmethod
    def plot_cycle_lr(epochs, lr_values):
        plt.plot(np.arange(epochs), lr_values)
        plt.xlabel("Num of Epochs")
        plt.ylabel("Learning rate")
        plt.title("Lr v/s Epochs")
        plt.show()
