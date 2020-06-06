import torch
import torch.nn.functional as F
from tqdm import tqdm
from helper import HelperModel


class Test(object):

    def __init__(self):
        self.test_losses = []
        self.test_acc = []
        self.misclassified_images = []
        self.trueclassified_images = []

    def display_classwise_accuracy(self, test_loader, model, classes, device):
        class_correct = list(0. for i in range(len(classes)))
        class_total = list(0. for i in range(len(classes)))
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1
        for i in range(len(classes)):
            #writer.add_histogram('test accuracy per class', 100 * class_correct[i] / class_total[i], i)
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
        #writer.close()

    def update_classified_images(self, data, target, pred, misclassfied_required, trueclassified_required):
        target_change = target.view_as(pred)
        for i in range(len(pred)):
            if misclassfied_required and pred[i].item() != target_change[i].item():
                self.misclassified_images.append([data[i], pred[i], target_change[i]])
            if trueclassified_required and pred[i].item() == target_change[i].item():
                self.trueclassified_images.append([data[i], pred[i], target_change[i]])

    def test(self, model, device, test_loader, criterion, misclassfied_required=False, trueclassified_required=False, class_accuracy=None, classes=None):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                # update mis and correct classified images if requested
                if misclassfied_required or trueclassified_required:
                    self.update_classified_images(data, target, pred, misclassfied_required, trueclassified_required)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        test_acc = 100. * correct / len(test_loader.dataset)
        self.test_acc.append(test_acc)
        if (classes and class_accuracy) and test_acc >= float(class_accuracy):
            self.display_classwise_accuracy(test_loader, model, classes, device)
