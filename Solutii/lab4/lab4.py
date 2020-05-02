import numpy as np
import matplotlib.pyplot as plt

class KnnClassifier:
    #1
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    #2
    def classify_image(self, test_image, num_neighbors = 3, metric = 'l2'):

        if metric == 'l1':
            distances = np.sum(abs(self.train_images-test_image), axis = 1)
        else:
            distances = np.sqrt(np.sum((self.train_images-test_image)**2, axis = 1 ))

        k_sorted = np.argsort(distances)
        k_sorted = k_sorted[:num_neighbors]

        nearest_neighbors = self.train_labels[k_sorted]
        x = np.bincount(nearest_neighbors)
        return np.argmax(x)

    #3
    def classify_images(self, test_images, num_neighbors = 3, metric = 'l2'):
        num_images = test_images.shape[0]
        predicted_labels = np.zeros(num_images)

        for i in range(num_images):
            predicted_labels[i] = self.classify_image(test_images[i], num_neighbors, metric)

        return predicted_labels

    def accuracy(self, predicted, labels):
        return np.mean(predicted == labels)


test_images = np.loadtxt('test_images.txt')
test_labels = np.loadtxt('test_labels.txt', 'int')
train_images = np.loadtxt('train_images.txt')
train_labels = np.loadtxt('train_labels.txt', 'int')

knn_classifier = KnnClassifier(train_images, train_labels)
classified = knn_classifier.classify_images(test_images, 3, 'l2')
acc = knn_classifier.accuracy(classified, test_labels)

print(acc)

#4
def getAc(knn_classifier, num_neighbors, metric):
    classified = knn_classifier.classify_images(test_images, num_neighbors, metric)
    acc = knn_classifier.accuracy(classified, test_labels)
    return acc

num_neighbors = [1, 3, 5, 7, 9]

accL1 = [0,0,0,0,0]
accL2 = [0,0,0,0,0]

for i in range(0, len(num_neighbors)):
    accL1[i] = getAc(knn_classifier, num_neighbors[i], "l1")

for i in range(0, len(num_neighbors)):
    accL2[i] = getAc(knn_classifier, num_neighbors[i], "l2")

print(accL1)
print(accL2)

plt.plot(num_neighbors, accL1)
plt.plot(num_neighbors, accL2, "r")
plt.show()