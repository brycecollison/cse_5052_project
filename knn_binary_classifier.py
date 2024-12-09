import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import os

# load data set
data = scipy.io.loadmat('mnist_49_3000.mat') # change to relative path to data file
x = np.array(data['x'])
y = np.array(data['y'][0])
# remap "4" samples label to 0
y[y==-1] = 0

# split data into train and test sets (first 2000 entries train, last 1000 entries test)
x_train = x[:,0:2000]
y_train = y[0:2000]
x_test = x[:,2000:]
y_test = y[2000:]

print(np.shape(x_train), np.shape(y_train))
print(np.shape(x_test), np.shape(y_test))

# define Euclidean distance function for use in classifier
def euclidean_distances(test_x, train):
  distances = [np.sqrt(np.sum((test_x - train[:, i]) ** 2)) for i in range(train.shape[1])]
  return distances

# define KNN class
class KNearestNeighbors:
  # initialize class with default k value of 3 and undefined training sets
  def __init__(self, k=3):
    self.k = k
    self.x_train = None
    self.y_train = None

  # define function to establish training data set
  def fit(self, train_x, train_y):
    self.x_train = train_x
    self.y_train = train_y
    print(np.shape(x_train))
    print(np.shape(y_train))

  # predict label of a given test data point based on K-nearest neighbors
  def _predict_one(self, test_x):
    distances = euclidean_distances(test_x, self.x_train)
    k_indices = np.argsort(distances)[:self.k]
    k_nearest_labels = [self.y_train[i] for i in k_indices]
    return np.bincount(k_nearest_labels).argmax()

  # predict labels of a given set of data points
  def predict(self, test_x):
    predictions = [self._predict_one(test_x[:, i]) for i in range(test_x.shape[1])]
    return np.array(predictions)
  
  # optimize k using validation set
  def optimize_k(self, x_val, y_val, k_values):
      best_k = None
      best_accuracy = 0

      for k in k_values:
          self.k = k
          predictions = self.predict(x_val)
          accuracy = np.mean(predictions == y_val)
          # display accuracy to 4 decimal places
          print(f"k = {k}, Validation Accuracy = {accuracy:.4f}")
          
          if accuracy > best_accuracy:
              best_k = k
              best_accuracy = accuracy

      print(f"Optimal k: {best_k}, Accuracy: {best_accuracy:.4f}")
      self.k = best_k  # Set the model to use the optimal k
      return best_k
  
# initialize KNN with k=3. K will be tuned below
knn = KNearestNeighbors(k=3)

# fit training data to model
knn.fit(x_train, y_train)

# tune k
k_range = range(1, 11)  # test k-values from 1 to 20
optimal_k = knn.optimize_k(x_test, y_test, k_range)

# classify with fitted model
predicted_labels = knn.predict(x_test)

# calculate model accuracy
test_accuracy = np.mean(predicted_labels == y_test)

print(f"Test Accuracy with k={optimal_k}: {test_accuracy:.4f}")

# compare predictions with true labels
mislabeled_mask = predicted_labels != y_test

# get indices of mislabeled points
mislabeled_indices = np.where(mislabeled_mask)[0]

# extract mislabeled data points and labels
mislabeled_data = x_test[:, mislabeled_indices]
true_labels = y_test[mislabeled_indices]
predicted_labels_mislabeled = predicted_labels[mislabeled_indices]

# display information about mislabeled points
print(f"Number of mislabeled points: {len(mislabeled_indices)}")
for i, idx in enumerate(mislabeled_indices):
    print(f"Index: {idx}, True Label: {true_labels[i]}, Predicted Label: {predicted_labels_mislabeled[i]}")

for i, idx in enumerate(mislabeled_indices[:5]):  # Visualize up to 5 mislabeled samples
    plt.figure(figsize=(3, 3))
    plt.imshow(mislabeled_data[:, i].reshape(28, 28), cmap='gray')
    plt.title(f"True: {true_labels[i]}, Predicted: {predicted_labels_mislabeled[i]}")
    plt.axis('off')
    plt.show()

# create directory for misclassified image examples
output_dir = "mislabeled_images"
os.makedirs(output_dir, exist_ok=True)

# Save up to 5 mislabeled images
for i, idx in enumerate(mislabeled_indices[:5]):
    # Reshape the sample
    image = mislabeled_data[:, i].reshape(28, 28)
    plt.figure(figsize=(3, 3))
    plt.imshow(image, cmap='gray')
    plt.title(f"True: {true_labels[i]}, Predicted: {predicted_labels_mislabeled[i]}")
    plt.axis('off')
    
    # Save the image
    image_path = os.path.join(output_dir, f"mislabeled_{idx}_true_{true_labels[i]}_pred_{predicted_labels_mislabeled[i]}.png")
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()  # Close the figure to avoid display and memory usage

    print(f"Saved image: {image_path}")