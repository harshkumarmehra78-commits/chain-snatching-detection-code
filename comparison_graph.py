import matplotlib.pyplot as plt

models = ["CNN+LSTM", "CNN", "RF", "SVM"]
accuracy = [0.98, 0.88, 0.80, 0.78]

plt.bar(models, accuracy)
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.show()