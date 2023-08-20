import os
import json
import numpy as np
import nltk
from tqdm.auto import tqdm

import tensorflow as tf
import keras.preprocessing.text as kpt
from keras.utils import pad_sequences
import keras
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve, auc, accuracy_score, precision_score, \
    recall_score

# # 下载所需的nltk数据包
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')

# 定义停用词集合、词形还原器、最大文本长度和最大词汇量
stopWords = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()
maxLength = 1000
maxVocab = 20000

trainPath = "ECHR_Dataset/EN_train"
validationPath = "ECHR_Dataset/EN_dev"
testPath = "ECHR_Dataset/EN_test"


def addToLabels(texts, labels, tokenizedText, jsonData):
    # 计算违反规定的总数
    totalviolated = getViolated(jsonData)
    # 将处理后的文本添加到texts列表中
    texts.append(tokenizedText)
    # 根据违反规定的总数，将对应的标签（1或0）添加到labels列表中
    if totalviolated > 0:
        labels.append(1)
    else:
        labels.append(0)


def getViolated(jsonData):
    # 从jsonData中提取违反规定的文章、段落和重点
    article = jsonData['VIOLATED_ARTICLES']
    paragraph = jsonData['VIOLATED_PARAGRAPHS']
    bulletpoint = jsonData['VIOLATED_BULLETPOINTS']
    # 计算违反规定的总数
    totalviolated = len(article) + len(paragraph) + len(bulletpoint)
    return totalviolated


def preProcessing(value):
    # 将输入的文本转换为字符串
    stringList = "".join(value)
    # 对字符串进行分词
    list = nltk.word_tokenize(stringList)
    # 将单词转换为小写并过滤掉非字母字符
    list = [word.lower() for word in list if word.isalpha()]
    # 将分词后的结果进行词形还原
    list = ([lemmatizer.lemmatize(x) for x in list])
    # 将处理后的词汇重新组合成字符串
    newList = ""
    newL = []
    for x in list:
        newList += x
        newList += " "
    # 如果新字符串为空，则返回原始值；否则返回处理后的字符串
    if (len(newList) == 0):
        newL.append(value)
    else:
        newL.append(newList)
    return newList


def load_data(trainPath, validationPath, testPath, preProcessing, addToLabels, flatten_text):
    # 获取训练、验证和测试文件夹中的所有JSON文件
    jsonFilesTrain = [x for x in os.listdir(trainPath) if x.endswith("json")]
    jsonFilesValidation = [x for x in os.listdir(validationPath) if x.endswith("json")]
    jsonFilesTest = [x for x in os.listdir(testPath) if x.endswith("json")]

    # 初始化文本和标签列表
    traintexts = []
    trainlabels = []
    validationtexts = []
    validationlabels = []
    testtexts = []
    testlabels = []

    def process_data(jsonFiles, path, texts, labels, flatten_text):
        for json_file in jsonFiles:
            datasetPath = os.path.join(path, json_file)
            with open(datasetPath, "r") as f:
                jsonData = json.load(f)
                value = jsonData['TEXT']

                if flatten_text:
                    # 将JSON文件中的文本合并为一个字符串
                    text = ""
                    for i in range(0, len(value)):
                        text += value[i]
                else:
                    text = preProcessing(value)

                # 添加处理后的文本和对应标签到相应列表中
                addToLabels(texts, labels, text, jsonData)

    def print_texts_and_labels(dataset_type, texts, labels):
        for i in range(1):
            print(f"{dataset_type} Text {i}: {texts[i]}")
            print(f"{dataset_type} Label {i}: {labels[i]}\n")

    process_data(jsonFilesTrain, trainPath, traintexts, trainlabels, flatten_text)
    process_data(jsonFilesValidation, validationPath, validationtexts, validationlabels, flatten_text)
    process_data(jsonFilesTest, testPath, testtexts, testlabels, flatten_text)

    print_texts_and_labels("Train", traintexts, trainlabels)
    print_texts_and_labels("Validation", validationtexts, validationlabels)
    print_texts_and_labels("Test", testtexts, testlabels)

    return traintexts, trainlabels, validationtexts, validationlabels, testtexts, testlabels


class LSTMTextClassifier:
    def __init__(self, max_vocab=5000):
        self.max_vocab = max_vocab
        self.tokenizer = kpt.Tokenizer(max_vocab)
        self.model = None

    def preprocess_data(self, traintexts, trainlabels, validationtexts, validationlabels, testtexts, testlabels):
        self.tokenizer.fit_on_texts(traintexts)

        sequences = self.tokenizer.texts_to_sequences(traintexts)
        validation_sequences = self.tokenizer.texts_to_sequences(validationtexts)
        test_sequences = self.tokenizer.texts_to_sequences(testtexts)

        train_data = pad_sequences(sequences, 512, padding="pre", truncating="pre")
        validation_data = pad_sequences(validation_sequences, 512, padding="pre", truncating="pre")
        test_data = pad_sequences(test_sequences, 512, padding="pre", truncating="pre")

        self.messages_train = np.asarray(train_data)
        self.labels_train = np.asarray(trainlabels)
        self.messages_validation = np.asarray(validation_data)
        self.labels_validation = np.asarray(validationlabels)
        self.messages_test = np.asarray(test_data)
        self.labels_test = np.asarray(testlabels)

    def build_model(self):
        embedding_mat_columns = 32

        self.model = keras.Sequential()
        self.model.add(
            keras.layers.Embedding(input_dim=self.max_vocab, output_dim=embedding_mat_columns, input_length=512))
        self.model.add(keras.layers.LSTM(units=128, dropout=0, recurrent_dropout=0, activation='tanh'))
        self.model.add(keras.layers.Dropout(0.3))
        self.model.add(keras.layers.Dense(1, activation='sigmoid'))

        plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        initial_learning_rate = 0.0001

        lr_schedule = tf.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True)

        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), loss='binary_crossentropy',
                           metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.Precision(), keras.metrics.Recall(),
                                    keras.metrics.FalsePositives(), keras.metrics.TruePositives(),
                                    keras.metrics.FalseNegatives(),
                                    keras.metrics.TrueNegatives()])

        self.model.summary()

    def train_model(self, param_grid):
        history = self.model.fit(self.messages_train, self.labels_train, epochs=8, batch_size=128,
                                 validation_data=(self.messages_validation, self.labels_validation))

        self.model.save('lastModel.h5')

        print(history.history.keys())

        plt.plot(history.history['binary_accuracy'])
        plt.plot(history.history['val_binary_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('val_accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('accuracy.png', dpi=300)
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('loss.png', dpi=300)
        plt.show()

    def evaluate_model(self):
        loss, accuracy, precision, recall, false_positives, true_positives, false_negatives, true_negatives = self.model.evaluate(
            self.messages_test, self.labels_test, batch_size=128)

        confusion = np.array([[true_negatives, false_positives],
                              [false_negatives, true_positives]])

        return loss, accuracy, precision, recall, confusion


class MLTextClassifier:

    def __init__(self):
        self.model = None

    def preprocess_data(self, traintexts, trainlabels, validationtexts, validationlabels, testtexts, testlabels):

        self.messages_train = np.asarray(traintexts)
        self.labels_train = np.asarray(trainlabels)
        self.messages_test = np.asarray(testtexts)
        self.labels_test = np.asarray(testlabels)
        self.messages_validation = np.asarray(validationtexts)
        self.labels_validation = np.asarray(validationlabels)

        print("Sample of unvectorized train data:")
        print(self.messages_train[:1])

        print("Sample of unvectorized test data:")
        print(self.messages_test[:1])

        vectorizer = CountVectorizer()
        self.messages_train = vectorizer.fit_transform(self.messages_train)
        self.messages_test = vectorizer.transform(self.messages_test)
        self.messages_validation = vectorizer.transform(self.messages_validation)

    def build_model(self):
        pass

    def train_model(self, param_grid=None):
        if param_grid is not None:
            print("Training model with grid search...")
            grid_search = GridSearchCV(self.model, param_grid=param_grid, cv=5)
            for i in tqdm(range(1)):  # 使用tqdm显示进度条
                grid_search.fit(self.messages_train, self.labels_train)
            self.model = grid_search.best_estimator_
        else:
            print("Training model...")
            for i in tqdm(range(1)):  # 使用tqdm显示进度条
                self.model.fit(self.messages_train, self.labels_train)

    def evaluate_model(self):

        print("Evaluating model...")
        predictions = []
        for message in tqdm(self.messages_test):  # 使用tqdm显示进度条
            prediction = self.model.predict(message.reshape(1, -1))
            predictions.append(prediction[0])

        predictions = self.model.predict(self.messages_test)
        accuracy = accuracy_score(self.labels_test, predictions)
        precision = precision_score(self.labels_test, predictions)
        recall = recall_score(self.labels_test, predictions)
        confusion = confusion_matrix(self.labels_test, predictions)
        loss = None

        # 绘制混淆矩阵
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_confusion_matrix(self.model, self.messages_test, self.labels_test, ax=ax)
        ax.set_title('Confusion matrix')
        plt.show()

        # 绘制ROC曲线
        y_proba = self.model.predict_proba(self.messages_test)
        fpr, tpr, _ = roc_curve(self.labels_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver operating characteristic')
        ax.legend(loc="lower right")
        plt.show()

        return loss, accuracy, precision, recall, confusion


class SVMTextClassifier(MLTextClassifier):
    def build_model(self):
        self.model = SVC(kernel='linear', probability=True)


class RandomForestTextClassifier(MLTextClassifier):
    def build_model(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)


def choice(classifier_type):
    if classifier_type == 'lstm':
        traintexts, trainlabels, validationtexts, validationlabels, testtexts, testlabels = load_data(
            trainPath, validationPath, testPath, preProcessing, addToLabels, flatten_text=False)

        classifier = LSTMTextClassifier()
        param_grid = None
    else:
        traintexts, trainlabels, validationtexts, validationlabels, testtexts, testlabels = load_data(
            trainPath, validationPath, testPath, preProcessing, addToLabels, flatten_text=True)

        if classifier_type == 'svm':
            classifier = SVMTextClassifier()
            param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        elif classifier_type == 'random_forest':
            classifier = RandomForestTextClassifier()
            param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]}
        else:
            raise ValueError("Invalid classifier type")

    classifier.preprocess_data(traintexts, trainlabels, validationtexts, validationlabels, testtexts, testlabels)
    classifier.build_model()
    classifier.train_model(param_grid=param_grid)

    loss, accuracy, precision, recall,  confusion = classifier.evaluate_model()
    print(f"Classifier: {classifier_type}")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Loss: ", loss)
    print("Confusion Matrix: \n", confusion)


if __name__ == "__main__":
    # classifier_types = ['lstm', 'svm', 'random_forest']
    # for classifier_type in classifier_types:
    choice('random_forest')
