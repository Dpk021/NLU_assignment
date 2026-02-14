import pandas as pd
import numpy as np
import math
from collections import Counter
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

logging.basicConfig(
    level=logging.INFO,  ## Defines Level of Criticality eg info just gives message we want to print
    format="%(asctime)s - %(levelname)s - %(message)s",
    force =True
)

try:
    logging.info("Loading Dataset..........")
    df = pd.read_csv('sports_politics_dataset.csv')
    logging.info("Dataset Loaded Sucessfully....")
    print(df.head(2).to_markdown())
except Exception:
    logging.info("Dataset Loading unsuccessful")

def basic_data_quality(df):
    logging.info("Checking for null values")
    null_count = df.isna().sum().sum()
    if null_count > 0:
        logging.info(f"Found {null_count} null values. Dropping rows.")
        df = df.dropna()
        logging.info(f"Shape after null drop: {df.shape}")
    else:
        logging.info("No null values found.")
    df["text"]=df["text"].str.lower()
    return df
df = basic_data_quality(df)

class Text_cleaner:
    def __init__(self,df):
        self.df = df
    def cleaner_1(self):
        self.df["tokens"]=self.df["text"].str.split()
        return self.df
instance=Text_cleaner(df)
df = instance.cleaner_1()
print(df.head().to_markdown())

def different_classes(df):
    return df["category"].value_counts().reset_index(name="Counts")
different_classes(df)

# Lets Build Bag of words for Our Vocabalry

def bag_of_words(df):
    n=len(df)
    arr=[]
    store={}
    for items in range(n):
        arr+=df["tokens"][items]
    for items in arr:
        store[items]=store.get(items,0)+1
    return sorted(store.items(),key=lambda x:x[1],reverse=True)
bag_of_words(df)

# Lets Analyze the Top occouring words in Sports
def sports_counter(df):
    n=len(df[df["category"]=="sport"])
    arr=[]
    store={}
    for items in df[df["category"]=="sport"]["tokens"]:
        arr+=items
    for items in arr:
        store[items]=store.get(items,0)+1
    return sorted(store.items(),key=lambda x:x[1],reverse=True)
sports_counter(df)

# Lets Analyze the Top occouring words in Politics

def politics_counter(df):
    n=len(df[df["category"]=="politics"])
    arr=[]
    store={}
    for items in df[df["category"]=="politics"]["tokens"]:
        arr+=items
    for items in arr:
        store[items]=store.get(items,0)+1
    return sorted(store.items(),key=lambda x:x[1],reverse=True)
politics_counter(df)

# Just Making the data Organised ..................

df=df[["text","tokens","category"]]

#  TF-IDF FROM SCRATCH
class TFIDF_Vectorizer:
    def __init__(self):
        self.vocabulary_ = {}
        self.idf_ = {}

    def fit(self, documents):
        # Build vocabulary
        vocab = set()
        for doc in documents:
            words = doc.lower().split()
            vocab.update(words)
        self.vocabulary_ = {word: idx for idx, word in enumerate(sorted(vocab))}

        # Compute IDF
        N = len(documents)
        doc_freq = Counter()
        for doc in documents:
            words = set(doc.lower().split())
            for word in words:
                doc_freq[word] += 1

        for word, idx in self.vocabulary_.items():
            df = doc_freq.get(word, 0)
            self.idf_[word] = math.log((N + 1) / (df + 1)) + 1  # smooth IDF
        return self

    def transform(self, documents):
        rows = []
        for doc in documents:
            words = doc.lower().split()
            tf = Counter(words)
            total_words = len(words)
            vector = np.zeros(len(self.vocabulary_))
            for word, count in tf.items():
                if word in self.vocabulary_:
                    tf_val = count / total_words
                    idf_val = self.idf_.get(word, 0)
                    vector[self.vocabulary_[word]] = tf_val * idf_val
            # L2 normalization
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            rows.append(vector)
        return np.array(rows)

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)

#  LOGISTIC REGRESSION FROM SCRATCH

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.classes_ = None

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        if len(self.classes_) == 2:
            # Binary classification
            self._fit_binary(X, y)
        else:
            # One vs Rest for multiclass
            self.classifiers_ = {}
            for cls in self.classes_:
                binary_y = (y == cls).astype(int)
                w, b = self._train_binary(X, binary_y)
                self.classifiers_[cls] = (w, b)

    def _train_binary(self, X, y):
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        bias = 0

        for _ in range(self.n_iters):
            linear_model = X.dot(weights) + bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * X.T.dot(y_predicted - y)
            db = (1 / n_samples) * np.sum(y_predicted - y)

            weights -= self.lr * dw
            bias -= self.lr * db

        return weights, bias

    def _fit_binary(self, X, y):
        self.weights, self.bias = self._train_binary(X, (y == self.classes_[1]).astype(int))

    def predict(self, X):
        if len(self.classes_) == 2:
            linear_model = X.dot(self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            class_preds = (y_predicted >= 0.5).astype(int)
            return np.array([self.classes_[p] for p in class_preds])
        else:
            # OvR: pick class with highest probability
            scores = np.zeros((X.shape[0], len(self.classes_)))
            for i, cls in enumerate(self.classes_):
                w, b = self.classifiers_[cls]
                scores[:, i] = self._sigmoid(X.dot(w) + b)
            return self.classes_[np.argmax(scores, axis=1)]

#  MULTINOMIAL NAIVE BAYES FROM SCRATCH

class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            self.class_log_prior_[i] = math.log(X_cls.shape[0] / X.shape[0])

            # Sum of feature counts for this class + smoothing
            feature_count = X_cls.sum(axis=0) + self.alpha
            total_count = feature_count.sum()
            self.feature_log_prob_[i] = np.log(feature_count / total_count)

    def predict(self, X):
        # log P(class) + sum of log P(feature|class)
        log_probs = X.dot(self.feature_log_prob_.T) + self.class_log_prior_
        return self.classes_[np.argmax(log_probs, axis=1)]

#  DECISION TREE FROM SCRATCH

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _gini(self, y):
        classes = np.unique(y)
        n = len(y)
        if n == 0:
            return 0
        gini = 1.0
        for cls in classes:
            p = np.sum(y == cls) / n
            gini -= p ** 2
        return gini

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gini = float('inf')
        n_features = X.shape[1]

        # Sample features for efficiency (use sqrt of total features)
        n_sample_features = min(int(math.sqrt(n_features)) + 1, n_features)
        feature_indices = np.random.choice(n_features, n_sample_features, replace=False)

        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])
            # Sample thresholds if too many
            if len(thresholds) > 20:
                thresholds = np.percentile(X[:, feature_idx], np.linspace(0, 100, 20))

            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                n = len(y)
                weighted_gini = (np.sum(left_mask) / n) * gini_left + (np.sum(right_mask) / n) * gini_right

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        # Stopping conditions
        if (depth >= self.max_depth or
            len(np.unique(y)) == 1 or
            len(y) < self.min_samples_split):
            # Leaf node: majority class
            values, counts = np.unique(y, return_counts=True)
            return {'leaf': True, 'class': values[np.argmax(counts)]}

        feature, threshold = self._best_split(X, y)

        if feature is None:
            values, counts = np.unique(y, return_counts=True)
            return {'leaf': True, 'class': values[np.argmax(counts)]}

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _predict_sample(self, x, tree):
        if tree['leaf']:
            return tree['class']
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

# PIPELINE FROM SCRATCH


class Pipeline:
    def __init__(self, steps):
        self.steps = steps  # list of (name, object) tuples

    def fit(self, X, y):
        X_transformed = X
        for name, step in self.steps[:-1]:
            X_transformed = step.fit_transform(X_transformed)
        # Fit the final model
        model_name, model = self.steps[-1]

        # Convert y to numpy array if needed
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        model.fit(X_transformed, y)

    def predict(self, X):
        X_transformed = X
        for name, step in self.steps[:-1]:
            X_transformed = step.transform(X_transformed)
        model_name, model = self.steps[-1]
        return model.predict(X_transformed)

X=df['text']
y=df['category'].apply(lambda x : 1 if x=='sport' else 0)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=101)

# Using all previously Build Models From Scratch

class Models_Training:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def logistic_regression(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=101)
        pipe_Logistic = Pipeline([
            ('tfidf_vectorization', TFIDF_Vectorizer()),
            ('model', LogisticRegression(lr=0.1, n_iters=500))])
        pipe_Logistic.fit(X_train, y_train)
        y_pred = pipe_Logistic.predict(X_test)
        print(classification_report(y_test, y_pred))

    def MultinomialNaiveBayes(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=101)
        pipe_MNB = Pipeline([
            ('tfidf_vectorization', TFIDF_Vectorizer()),
            ('model', MultinomialNB())])
        pipe_MNB.fit(X_train, y_train)
        y_pred = pipe_MNB.predict(X_test)
        print(classification_report(y_test, y_pred))

    def DecisionTree(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=101)
        pipe_DT = Pipeline([
            ('tfidf_vectorization', TFIDF_Vectorizer()),
            ('model', DecisionTree(max_depth=15, min_samples_split=5))])
        pipe_DT.fit(X_train, y_train)
        y_pred = pipe_DT.predict(X_test)
        print(classification_report(y_test, y_pred))

Models = Models_Training(X, y)
print(f"Required Logistic_Model's Performance are\n")
ans_1 = Models.logistic_regression()
print(f"\n")
print(f"Required Multi-NomialNaive Bayes _Model's Performance are\n")
ans_2 = Models.MultinomialNaiveBayes()
print(f"\n")
print(f"Required Decision Tree _Model's Performance are\n")
ans_3 = Models.DecisionTree()
print(f"\n")