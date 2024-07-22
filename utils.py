import os
from pydub import AudioSegment
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.preprocessing import LabelEncoder, StandardScaler

label_encoder = joblib.load('trained_models/speakerGender_KaggleVAD/label_encoder.pkl')
scaler = joblib.load('trained_models/speakerGender_KaggleVAD/scaler.pkl')


def convert_m4a_to_wav(file_path):
    # Check if the file has the correct extension
    if not file_path.lower().endswith('.m4a'):
        raise ValueError("The input file must be an .m4a file")

    # Define the output file path
    wav_file_path = file_path.rsplit('.', 1)[0] + '.wav'

    # Load the .m4a file
    audio = AudioSegment.from_file(file_path, format='m4a')
    print(wav_file_path)
    # Export as .wav
    audio.export(wav_file_path, format='wav')

    # Remove the original .m4a file
    os.remove(file_path)


class MLModelManager:
    def __init__(self, all_models=True, other_models = []):
        self.models = self.init_classification_MLModels(all_models, other_models)
    
    def init_classification_MLModels(self, all_models, other_models):
        '''
            Initialize classification ML models as dictionary.
            Individual models can be called by using keys of the dictionary

            Current models:
            'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'SVM': SVC(),
            'KNN': KNeighborsClassifier(),
            'AdaBoost': AdaBoostClassifier(),
            'Gradient Boosting': GradientBoostingClassifier()
        '''
        
        if all_models:
            models = {
                'Logistic Regression': LogisticRegression(),
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'SVM': SVC(),
                'KNN': KNeighborsClassifier(),
                'AdaBoost': AdaBoostClassifier(),
                'Gradient Boosting': GradientBoostingClassifier()
            }
        if other_models:
            for i in range(len(other_models)):
                models[f'model{i}'] = other_models[i]
        
        return models

    def train_classification_MLModels(self, X_train, y_train, X_test, y_test):
        '''
        Train and test classification ML models.
        Returns a dictionary of model performances.
        '''
        evaluation_results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            evaluation_results[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred)
            }
            print(f"---xxx---{name} is trained---xxx---")
        return evaluation_results

    def save_classification_MLModels(self, path):
        '''
        Save classification ML models to the specified path.
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        for name, model in self.models.items():
            model_path = os.path.join(path, f"{name}.joblib")
            joblib.dump(model, model_path)
            print(f"Saved {name} model to {model_path}")

    def load_classification_MLModels(self, path):
        '''
        Load classification ML models from the specified path.
        '''
        loaded_models = {}
        for name in self.models.keys():
            model_path = os.path.join(path, f"{name}.joblib")
            if os.path.exists(model_path):
                loaded_models[name] = joblib.load(model_path)
                print(f"Loaded {name} model from {model_path}")
            else:
                print(f"No saved model found for {name} at {model_path}")
        self.models = loaded_models



def format_labels_gender(labels):

