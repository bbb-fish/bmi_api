import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

le = LabelEncoder()
std_sc = StandardScaler()

# read data
data = pd.read_csv('dataset.csv')
df = pd.DataFrame(data)
print(df.sample(10))

# convert gender from string to binary 1 or 0
df['Gender'] = le.fit_transform(df['Gender'])

# scale features for better model training
df.iloc[:,0:-1] = std_sc.fit_transform(df.iloc[:,0:-1])

# check data
print(df.sample(10))

# split out training data
X = df.drop(columns=['Index'])
Y = df['Index']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

# ensemble learning (random forest classification)
model = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
model.fit(x_train, y_train)

# predict results
y_pred_rfc = model.predict(x_test)

# measure accuracy
rfc_cm = confusion_matrix(y_test, y_pred_rfc)
print(rfc_cm)

rfc_acc = accuracy_score(y_test, y_pred_rfc)
print(rfc_acc*100)


# test person's health status based on model
def predict_mpg(config, model):
    
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config
    
    y_pred = model.predict(df)
    
    if y_pred == 0:
        return 'Extremely Weak'
    elif y_pred == 1:
        return 'Weak'
    elif y_pred == 2:
        return 'Normal'
    elif y_pred == 3:
        return 'Overweight'
    elif y_pred == 4:
        return 'Obesity'
    elif y_pred == 5:
        return 'Extreme Obesity'

config = {
    'Gender': [1], # 1- Male, 2- Female
    'Height': [177],
    'Weight': [188]
}

test_output = predict_mpg(config, model)
print(test_output)

# create model file
# save file to the current working directory
pkl_filename = 'model.pkl'
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

# load from file
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

# calculate the accuracy score and predict target values
score = pickle_model.score(x_test, y_test)
print("test score: {0:2f} %".format(100 * score))
Ypredict = pickle_model.predict(x_test)

## loading the moedl from the saved file
pkl_filename = "model.pkl"
with open(pkl_filename, 'rb') as f_in:
    model = pickle.load(f_in)

predictValue = predict_mpg(config, model)
print(predictValue)