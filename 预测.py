import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

# 读取数据
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

# 数据预处理
def preprocess(df):
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Cabin"] = df["Cabin"].fillna("U")
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Deck"] = df["Cabin"].apply(lambda x: x[0])
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    df["Deck"] = df["Deck"].map({"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6,
                                 "F": 2.0, "G": 2.4, "T": 2.8, "U": 1.5})
    return df

train = preprocess(train)
test = preprocess(test)

# 删除无用列
train = train.drop(["Cabin", "Name", "Ticket"], axis=1)
test = test.drop(["Cabin", "Name", "Ticket"], axis=1)

# 特征列
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Deck"]

x_train = train[predictors]
y_train = train["Survived"]
x_test = test[predictors]
test_ids = test["PassengerId"]

# 标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 模型训练与预测
rf = RandomForestClassifier(n_estimators=100, random_state=1)
bagging = BaggingClassifier(base_estimator=rf, n_estimators=20)
bagging.fit(x_train, y_train)
predictions = bagging.predict(x_test)

# 保存结果
submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Survived": predictions
})
submission.to_csv("./data/submission.csv", index=False)
print("✅ 预测完成，结果已保存为ubmission.csv")
