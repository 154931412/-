import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score

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
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Deck"]
X_train = train[features]
y_train = train["Survived"]
X_test = test[features]
test_ids = test["PassengerId"]

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 网格搜索优化随机森林超参数
rf = RandomForestClassifier(random_state=1)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print("✅ 最佳参数：", grid_search.best_params_)

# 用优化后的随机森林作为 Bagging 的基学习器
bagging = BaggingClassifier(estimator=best_rf, n_estimators=15, random_state=1)
scores = cross_val_score(bagging, X_train, y_train, cv=5)
print("📊 Bagging + RF 交叉验证准确率：", scores.mean())

# 最终模型拟合 + 预测
bagging.fit(X_train, y_train)
predictions = bagging.predict(X_test)

# 保存结果
submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Survived": predictions
})
submission.to_csv("./data/titanic_submission_optimized.csv", index=False)
print("✅ 提交文件已保存为 titanic_submission_optimized.csv")
