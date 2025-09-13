import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt

# 设置中文字体（适用于 Windows，优先选用 SimHei）
matplotlib.rcParams['font.family'] = 'SimHei'
# 解决负号无法显示的问题
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. 数据加载
train = pd.read_csv("./data/train.csv")

# 2. 数据预处理
def preprocess(df):
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Cabin"] = df["Cabin"].fillna("U")
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Deck"] = df["Cabin"].apply(lambda x: x[0])
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    df["Deck"] = df["Deck"].map({
        "A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6,
        "F": 2.0, "G": 2.4, "T": 2.8, "U": 1.5
    })
    return df

train = preprocess(train)
train = train.drop(["Cabin", "Name", "Ticket"], axis=1)

# 3. 特征与标签
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Deck"]
X = train[features]
y = train["Survived"]

# 4. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. 模型训练（Bagging + 随机森林）
rf = RandomForestClassifier(n_estimators=100, random_state=1)
model = BaggingClassifier(estimator=rf, n_estimators=20, random_state=1)
model.fit(X_scaled, y)

# 6. 混淆矩阵绘图
from sklearn.metrics import ConfusionMatrixDisplay

y_pred = cross_val_predict(model, X_scaled, y, cv=5)
cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['死亡', '生还'], yticklabels=['死亡', '生还'])
plt.title("Titanic 生存预测混淆矩阵")
plt.xlabel("预测标签")
plt.ylabel("实际标签")
plt.tight_layout()
plt.savefig("./images/titanic_confusion_matrix.png")  # 可选保存
plt.show()

# 7. 特征重要性绘图（基于随机森林）
rf.fit(X_scaled, y)
importances = rf.feature_importances_

plt.figure(figsize=(8, 6))
sns.barplot(x=importances, y=features, palette="viridis")
plt.title("Titanic 随机森林特征重要性")
plt.xlabel("重要性得分")
plt.ylabel("特征")
plt.tight_layout()
plt.savefig("./images/titanic_feature_importance.png")  # 可选保存
plt.show()

# 8. 输出交叉验证准确率
scores = cross_val_score(model, X_scaled, y, cv=5)
print("📊 Bagging + 随机森林 平均交叉验证准确率：", round(scores.mean(), 4))
