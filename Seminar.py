#Seminar
import pandas as pd
# נטען את הספרייה על מנת לטעון קובץ csv
import numpy as np
# טעינה של הספרייה על מנת שימוש בפונקציות
import matplotlib.pyplot as plt
# הצגה של גרפים
from sklearn.metrics import plot_confusion_matrix
# הצגה של מטריצת הבילבול
from sklearn.model_selection import train_test_split
# פיצול הדאטה לקבוצת אימון וקבוצת בקרה
from sklearn.preprocessing import minmax_scale
# נרמול הדאטה לפי MinMax_Scale


Original_Data = pd.read_csv("DataSetNewCSV.csv")
# הדאטה המקורי ללא שינוי
DataSet = pd.read_csv("DataSetNewCSV.csv")
# דאטה סט המכיל 300 שורות ו9 עמודות
DataSet_Convert_Gander = pd.factorize(DataSet.gander)[0]
# המרה של ערכים קטגוראלים מתוך עמודת gander ל0 ו1 על מנת לבצע נרמול
DataSet ['gander'] = DataSet_Convert_Gander
# החלפה של העמודה המקורית בעמודה החדשה שיצרנו
DataSet_MinMaxScale = DataSet.iloc[:,range(1,9)]
# הכנת הדאטה על מנת לבצע נרמול שימוש בעמודות הרלוונטיות בלבד
# שימוש בכל העמודות פרט לעמודת הסיווג
Data_Scaled = minmax_scale(DataSet_MinMaxScale ,copy = True)
# נרמול הדאטה ל0 ו1 לפי שיטת MinMax_Scale
DataSet_target = DataSet.iloc[:,0]
# שליפת העמודה של הסיווגים של כל נקודה בדאטה
DataSet_target_Convert_number = pd.factorize(DataSet.Class)[0]
# המרה של עמודת הסיווגים לערכים מספריים
DataSet ['Class'] = DataSet_target_Convert_number
# החלפה של העמודה המקורית של הסיווג בעמודה המספרית שיצרנו
trainingSet,testSet,trainingSet_targets,testSet_targets = train_test_split(Data_Scaled,DataSet_target,test_size = 0.2 ,train_size = 0.8)
# פיצול הדאטה לקבוצת אימון וקבוצת בקרה על מנת לבצע שימוש במסווגים
# 20% מהווה קבוצת הבקרה ו80% מהווה קבוצת האימון


'-----------------------------------------------------'
#Knn algorithm

from sklearn.neighbors import KNeighborsClassifier

KNNClassifier = KNeighborsClassifier(3)
KNNClassifier.fit(trainingSet, trainingSet_targets)
KNNResults = KNNClassifier.predict(testSet)
print("The results of the Knn classifier for the test group are:\n\n",KNNResults)

print('\n')

from sklearn import metrics
print("metrics of Knn classifier:\n",metrics.classification_report(testSet_targets, KNNResults))
# הצגה של מדדי האיכות של המסווג

plot_confusion_matrix(KNNClassifier,testSet,testSet_targets)
plt.title('Knn Classifier')
plt.show()
#ConfusiomMatrixShow


print('\n')

baby = np.array(["The baby's condition is healthy and developing properly well done!" ,
                 "The cognitive problem can be solved by a number of ways:\n 1. Experience through the senses\n 2. Motional thinking\n 3. Exploration and operation of objects",
                 "The physiological problem can be solved by a number of ways:\n 1. Hold the baby head.\n 2. Lying the baby on his stomach.\n 3. Lying the baby on its back.\n 4. Lying the baby on its side."])
# הודעות הצגה למשתמש על מנת לשפר את המצב הקיים או הודעת עדכון כי המצב תקין
       
i = 0
for i in KNNResults[0:5]:
    if i == 'H':
        print(baby[0],'\n')
    if i == 'C' :
        print(baby[1],'\n')
    if i == 'P' :
        print(baby[2],'\n')
        
print('\n\n')
# לולאה על מנת להציג כל הודעה בהתאם על פי תוצאת הסיווג
# הצגה עבור 5 תוצאות סיווג לשם המחשת הרעיון

'-----------------------------------------------------'

#Naive Bayes algorithm

from sklearn.naive_bayes import GaussianNB
# טעינה של האלגוריתם הרצוי

NBClassifier = GaussianNB()
# שימוש באלגוריתם הרצוי
NBClassifier.fit(trainingSet, trainingSet_targets)
# תהליך הלמידה בין קבוצת האימון לסיווגים
NBResults = NBClassifier.predict(testSet)
# מציאת הסיווג של הנקודה החדשה מתוך קבוצת האימון
print("The results of the Naive Bayes classifier for the test group are:\n",NBResults)
# הדפסת התוצאה של המסווג

print('\n')

from sklearn import metrics
print("metrics of Naive Bayes classifier:\n",metrics.classification_report(testSet_targets, NBResults))
# הצגה של מדדי האיכות של המסווג

print('\n')

baby = np.array(["The baby's condition is healthy and developing properly well done!" ,
                 "The cognitive problem can be solved by a number of ways:\n 1. Experience through the senses\n 2. Motional thinking\n 3. Exploration and operation of objects",
                 "The physiological problem can be solved by a number of ways:\n 1. Hold the baby head.\n 2. Lying the baby on his stomach.\n 3. Lying the baby on its back.\n 4. Lying the baby on its side."])
# הודעות הצגה למשתמש על מנת לשפר את המצב הקיים או הודעת עדכון כי המצב תקין
       
i = 0
for i in NBResults[0:5]:
    if i == 'H':
        print(baby[0],'\n')
    if i == 'C' :
        print(baby[1],'\n')
    if i == 'P' :
        print(baby[2],'\n')
        
print('\n\n')
# לולאה על מנת להציג כל הודעה בהתאם על פי תוצאת הסיווג
# הצגה עבור 5 תוצאות סיווג לשם המחשת הרעיון

plot_confusion_matrix(NBClassifier,testSet,testSet_targets)
plt.title('Naive Bayes Classifier')
plt.show()
#ConfusiomMatrixShow



'-------------------------------------------------------'


#Decision Tree algorithm

from sklearn.tree import DecisionTreeClassifier
# טעינה של האלגוריתם הרצוי
from sklearn  import tree
# הצגה של העץ
DTClassifier = DecisionTreeClassifier(criterion='gini')
# שימוש באלגוריתם הרצוי על פי מדד gini
DTClassifier.fit(trainingSet, trainingSet_targets)
# תהליך הלמידה בין קבוצת האימון לסיווגים
DTResults = DTClassifier.predict(testSet) 
# מציאת הסיווג של הנקודה החדשה מתוך קבוצת האימון
print("The results of the Decision Tree classifier for the test group are:\n",DTResults)
# הדפסת התוצאה של המסווג

print('\n')

from sklearn import metrics
print("metrics of Decision Tree classifier:\n",metrics.classification_report(testSet_targets, DTResults))
# הצגה של מדדי האיכות של המסווג

print('\n')

baby = np.array(["The baby's condition is healthy and developing properly well done!" ,
                 "The cognitive problem can be solved by a number of ways:\n 1. Experience through the senses\n 2. Motional thinking\n 3. Exploration and operation of objects",
                 "The physiological problem can be solved by a number of ways:\n 1. Hold the baby head.\n 2. Lying the baby on his stomach.\n 3. Lying the baby on its back.\n 4. Lying the baby on its side."])
# הודעות הצגה למשתמש על מנת לשפר את המצב הקיים או הודעת עדכון כי המצב תקין
       
i = 0
for i in DTResults[0:5]:
    if i == 'H':
        print(baby[0],'\n')
    if i == 'C' :
        print(baby[1],'\n')
    if i == 'P' :
        print(baby[2],'\n')
        
# לולאה על מנת להציג כל הודעה בהתאם על פי תוצאת הסיווג
# הצגה עבור 5 תוצאות סיווג לשם המחשת הרעיון


plot_confusion_matrix(DTClassifier,testSet,testSet_targets)
plt.title('Decision Tree Classifier')
plt.show()
#ConfusiomMatrixShow

plt.figure(figsize = (40,45))
# פריסת הצגת העץ
tree.plot_tree(DTClassifier,max_depth = None, fontsize = 16,filled = True,class_names = ['H','P','C'])
plt.show()
# הצגה גרפית של העץ



'---------------------------------------------'

# הצגה של מטריצת הקורלציה בין כל עמודה ועמודה לא כולל עמודת הסיווג
correlationMatrix = DataSet_MinMaxScale.corr()
print('\n')
print('The strongest relationship between columns according to the calculation of the correlation is:\nAge and height ')
# הדפסת שני העמודות בעלות הקשר החזק ביותר על פי מטריצת הקורלציה

# הצגה של קו הרגרסיה הלינארית על מנת להמחיש את התלות בין שני העמודות גיל וגובה
# ניתן לראות על פי הגרף כי הקשר הוא חזק וחיובי
Height = DataSet.iloc[:,5].values
Age = DataSet.iloc[:,7].values

n = len(Height)
sumX = np.sum(Height)
sumY = np.sum(Age)
sumX2 = np.sum(Height**2)
sumXY = np.sum(Height*Age)
sumY2 = np.sum(Age**2)

a = (n*sumXY - sumX*sumY)/(n*sumX2-(sumX**2))
b = (sumY*sumX2-sumX*sumXY)/(n*sumX2-(sumX**2))
# y = ax+b

x_plot = np.linspace(0,200,100)
y_hat = a*x_plot + b
plt.scatter(Height,Age)
plt.plot(x_plot,y_hat)
plt.xlabel('Height')
plt.ylabel('Age')
plt.title('Linear regression')

'-------------------------------------------------------'

# הצגה של מדד הקורלציה בין כל עמודה בנפרד לעמודת הסיווג
# בדיקת התלות בין כל עמודה לעמודת הסיווג
Data_Corr = DataSet.corr()
correlationMatrix_Classfier = Data_Corr.iloc[:,0]

'----------------------------------------------------------'

# הצגה גרפית של סך כל התינוקות ומצבם הבריאותי
Healthy = np.sum(Original_Data.iloc[:,0] == 'H')
CognitiveProblem = np.sum(Original_Data.iloc[:,0] == 'C')
PhysiologicalProblem =  np.sum(Original_Data.iloc[:,0] == 'P')
df = pd.DataFrame({'':['Healthy', 'CognitiveProblem' ,'PhysiologicalProblem'],'Quantity': [Healthy, CognitiveProblem ,PhysiologicalProblem]})
Total_Babies_HealthCondition = df.plot.bar(x = '' ,y = 'Quantity' , rot = 0)
plt.ylim([0, 150])
plt.title('Developmental status')


# הצגת דיאגרמת עוגה של סך כל התינוקות באחוזים לפי מגדר
Male = np.sum(Original_Data.iloc[:,8] == 'Male')
Female = np.sum(Original_Data.iloc[:,8] == 'Female')
df1 = pd.DataFrame({'':[Male, Female]}, index = ['Male', 'Female'])
plot_Gander_Pie = df1.plot(kind='pie', subplots=True, shadow = True,startangle=90,figsize=(5,5), autopct='%1.1f%%')
plt.title('Gander')
  