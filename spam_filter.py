from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
import pandas
import csv
import random

data = pandas.read_csv('../spam.csv', encoding='latin-1')
more_data = pandas.read_csv('../small_sms_spam_messages.csv', encoding='latin-1')

training_part = int(len(data) * 0.7)
train_data = data[:training_part]
test_data = data[training_part:]

classifier = SGDClassifier(loss='log')
vectorizer = TfidfVectorizer()

# train
vectorize_text = vectorizer.fit_transform(train_data.v2)
classifier.partial_fit(vectorize_text, train_data.v1, classes=['spam', 'ham'])

# score
vectorize_text = vectorizer.transform(test_data.v2)
score = classifier.score(vectorize_text, test_data.v1)
print(score) # 98,8

t1 = 'WRewards: Swipe your Woolies card from 12-20 Oct to win 2 tickets to the rugby finals in Japan. Get 1 entry for every day you shop! T&Cs apply bit.ly/2OKl2Ua'
vectorize_text = vectorizer.transform([t1])
predict = classifier.predict_proba(vectorize_text)[0]
print('{} - {} - {}'.format(t1, predict, classifier.predict(vectorize_text)))

vectorize_text = vectorizer.transform(more_data.v2)
classifier.partial_fit(vectorize_text, more_data.v1)

# score
vectorize_text = vectorizer.transform(test_data.v2)
score = classifier.score(vectorize_text, test_data.v1)
print(score)

t1 = 'WRewards: Swipe your Woolies card from 12-20 Oct to win 2 tickets to the rugby finals in Japan. Get 1 entry for every day you shop! T&Cs apply bit.ly/2OKl2Ua'
vectorize_text = vectorizer.transform([t1])
predict = classifier.predict_proba(vectorize_text)
print('{} - {} - {}'.format(t1, predict, classifier.predict(vectorize_text)))

t1 = 'You are 100% special to me, lets go to Woolies'
vectorize_text = vectorizer.transform([t1])
predict = classifier.predict_proba(vectorize_text)
print('{} - {} - {}'.format(t1, predict, classifier.predict(vectorize_text)))

csv_arr = []
for index, row in test_data.iterrows():
    answer = row[0]
    text = row[1]
    vectorize_text = vectorizer.transform([text])
    predict = classifier.predict(vectorize_text)[0]
    if predict == answer:
        result = 'right'
    else:
        result = 'wrong'
    csv_arr.append([len(csv_arr), text, answer, predict, result])


# write csv
with open('test_score.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=';',
            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['#', 'text', 'answer', 'predict', result])

    for row in csv_arr:
        spamwriter.writerow(row)
