#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Cw 6.1
import numpy as np

data = np.genfromtxt("wine.txt",dtype=float, delimiter=',')
X = data[:,1:len(data[0])]
y = data[:,0]

from sklearn.preprocessing import KBinsDiscretizer
bins = 4 #poziomow bo jak nie bd w zbiorze wartosci danej to nie bd zwracac dla tego prawdopodobienstwa warunkowego
est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
est.fit(X)
Xt = est.transform(X)   #zbior uczacy zdyskretyzowany 


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.33, random_state=41)   #zbior testowy


from sklearn.base import BaseEstimator, ClassifierMixin

                        
class BayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    def fit(self, X, y, bins, laPlace = False): #przekazuje fit liczbe przedzialow wartosci
        self.X = X
        self.y = y
        self.bins = bins
        self.laPlace = laPlace
        self.iloscKlas = len(np.unique(y))
        self.klasy = np.unique(y)
        self.iloscAtrybutow = self.X.shape[1]
        #obliczenie a priori
        self.aPriori = np.array([])
        for klasa in np.unique(self.y):
            classProb = len(self.y[self.y == klasa])/len(self.y)
            self.aPriori = np.append(self.aPriori, classProb)
        #obliczenie rozk warunkowego
        self.rozkWarunkowy = np.zeros((self.iloscKlas, self.iloscAtrybutow, self.bins))
        itKlasy = 0 
        for klasa in np.unique(self.y): # w klasie
            itAtryb = 0
            for atryb in self.X[self.y==klasa].T: #dla kazdego atrybutu inne prawdopodobienstwo warunkowe   # w kolumnie
                for wartosc in range(self.bins): 
                    if(self.laPlace):
                        pWar = (len(atryb[atryb == wartosc]) + 1) / (len(atryb) + self.bins)
                    else:
                        pWar = len(atryb[atryb == wartosc])/len(atryb)
                    self.rozkWarunkowy[itKlasy][itAtryb][wartosc] = pWar
                itAtryb += 1
            itKlasy += 1
            
        #print(self.rozkWarunkowy.shape)
    
    
    def predict(self, X):
        iloscProbek = X.shape[0]
        aPosteriori = np.ones((iloscProbek, self.iloscKlas))
        probkaIdx = 0
        for probka in X:
            for klasaIdx in range(self.iloscKlas):
                for atrybutIdx in range(self.iloscAtrybutow):
                    aPosteriori[probkaIdx][klasaIdx] *= self.rozkWarunkowy[klasaIdx][atrybutIdx][int(probka[atrybutIdx])]
                aPosteriori[probkaIdx][klasaIdx] *= self.aPriori[klasaIdx]
            probkaIdx += 1
        return self.klasy[np.argmax(aPosteriori, axis=1)]
    
    def predict_proba(self, X):
        iloscProbek = X.shape[0]
        aPosteriori = np.ones((iloscProbek, self.iloscKlas))
        probkaIdx = 0
        for probka in X:
            for klasaIdx in range(self.iloscKlas):
                for atrybutIdx in range(self.iloscAtrybutow):
                    aPosteriori[probkaIdx][klasaIdx] *= self.rozkWarunkowy[klasaIdx][atrybutIdx][int(probka[atrybutIdx])]
                aPosteriori[probkaIdx][klasaIdx] *= self.aPriori[klasaIdx]
            probkaIdx += 1
        
        prob = np.zeros((aPosteriori.shape[0],aPosteriori.shape[1]))
        for i in range(len(prob)):
            for j in range(len(prob[0])):
                if(not self.laPlace and (np.sum(aPosteriori[i]) == 0)):   #gdy iloczyn = 0 bo brak poprawki LapPlace, zmieniam prawodpodobienstwa klas na takie same
                    prob[i] = 1/self.iloscKlas
                    break
                else:
                    prob[i][j]  = (aPosteriori[i][j]) /np.sum(aPosteriori[i])
        return prob

bc= BayesClassifier()

bc.fit(X_train, y_train, bins, False)
print("bez poprawki LaPlace")
print("dokladnosc na zbiorze uczacym: ", bc.score(X_train, y_train))
print("dokladnosc na zbiorze testowym: ", bc.score(X_test, y_test))
print("z poprawka LaPlace")
bc.fit(X_train, y_train, bins, True)
print("dokladnosc na zbiorze uczacym: ", bc.score(X_train, y_train))
print("dokladnosc na zbiorze testowym: ", bc.score(X_test, y_test))
print("dla malych zbiorow(tutaj zbioru uczacego) poprawka LaPlace nieznacznie psuje dokladnosc,\nale dla danych testowych zwieksza ja \n")  

#prob = bc.predict_proba(X_test)
#np.amax(prob, axis=1)
#print(bc.get_params())


# In[4]:


#Cw 6.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=41)   #zbior testowy

class BayesClassifierContinuousData(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    def fit(self, X, y): #przekazuje fit liczbe przedzialow wartosci
        self.X = X
        self.y = y
        self.iloscKlas = len(np.unique(y))
        self.klasy = np.unique(y)
        self.iloscAtrybutow = self.X.shape[1]
        #obliczenie a priori
        self.aPriori = np.array([])
        for klasa in np.unique(self.y):
            classProb = len(self.y[self.y == klasa])/len(self.y)
            self.aPriori = np.append(self.aPriori, classProb)
        #obliczenie srednich i odchylen
        self.srednia = np.zeros((self.iloscKlas, self.iloscAtrybutow)) #srednia
        self.odchylenie = np.zeros((self.iloscKlas, self.iloscAtrybutow)) #odchylenie
        itKlasy = 0 
        for klasa in np.unique(self.y): # w klasie
            itAtryb = 0
            for atryb in self.X[self.y==klasa].T: #w tej petli dla kazdego atrybutu inna srednia   # w kolumnie
                self.srednia[itKlasy][itAtryb] = np.mean(atryb)
                srednia = self.srednia[itKlasy][itAtryb]
                self.odchylenie[itKlasy][itAtryb] = np.sqrt(np.sum((atryb-srednia)**2)/(atryb.shape[0]-1))
                itAtryb += 1
            itKlasy += 1
    
    
    def predict(self, X):
        iloscProbek = X.shape[0]
        aPosteriori = np.ones((iloscProbek, self.iloscKlas))
        probkaIdx = 0
        for probka in X:
            for klasaIdx in range(self.iloscKlas):
                for atrybutIdx in range(self.iloscAtrybutow):
                    fWykladnicza = np.exp(-((probka[atrybutIdx]-self.srednia[klasaIdx][atrybutIdx])**2)/(2*self.odchylenie[klasaIdx][atrybutIdx]**2))
                    aPosteriori[probkaIdx][klasaIdx] *= 1/(self.odchylenie[klasaIdx][atrybutIdx]*np.sqrt(2*np.pi)) * fWykladnicza
                aPosteriori[probkaIdx][klasaIdx] *= self.aPriori[klasaIdx]
            probkaIdx += 1
        return self.klasy[np.argmax(aPosteriori, axis=1)]
    
    def predict_proba(self, X):
        iloscProbek = X.shape[0]
        aPosteriori = np.ones((iloscProbek, self.iloscKlas))
        probkaIdx = 0
        for probka in X:
            for klasaIdx in range(self.iloscKlas):
                for atrybutIdx in range(self.iloscAtrybutow):
                    fWykladnicza = np.exp(-((probka[atrybutIdx]-self.srednia[klasaIdx][atrybutIdx])**2)/(2*self.odchylenie[klasaIdx][atrybutIdx]**2))
                    aPosteriori[probkaIdx][klasaIdx] *= 1/(self.odchylenie[klasaIdx][atrybutIdx]*np.sqrt(2*np.pi)) * fWykladnicza
                aPosteriori[probkaIdx][klasaIdx] *= self.aPriori[klasaIdx]
            probkaIdx += 1
            
        prob = np.zeros((aPosteriori.shape[0],aPosteriori.shape[1]))
        for i in range(len(prob)):
            for j in range(len(prob[0])):
                prob[i][j]  = (aPosteriori[i][j]) /np.sum(aPosteriori[i])
            
        return prob
    
bcCiagly = BayesClassifierContinuousData()
bcCiagly.fit(X_train, y_train)
print("wariant ciagly")
print("dokladnosc na zbiorze testowym: ", bcCiagly.score(X_test, y_test)) 


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("wariant ciagly - gotowa implementacja")
print("dokladnosc na zbiorze testowym: ", gnb.score(X_test,y_test)) 

X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.33, random_state=41)   #zbior testowy
bc.fit(X_train, y_train, bins, False)
print("wariant dyskretny bez poprawki LaPlace")
print("dokladnosc na zbiorze testowym: ", bc.score(X_test, y_test))
print("wariant dyskretny z poprawka LaPlace")
bc.fit(X_train, y_train, bins, True)
print("dokladnosc na zbiorze testowym: ", bc.score(X_test, y_test))
print("dokladnosc klasyfikatora Bayesa na danych ciaglych jest wieksza")



# In[5]:


#Cw 6.4

print("Przyklad sytuacji niebezpiecznej:")
X = data[:,1:len(data[0])]
#powtarzanie kolumn
for i in range(7):
    X = np.concatenate((X,X),axis=1)
#print(X.shape)
y = data[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=41)   #zbior testowy
bcCiagly = BayesClassifierContinuousData()
bcCiagly.fit(X_train, y_train)
print("wariant ciagly")
print("dokladnosc na zbiorze testowym: ", bcCiagly.score(X_test, y_test)) 

bins = 4
est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
est.fit(X)
Xt = est.transform(X)   #zbior uczacy zdyskretyzowany 



X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.33, random_state=41)   #zbior testowy

bc.fit(X_train, y_train, bins, True)
print("wariant dyskretny")
print("dokladnosc na zbiorze testowym: ", bc.score(X_test, y_test))


# In[6]:


class BayesClassifierSafe(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    def fit(self, X, y, bins, laPlace = False): #przekazuje fit liczbe przedzialow wartosci
        self.X = X
        self.y = y
        self.bins = bins
        self.laPlace = laPlace
        self.iloscKlas = len(np.unique(y))
        self.klasy = np.unique(y)
        self.iloscAtrybutow = self.X.shape[1]
        #obliczenie a priori
        self.aPriori = np.array([])
        for klasa in np.unique(self.y):
            classProb = len(self.y[self.y == klasa])/len(self.y)
            self.aPriori = np.append(self.aPriori, classProb) 
        #obliczenie rozk warunkowego
        self.rozkWarunkowy = np.zeros((self.iloscKlas, self.iloscAtrybutow, self.bins))
        itKlasy = 0 
        for klasa in np.unique(self.y): # w klasie
            itAtryb = 0
            for atryb in self.X[self.y==klasa].T: #dla kazdego atrybutu inne prawdopodobienstwo warunkowe   # w kolumnie
                for wartosc in range(self.bins): 
                    if(self.laPlace):
                        pWar = (len(atryb[atryb == wartosc]) + 1) / (len(atryb) + self.bins)
                    else:
                        
                        pWar = len(atryb[atryb == wartosc])/len(atryb)
                        #bez poprawki LaPlace log(pWar) mogloby przyjac warto -inf 
                        #print(np.log(pWar))
                        #print(len(atryb[atryb == wartosc])/len(atryb))
                        #print(len(atryb))
                    self.rozkWarunkowy[itKlasy][itAtryb][wartosc] = pWar
                itAtryb += 1
            itKlasy += 1
            
    
    
    def predict(self, X):
        iloscProbek = X.shape[0]
        aPosteriori = np.zeros((iloscProbek, self.iloscKlas))
        probkaIdx = 0
        for probka in X:
            for klasaIdx in range(self.iloscKlas):
                for atrybutIdx in range(self.iloscAtrybutow):
                    aPosteriori[probkaIdx][klasaIdx] += np.log(self.rozkWarunkowy[klasaIdx][atrybutIdx][int(probka[atrybutIdx])])
                aPosteriori[probkaIdx][klasaIdx] += np.log(self.aPriori[klasaIdx])
            probkaIdx += 1
        return self.klasy[np.argmax(aPosteriori, axis=1)]
    
    def predict_proba(self, X):
        iloscProbek = X.shape[0]
        aPosteriori = np.zeros((iloscProbek, self.iloscKlas))
        probkaIdx = 0
        for probka in X:
            for klasaIdx in range(self.iloscKlas):
                for atrybutIdx in range(self.iloscAtrybutow):
                    aPosteriori[probkaIdx][klasaIdx] += np.log(self.rozkWarunkowy[klasaIdx][atrybutIdx][int(probka[atrybutIdx])])
                aPosteriori[probkaIdx][klasaIdx] += np.log(self.aPriori[klasaIdx])
            probkaIdx += 1
        
        prob = np.zeros((aPosteriori.shape[0],aPosteriori.shape[1]))
        for i in range(len(prob)):
            for j in range(len(prob[0])):
                if(not self.laPlace and (np.sum(aPosteriori[i]) == 0)):   #gdy iloczyn = 0 bo brak poprawki LapPlace, zmieniam prawodpodobienstwa klas na takie same
                    prob[i] = 1/self.iloscKlas
                    break
                else:
                    prob[i][j]  = (aPosteriori[i][j]) /np.sum(aPosteriori[i])
        return prob

bcSafe = BayesClassifierSafe()
bcSafe.fit(X_train, y_train, bins, True)
print("wersja bezpieczna")
print("wersja dyskretna")
print("dokladnosc na zbiorze testowym: ", bcSafe.score(X_test, y_test))


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=41)   #zbior testowy

class BayesClassifierContinuousSafe(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    def fit(self, X, y): #przekazuje fit liczbe przedzialow wartosci
        self.X = X
        self.y = y
        self.iloscKlas = len(np.unique(y))
        self.klasy = np.unique(y)
        self.iloscAtrybutow = self.X.shape[1]
        #obliczenie a priori
        self.aPriori = np.array([])
        for klasa in np.unique(self.y):
            classProb = len(self.y[self.y == klasa])/len(self.y)
            self.aPriori = np.append(self.aPriori, classProb)
        #obliczenie srednich i odchylen
        self.srednia = np.zeros((self.iloscKlas, self.iloscAtrybutow)) #srednia
        self.odchylenie = np.zeros((self.iloscKlas, self.iloscAtrybutow)) #odchylenie
        itKlasy = 0 
        for klasa in np.unique(self.y): # w klasie
            itAtryb = 0
            for atryb in self.X[self.y==klasa].T: #w tej petli dla kazdego atrybutu inna srednia   # w kolumnie
                self.srednia[itKlasy][itAtryb] = np.mean(atryb)
                srednia = self.srednia[itKlasy][itAtryb]
                self.odchylenie[itKlasy][itAtryb] = np.sqrt(np.sum((atryb-srednia)**2)/(atryb.shape[0]-1))
                itAtryb += 1
            itKlasy += 1
    
    
    def predict(self, X):
        iloscProbek = X.shape[0]
        aPosteriori = np.zeros((iloscProbek, self.iloscKlas))
        probkaIdx = 0
        for probka in X:
            for klasaIdx in range(self.iloscKlas):
                for atrybutIdx in range(self.iloscAtrybutow):
                    drogaRoznica = (probka[atrybutIdx]-self.srednia[klasaIdx][atrybutIdx])**2/(2*self.odchylenie[klasaIdx][atrybutIdx]**2)
                    aPosteriori[probkaIdx][klasaIdx] += -np.log(self.odchylenie[klasaIdx][atrybutIdx]) - drogaRoznica
                aPosteriori[probkaIdx][klasaIdx] += np.log(self.aPriori[klasaIdx])
            probkaIdx += 1
        return self.klasy[np.argmax(aPosteriori, axis=1)]
    
    def predict_proba(self, X):
        iloscProbek = X.shape[0]
        aPosteriori = np.zeros((iloscProbek, self.iloscKlas))
        probkaIdx = 0
        for probka in X:
            for klasaIdx in range(self.iloscKlas):
                for atrybutIdx in range(self.iloscAtrybutow):
                    drogaRoznica = (probka[atrybutIdx]-self.srednia[klasaIdx][atrybutIdx])**2/(2*self.odchylenie[klasaIdx][atrybutIdx]**2)
                    aPosteriori[probkaIdx][klasaIdx] += -np.log(self.odchylenie[klasaIdx][atrybutIdx]) - drogaRoznica
                aPosteriori[probkaIdx][klasaIdx] += self.aPriori[klasaIdx]
            probkaIdx += 1
            
        prob = np.zeros((aPosteriori.shape[0],aPosteriori.shape[1]))
        for i in range(len(prob)):
            for j in range(len(prob[0])):
                prob[i][j]  = (aPosteriori[i][j]) /np.sum(aPosteriori[i])
                    
        return prob

    
bcCiagly = BayesClassifierContinuousSafe()
bcCiagly.fit(X_train, y_train)
print("wersja bezpieczna")
print("wariant ciagly")
print("dokladnosc na zbiorze testowym: ", bcCiagly.score(X_test, y_test)) 


# In[9]:


#Cw 6.3
#link do zbioru 
#https://archive.ics.uci.edu/ml/machine-learning-databases/00194/
# dataFloat = np.genfromtxt("sensor_readings_24.csv",dtype=float, delimiter=',')
# X = dataFloat[:,0:len(dataFloat[0])-1]
# print(dataFloat.shape)
# dataStr = np.genfromtxt("sensor_readings_24.csv",dtype=str, delimiter=',', usecols=24)
# y=dataStr




from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target
print("wersja dyskretna")
#bins = 2
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.07)   #zbior testowy
    bc = BayesClassifierContinuousSafe()

    bc.fit(X_train, y_train)
    print(bc.score(X_test, y_test))
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    print("dostepny z neta: ", gnb.score(X_test,y_test)) 
# while(bins < 84):
#     #est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
#     #est.fit(X)
#     #Xt = est.transform(X)   #zbior uczacy zdyskretyzowany 

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.07, random_state=41)   #zbior testowy
#     bc = BayesClassifierContinuousSafe()

#     bc.fit(X_train, y_train, bins, False)
#     print("przedzialow: ", bins, end="   ")
#     print("bez LaPlace: ", bc.score(X_test, y_test),end=" ")
#     bc.fit(X_train, y_train, bins, True)
#     print("LaPlace: ", bc.score(X_test, y_test))
#     bins += 5

    


# In[31]:





# In[ ]:




