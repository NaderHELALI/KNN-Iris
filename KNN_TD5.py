import os
import csv
import math as m 

os.chdir("C:/Users/Nader/Documents/Cours Esilv/SEMESTRE 2/Datascience et IA/TD5_KNN")

#Charge les donnees dans le ficher'iris.txt'
def LoadData(nomFichier):
    Mesure=[]
    Class=[]
    with open(nomFichier,'r') as f:
        donnee=csv.reader(f)
        
        for col in donnee:
            Mesure.append(col[:-1])            
            Class.append(col[-1:])
    return Mesure,Class

#Fonction qui calcule la distance Euclidienne
def DistanceEuclidienne(EchantillonTest,EchantillonX):
    somme=0
    for i in range(len(EchantillonTest)):
        somme+=(float(EchantillonTest[i])-float(EchantillonX[i]))**2
    return m.sqrt(somme)

#Methode qui retourne la distance entre chaque caratéristique
def DistanceCaracteristique(ListeA,ListeNA):
    distance=[]
    for i,valNA in enumerate(ListeNA) :
        distance.append([])
        for valA in ListeA :
            distance[i].append(float(DistanceEuclidienne(valNA,valA)))
    return distance

#cherche les indices des k plus petites valeurs de liste
def kminIndice(liste,k):
    KminIndice=[]
    #Kmin=[]
    for h in range(k): 
        mini=liste[0]
        pos=0
        for i in range(len(liste)): 
            if liste[i]<=mini and (i not in KminIndice):
                mini=liste[i]
                pos=i
        KminIndice.append(pos)
                
    return KminIndice 

#PourcentageAcquis : Exemple pour 70% -> pourcentageAcquis=70
def Selectionnelesdonnes(liste,pourcentageAcquis):
    D_Aquis=[]
    D_NAquis=[]
    for i in range(int(len(liste)*pourcentageAcquis/100)):
        D_Aquis.append(liste[i])
    for i in range(int(len(liste)*pourcentageAcquis/100),(len(liste))):
        D_NAquis.append(liste[i])
    return D_Aquis,D_NAquis

#prédiction de la nature d'une liste d'individus
def predictionNature (liste): 
    NatureNAPredite=[]
    for k in range(len(liste)):
        nbSetosa = 0
        nbVirginica = 0
        nbVersicolor = 0
        #Compte la nature des voisins 
        for j in range(len(liste[k])):
            for i in range(len(liste[k][j])):
                if liste[k][j][i] =='Iris-versicolor':
                    nbVersicolor += 1
                elif liste[k][j][i] == 'Iris-setosa':
                    nbSetosa +=1
                elif liste[k][j][i] =='Iris-virginica':
                   nbVirginica += 1
         #Notre prediction est choisis en fonction du max de la nature des PPV 
        if max(nbSetosa, nbVirginica, nbVersicolor) == nbSetosa:
                NatureNAPredite.append('Iris-setosa')
        elif max(nbSetosa, nbVirginica, nbVersicolor) == nbVirginica:
                NatureNAPredite.append('Iris-virginica')
        elif max(nbSetosa, nbVirginica, nbVersicolor) == nbVersicolor:
                NatureNAPredite.append('Iris-versicolor')
    return(NatureNAPredite)

#Fonction qui determine le taux de reussite de prediction
def Erreur(ListeNA,ListePrediction):
    nberreur=0
    nbval=len(ListeNA)
    if len(ListeNA)!=len(ListePrediction):
        nbval-=1
    for i in range(nbval):
        if ListeNA[i][0]!=ListePrediction[i] :
            nberreur+=1
    taux=(nbval-nberreur)/nbval
    return taux
        
#Fonction qui calcule le nombre de relation entre les valeurs exacts et celle predite
#Et retourne matrice         
def MatriceRelationnel (listeDNA, listePredictions):
    compteurSS= 0
    compteurSVir= 0
    compteurSVers = 0
    compteurVirS=0
    compteurVirVir=0
    compteurVirVers=0
    compteurVersS=0
    compteurVersVir=0
    compteurVersVers=0

    for i in range(len(listePredictions)-1):
        if listeDNA[i][0] == "Iris-setosa":
            if listePredictions[i] == "Iris-setosa":
                compteurSS +=1
            elif listePredictions[i] == "Iris-virginica":
                compteurSVir +=1
            elif listePredictions[i] == "Iris-versicolor":
                compteurSVers +=1

        if listeDNA[i][0] == "Iris-virginica":
            if listePredictions[i] == "Iris-setosa":
                compteurVirS +=1
            elif listePredictions[i] == "Iris-virginica":
                compteurVirVir +=1
            elif listePredictions[i] == "Iris-versicolor":
                compteurVirVers +=1

        if listeDNA[i][0] == "Iris-versicolor":
            if listePredictions[i] == "Iris-setosa":
                compteurVersS +=1
            elif listePredictions[i] == "Iris-virginica":
                compteurVersVir +=1
            elif listePredictions[i] == "Iris-versicolor":
                compteurVersVers +=1


    print ('Matrice de Confusion:\n')
    print(compteurSS, '|',compteurSVers,'|',compteurSVir)
    print(compteurSVers, '|',compteurVersVers,'|',compteurVersVir)
    print(compteurVirS, '|',compteurVirVers,'|',compteurVirVir)
            

# K-NN: NEAREST NEIGHBORS
def knn(k,pourcentageAcquis):
    
    #Récupère les données du fichier
    Mesure,Nature =LoadData("iris.txt")
    
    #Définit les données des aquis et non aquis(ceux sur lesquels on fera le prediction) 
    ListeA,ListeNA = Selectionnelesdonnes(Mesure,pourcentageAcquis)
    NatureA,NatureNA = Selectionnelesdonnes(Nature,pourcentageAcquis)
    distance=DistanceCaracteristique(ListeA,ListeNA)
    
    #Stock les indices des k  plus proches voisins 
    KPPDistances=[kminIndice(dist,k)for dist in distance]
    
    #Stock la Nature des k  plus proches voisins :
    NatureKPPV=[]
    for i in range(len(KPPDistances)):
        NatureKPPV.append([])
        for j in range(k):
            NatureKPPV[i].append(Nature[KPPDistances[i][j]])
    
    #RECUPERE la liste de prediction des valeurs non acquises
    predictionNatureNA=predictionNature(NatureKPPV)
    
    tauxErreur=Erreur(NatureNA,predictionNatureNA)
    print(f'Le taux de bonne prédiction est de {round(tauxErreur*100,3)}%.')
    
    #Affiche la Matrice de Confusion
    MatriceRelationnel(NatureNA,predictionNatureNA)
    
#Execute le KNN
if __name__=='__main__':
    
    knn(1,60)
    