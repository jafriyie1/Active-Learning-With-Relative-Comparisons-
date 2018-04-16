import pandas as pd
from itertools import combinations
import numpy as np
import pickle
import re
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import grid_search
from sklearn.model_selection import train_test_split



def triple_generator(data):
    """ This function returns the
    generated triples
    input:
    data (list of tuples) from original data
    """
    triple = []
    for pair in combinations(data,3):
        triple.append(pair)
    return triple


def prep_triple(df):
    """This function formats the data
    into triples for future consumption
    input:
    df (pandas DataFrame) original data source
    """
    sample_data = []
    df = df.values.tolist()

    for i in range(len(df)):
        temp = tuple(df[i])
        sample_data.append(temp)

    triple = triple_generator(sample_data)
    return triple


def euclidean_distance_triple(triple):
    """ This function returns X features of
    triples and the label y that was suggested from
    Schultz and Joachims relative comparison paper for
    Support Vector Machines
    input:
    triple (list of tuples) contains the triple
    for relative comparison
    """
    final = []
    for i in range(len(triple)):
        label = 0
        #print(triple[i])
        x1, x2, x3 = triple[i]
        x1 = np.array(x1)
        x2 = np.array(x2)
        x3 = np.array(x3)

        dist1 = np.linalg.norm(x1-x2)
        dist2 = np.linalg.norm(x1-x3)

        if dist1 > dist2:
            label = [1]
            final.append((triple[i], label))
        elif dist1 < dist2:
            label = [-1]
            final.append((triple[i], label))
        else:
            continue

    # split data
    X_full = []
    y = []

    for x in final:
        X, label = x
        y.append(label)
        X_full.append(X)

    return X_full, y

def euclidean_distance_on_instance(triple):
    """ This function returns X features of
    triples and the label y that was suggested from
    Schultz and Joachims relative comparison paper for
    Support Vector Machines
    input:
    triple (list of tuples) contains the triple
    for relative comparison
    """
    final = []
    for i in range(len(triple)):
        label = 0
        #print(triple[i])
        x1, x2, x3 = triple[i]
        x1 = np.array(x1)
        x2 = np.array(x2)
        x3 = np.array(x3)

        dist1 = np.linalg.norm(x1-x2)
        dist2 = np.linalg.norm(x1-x3)

        if dist1 > dist2:
            label = [1]

            inst_one = tuple((x1, label))
            #print(inst_one)
            inst_two = tuple((x2, label))
            inst_three = tuple((x3, label))
            #final.append((triple[i], label))
            final.append(inst_one)
            final.append(inst_two)
            final.append(inst_three)
        elif dist1 < dist2:
            label = [-1]
            inst_one = tuple((x1, label))
            inst_two = tuple((x2, label))
            inst_three = tuple((x3, label))
            #final.append((triple[i], label))
            final.append(inst_one)
            final.append(inst_two)
            final.append(inst_three)
        else:
            continue

    print(final[0])
    # split data
    X_full = []
    y = []

    for x in final:
        #print(x)
        X, label = x
        y.append(label)
        X_full.append(X)

    return X_full, y

def process_one_data_instance(triple):
    """This function returns a cleaned
    version of the triple and a placeholder integer
    input:
    triple (list of tuples) contains the triple
    for relative comparison
    """
    print(type(triple))
    X_full = triple
    X_full = np.array(X_full)
    nx, ny =  X_full.shape

    X_new_full = X_full.reshape(1, nx*ny)
    print()

    return X_new_full, 0

def process_one_datapoint_instance(datapoint):
    """This function returns a cleaned
    version of the triple and a placeholder integer
    input:
    triple (list of tuples) contains the triple
    for relative comparison
    """
    print(type(datapoint))
    X_full = list(datapoint)
    print(X_full)
    X_full = np.array(X_full)
    print(X_full.shape)
    nx =  X_full.shape
    print(len(nx))


    X_new_full = X_full.reshape((1, 14))
    print()

    return X_new_full


def process_data(df, option=1):
    """This function returns a cleaned version
    of the data broken into X and y pairs. The cleaning
    includes creating triples as well as computing
    the Euclidean distance for the Support Vector Machine
    input: df (pandas DataFrame) original data source
    """
    if option is 1:
        triple = prep_triple(df)
        X_full, y = euclidean_distance_triple(triple)

        triple_two = prep_triple(df)
        X_full_test, y_pred_test = euclidean_distance_triple(triple_two)

        X_full = np.array(X_full)
        nsamples, nx, ny =  X_full.shape

        X_new_full = X_full.reshape(nsamples, nx*ny)
        y_full = np.array(y)

        X_full_test = np.array(X_full_test)
        nsamples, nx, ny =  X_full_test.shape
        X_new_full_test = X_full_test.reshape(nsamples, nx*ny)
        y_pred_test = np.array(y_pred_test)

        return X_new_full, y_full, X_new_full_test, y_pred_test

    elif option is 2:
        triple = prep_triple(df)
        X_full, y = euclidean_distance_on_instance(triple)

        triple_two = prep_triple(df)
        X_full_test, y_pred_test = euclidean_distance_on_instance(triple_two)

        X_full = np.array(X_full)
        nx, ny =  X_full.shape

        X_new_full = X_full
        y_full = np.array(y)

        X_full_test = np.array(X_full_test)
        nx, ny =  X_full_test.shape
        X_new_full_test = X_full_test
        y_pred_test = np.array(y_pred_test)

        return X_new_full, y_full, X_new_full_test, y_pred_test


def relative_comparison_distribution(X_test, y_test, X, y, n_labels, filename):
    """This is the semi-supervised relative comparison
    approach discussed in the paper Fern et al.
    inputs:
    X (numpy array) features of the variables
    y (numpy array) labels from Euclidean distance
    n_lables (int) the number of classes to be inputed
    for KMeans
    """

    models = []

    #train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20)
    print(X_train)
    #print(X_train.shape)
    print("Training")
    svm_class  = SVC().fit(X_train, y_train)
    models.append(svm_class)
    #print(X_test)
    pred = svm_class.predict(X_test)
    print(pred)
    score = str(metrics.accuracy_score(y_test, pred))
    print(score)

    new_x_df = pd.DataFrame(X_test)
    new_y_df = pd.DataFrame(pred)


    #SVM
    svm_pred_df = pd.concat([new_x_df, new_y_df], axis=1)
    X_train_cluster = svm_pred_df


    #KMeans
    kmeans = KMeans(n_clusters=n_labels).fit(X_train_cluster)
    models.append(kmeans)
    pred_kmeans = kmeans.predict(X_train_cluster)
    print(pred_kmeans)
    print(kmeans.cluster_centers_)

    new_kmeans_y_df = pd.DataFrame(pred_kmeans)
    print(new_kmeans_y_df)
    kmean_pred_df = pd.concat([new_x_df, new_kmeans_y_df], axis=1)


    size = len(kmean_pred_df.columns) -1

    X_df = kmean_pred_df.iloc[:, 0:size]
    y_df = kmean_pred_df.iloc[:, size]

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=.20)

    #Random Forest
    random_forest = RandomForestClassifier(n_estimators=50).fit(X_train, y_train)
    pred = random_forest.predict(X_test)
    models.append(random_forest)
    print(pred)
    score = str(metrics.accuracy_score(y_test, pred))
    print(score)
    print("Done Training")

    with open("../Models/"+filename+".pkl", "wb") as f:
        for model in models:
            pickle.dump(model, f)

def load_rel(filename):
    """This function loads the relative_comparison_distribution
    models and appends it to a list for future use cases
    """
    models = []
    with open("../Models/"+filename+".pkl", "rb") as f:
        while True:
            try:
                models.append(pickle.load(f))
            except:
                break

    return models

def clean_data(data):
    """This function returns a cleaned version of
    the data
    input:
    data (pandas DataFrame) original dataframe from csv
    file
    """
    for col in data:
        data[col] = data[col].replace('?', np.NaN)

    data.dropna(inplace=True)
    data = data.astype("float64")
    return data


def pred_on_rel(X, option, filename):
    """ This function returns the probabilities
    associated with each class and the predicted
    class label
    input:
    X (numpy array) triple of data points
    option (string) decision for type of the data
    and prediction
    """
    if option == "l":

        X, _, _, _ = process_data(X)
        models = load_rel()

        first = models[0].predict(X)
        pred = pd.DataFrame(first)
        #print(pred)
        X = pd.DataFrame(X)
        joined_df = pd.concat([X,pred], axis=1)

        sec = models[1].predict(joined_df)
        pred_two = pd.DataFrame(sec)
        joined_df_two = pd.concat([X, pred_two], axis=1)

        size = len(joined_df_two.columns) -1
        X_df = joined_df_two.iloc[:, 0:size]

        third = models[2].predict(X_df)
        pred_class_labels = third
        probabilities = models[2].predict_proba(X_df)
        return pred_class_labels, probabilities

    elif option == "a":
        models = load_rel(filename)

        first = models[0].predict(X)
        pred = pd.DataFrame(first)
        #print(pred)
        X = pd.DataFrame(X)
        joined_df = pd.concat([X,pred], axis=1)

        sec = models[1].predict(joined_df)
        pred_two = pd.DataFrame(sec)
        joined_df_two = pd.concat([X, pred_two], axis=1)

        size = len(joined_df_two.columns) -1
        X_df = joined_df_two.iloc[:, 0:size]

        third = models[2].predict(X_df)
        pred_class_labels = third
        probabilities = models[2].predict_proba(X_df)
        return pred_class_labels, probabilities


def dont_know_case(x1, x2, x3):
    label_one, probs_one = x1
    label_two, probs_two = x2
    label_three, probs_three = x3

    probs_one = probs_one[0]
    probs_two = probs_two[0]
    probs_three = probs_three[0]

    yes = 0.0
    no  = 0.0

    if label_one == label_two:
        yes += (probs_one[i] * probs_two[i]) * (1-probs_three[i])

    if label_one == label_three:
        no += (probs_one[i] * probs_three[i]) * (1-probs_two[i])

    #print(yes)
    #print(no)
    return 1 - yes - no

def probability_estimates(x1, x2, x3):
    label_one, probs_one = x1
    label_two, probs_two = x2
    label_three, probs_three = x3

    probs_one = probs_one[0]
    probs_two = probs_two[0]
    probs_three = probs_three[0]
    #print(label_one, label_two, label_three)
    #print(len(probs_one[0]))
    yes = 0.0
    no = 0.0
    dk = 0.0
    for i in range(len(probs_one)):
        yes += (probs_one[i] * probs_two[i]) * (1-probs_three[i])

    for i in range(len(probs_one)):
        no += (probs_one[i] * probs_three[i]) * (1-probs_two[i])

    dk = 1 - yes - no
    return yes, no ,dk


def probability_estimates_given_triple(x1, x2, x3):
    label_one, probs_one = x1
    label_two, probs_two = x2
    label_three, probs_three = x3

    probs_one = probs_one[0]
    probs_two = probs_two[0]
    probs_three = probs_three[0]
    #print(label_one, label_two, label_three)
    #print(len(probs_one[0]))
    oracle_prob = 0.0
    if label_one == label_two and label_one != label_three:
        for i in range(len(probs_one)):
            oracle_prob += (probs_one[i] * probs_two[i]) * (1-probs_three[i])

    elif label_one != label_two and label_one == label_three:
        for i in range(len(probs_one)):
            oracle_prob += (probs_one[i] * probs_three[i]) * (1-probs_two[i])


    else:
        oracle_prob = 0.0

    return oracle_prob


def equation_two_entropy(triplet, option, filename):
    total = 0
    final_total = 0
    for h in triplet:
        h = np.array(h)
        h = h.reshape(1,-1)
        pred_class_labels, probs = pred_on_rel(h, option, filename)
        probs = probs[0]
        for i in range(len(probs)):
            if probs[i] >0:
                total+=probs[i]*math.log2(probs[i])
        final_total+= total

    return -1*final_total



def oracle_answer(Rp, n_labels, option, filename):
    p_yes = []
    p_no = []
    p_dk = []

    for triplet in Rp:
        '''triple, _  = process_one_data_instance(triplet)
        pred_class_labels, probs = pred_on_rel(triple, option, filename)
        print(probs)
        print(pred_class_labels)
        print()'''
        x1, x2, x3 = triplet

        x1 = np.array(x1)
        x1 = x1.reshape(1,-1)

        x2 = np.array(x2)
        x2 = x2.reshape(1,-1)

        x3 = np.array(x3)
        x3 = x3.reshape(1,-1)


        pred_class_labels, probs = pred_on_rel(x1, option, filename)
        pred_class_labels_two, probs_two = pred_on_rel(x2, option, filename)
        pred_class_labels_three, probs_three = pred_on_rel(x3, option, filename)

        #print(x1)
        print(pred_class_labels)
        print(probs)

        #print(x2)
        print(pred_class_labels_two)
        print(probs_two)

        #print(x3)
        print(pred_class_labels_three)
        print(probs_two)

        x1_info = (pred_class_labels, probs)
        x2_info = (pred_class_labels_two, probs_two)
        x3_info = (pred_class_labels_three, probs_three)

        oracle_prob = probability_estimates_given_triple(x1_info, x2_info, x3_info)
        yes ,no , dk = probability_estimates(x1_info, x2_info, x3_info)

        info = equation_two_entropy(triplet, option, filename)
        print(info)


        print()

        #triple_test = process_one_datapoint_instance(x1)
        #pred_class_labels, probs = pred_on_rel(triple_test, option)


    #for c in range(n_labels):


def main():

    decision = input("Would you like to train the model, load, or use algorithm (t for train or l for load a for algorithm)?:  ")

    if decision == "t":

        labels = int(input("The number of classes within the program: "))
        datafile = str(input("Please input the datafile: ")).strip(" ")
        filename = str(input("Please output filename for model: ")).strip(" ")
        base_folder = "../Data/"
        path = base_folder + datafile
        data = pd.read_csv(path)
        data = clean_data(data)

        pred_df = data[50:100]
        df = data

        X_new_full, y, X_new_full_test, y_pred_test = process_data(df, 2)
        relative_comparison_distribution(X_new_full_test, y_pred_test, X_new_full, y, labels, filename)

    elif decision == "l":
        datafile = str(input("Please input the datafile: ")).strip(" ")
        filename = str(input("Please output filename for model: ")).strip(" ")
        base_folder = "../Data/"
        path = base_folder + datafile
        data = pd.read_csv(path)
        data = clean_data(data)

        df  = data[200:250]
        pred_class_labels, probs = pred_on_rel(df, decision, filename)
        print(pred_class_labels)
        print(probs)

    elif decision == "a":

        datafile = str(input("Please input the datafile: ")).strip(" ")
        filename = str(input("Please output filename for model: ")).strip(" ")

        base_folder = "../Data/"
        path = base_folder + datafile
        data = pd.read_csv(path)
        data = clean_data(data)
        #data = data.iloc[1:5]

        Ru = prep_triple(data)

        # Initialize Rl and Ru
        Rl = []
        Rp = Ru[:10]

        # Use Section 3.5. This relies on the relative_comparison_distribution
        # function to do the distance metric learning for estimating the
        # probability of a class and class label given a set of triples
        # We will use the oracle_answer() function to house the main
        # active learning algorithm

        oracle_answer(Rp, 5, decision, filename)




if __name__ == "__main__":
    main()
