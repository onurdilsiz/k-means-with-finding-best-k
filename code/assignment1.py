import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
import math
from sklearn.datasets import make_blobs


def plotclusters(X, labels, centers, k):

    colors = ['C{}'.format(i) for i in range(k)]  # [C(1), C(2), C(3)] clusters
    for i in range(k):
        plt.scatter(X[labels == i, 0], X[labels == i, 1], color=colors[i],
                    label='Cluster {}'.format(i+1))  # for cluster 1,
    plt.scatter(centers[:, 0], centers[:, 1], color='black',
                marker='x', s=200, linewidths=k, label='Cluster centers')
    plt.legend()
    plt.show()


#assignment of labels
def assignment(df, centres):
    labels = np.zeros(df.shape[0],dtype=int) ## labels vector with the same dimension as dataset
    length= df.shape[0] ## amount of datapoints
    for i in range(0,length): ##Assignment for all datapoints
        mindistance= 10000000
        point= df[i]
#         print(point[0], point[1])
        for k in range(len(centres)): ## calculate function= shortest distance
            center= centres[k]
            distance = (center[0]-point[0])**2 + (center[1]- point[1])**2
            distance = math.sqrt(distance)
            if(distance< mindistance):
                mindistance = distance
                labels[i] =  k
    return labels


## Part 3. Function: replace the centres
def replace(df, centres):
    labels = assignment(df, centres)
    k = len(centres)
    clusterCenterSums = np.zeros((k, 2))
    length = df.shape[0]

    clusterCount = np.zeros(k)
    for k in labels:
        clusterCount[int(k)] += 1

    for index in range(length):
        point = df[index]
        k = int(labels[index])
        clusterCenterSums[k][0] += point[0]
        clusterCenterSums[k][1] += point[1]

    for index in range(len(clusterCenterSums)):
        centreX = clusterCenterSums[index][0] / clusterCount[index]
        centreY = clusterCenterSums[index][1] / clusterCount[index]
        centres[index] = [centreX, centreY]
    return centres


def calculate_objective_function(X, labels, centres):
    obj = 0
    for i in range(0, len(X)):
        #         print(labels[i])
        distance = (centres[labels[i]][0] - X[i][0]) ** 2 + (centres[labels[i]][1] - X[i][1]) ** 2

        obj = obj + distance
    return obj



def compareCentres(oldCenters, newCenters):
    for i in range(len(oldCenters)):
        if ((oldCenters[i][0] - newCenters[i][0]) ** 2 + (oldCenters[i][1] - newCenters[i][1]) ** 2 > 0.0001):
            return False
    return True

def measureDistance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)



def findNearestCluster(point,centres,clusterNo):
    minDistance = 1000000000
    cluster = -1
    for i in range(0,len(centres)):
        if(i !=clusterNo):
            distance = measureDistance(point,centres[i])
            if(distance< minDistance ):
                minDistance = distance
                cluster = i
    return cluster


def silhouetteCoefficient(X, centres, labels):
        k = len(centres)
        aList = np.zeros(k)
        bList = np.zeros(k)
        clusterCount = np.zeros(k)
        for j in labels:
            clusterCount[int(j)] += 1
        for i in range(len(X)):
            sumPoint = 0
            sumPointtob = 0
            nearestCluster = findNearestCluster(X[i], centres, labels[i])
            for k in range(len(X)):
                if (labels[i] == labels[k]):
                    dist = measureDistance(X[i], X[k])
                    sumPoint += dist
                if (labels[k] == nearestCluster):
                    disttob = measureDistance(X[i], X[k])
                    sumPointtob += disttob
            meanDist = sumPoint / (clusterCount[labels[i]] - 1)
            aList[labels[i]] += meanDist / clusterCount[labels[i]]
            meanDisttob = sumPointtob / (clusterCount[nearestCluster])
            bList[labels[i]] += meanDisttob / clusterCount[labels[i]]
        a = np.mean(aList)
        b = np.mean(bList)
        s = (b - a) / max(a, b)
        return s


## Part 4. Script.


def kmeans(k, X):
    count = 0
    objective_function = []
    n = len(X)
    df = X
    labels = np.random.randint(0, k, size=n)  # Randomly assign each data point to one of k clusters
    centres = np.zeros((k, 2))  ## empty array for centers coordinates


    for i in range(k):  ##randomly select k points as centers
        random_index = random.randint(0, len(df) - 1)  # Generate a random index within the range of data points
        random_point = df[random_index]
        centres[i] = random_point


    plotclusters(df, labels, centres, k)
    flag = True

    while (flag):
        oldCentres = centres.copy()
        #         print(np.array_equal(df,X))

        labels = assignment(df, centres)
        centres = replace(df, centres)
        objective_function.append(calculate_objective_function(X, labels, centres))
        #         print("old:\n",oldCentres)
        #         print("new:\n",centres)
        count += 1
        if (count < 4):
            print("Iteration ", count, ":")
            plotclusters(df, labels, centres, k)

        if (compareCentres(oldCentres, centres)):
            flag = False

    plotclusters(df, labels, centres, k)

    plt.plot(range(1, len(objective_function) + 1), objective_function)
    s = silhouetteCoefficient(X, centres, labels)
    print("silhouette:", s)
    return s


k=5
X, y = make_blobs(n_samples=1000, centers=k, n_features=2)
kmeans(k, X)

kmeans2 = KMeans(n_clusters=k)
kmeans2.fit(X)
# Plot the clusters and centers
plotclusters(X, kmeans2.labels_, kmeans2.cluster_centers_,k)
print("sci kit:",silhouetteCoefficient(X, kmeans2.cluster_centers_,kmeans2.labels_))


def kmeansWithoutPlot(k, X):
    objective_function = []
    n = len(X)
    df = X
    labels = np.random.randint(0, k, size=n)  # Randomly assign each data point to one of three clusters
    centres = np.zeros((k, 2))  ## empty array for centers coordinates
    count = 0

    for i in range(k):  ##randomly select k points as centers
        random_index = random.randint(0, len(df) - 1)  # Generate a random index within the range of data points
        random_point = df[random_index]
        centres[i] = random_point

    flag = True

    while (flag):
        oldCentres = centres.copy()
        labels = assignment(df, centres)
        centres = replace(df, centres)
        objective_function.append(calculate_objective_function(X, labels, centres))

        count += 1


        if (compareCentres(oldCentres, centres)):
            flag = False


    #     print(objective_function)
    plt.plot(range(1, len(objective_function) + 1), objective_function)
    plt.show()
    s = silhouetteCoefficient(X, centres, labels)
    # print("silhouette:", s)
    return s




def findBestK(X):
    maxSilhouette = -1
    bestk = -1.0
    for i in range(3, 10):

        silhouette = kmeansWithoutPlot(i, X)
        if (maxSilhouette < silhouette):
            maxSilhouette = silhouette
            # print(maxSilhouette, i)
            bestk = i
    return bestk


X, y = make_blobs(n_samples=1000, centers=8, n_features=2)



print("the best:", findBestK(X))


# Generate random data points in a circular pattern
def generate_circular_data(radius, num_points):
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(theta) + np.random.normal(0, 0.1, num_points)
    y = radius * np.sin(theta) + np.random.normal(0, 0.1, num_points)
    return np.column_stack((x, y))

# Generate random data points in a uniform distribution
def generate_uniform_data(num_points):
    x = np.random.uniform(-1, 1, num_points)
    y = np.random.uniform(-1, 1, num_points)
    return np.column_stack((x, y))

# Create the dataset
num_points = 200
data1 = generate_circular_data(1, num_points)
data2 = generate_uniform_data(num_points)
data = np.vstack((data1, data2))

# Plot the dataset
plt.scatter(data[:, 0], data[:, 1,], s=10)
plt.show()


kmeans(k, data)