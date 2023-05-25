
# Part 1: Association Rule Mining
# Given the eye tracking data where the fixations of users are recorded, based on a given stimula that is a webpage, we are 
# required to find the most frequent itemsets. The most frequent itemsets correspond to the set of fixations that occur
# most frequently together. E.g. set of the fixation (A,L,M) could be a frequent fixation. 


from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import json
import pandas as pd
from sklearn.cluster import DBSCAN
import os
import matplotlib.pyplot as plt
import numpy as np



data_file = "CNG514-Assignment-3-data/Babylon_sequence_dataset.txt"

# users_fixations is a list of list, where each list belongs to a user and stores their fixations.
# the time information for each fixation has been removed since it won't be used for part 1.
users_fixations = []
with open(data_file, 'r') as file:
    # Process each user
    for user in file:
        user_fixation = []
        user_fixations = eval(user)
        # process each fixation for a user
        for fixation in user_fixations:
            # only take the id of the fixation and leave behind time info
            user_fixation.append(fixation[0])
        users_fixations.append(user_fixation)

print((users_fixations)[0])


# Initialize TransactionEncoder
te = TransactionEncoder()

# te.fit_transform analyzes the input transaction data to determine the unique items present in the entire dataset.
# Fit and transform the itemsets to a one-hot encoded NumPy array
itemsets_encoded = te.fit_transform(users_fixations)

# Convert the encoded itemsets array to a DataFrame
df = pd.DataFrame(itemsets_encoded, columns=te.columns_)


# Apply the apriori algorithm to find frequent itemsets.
# The Apriori algorithm is used for association rule mining and determines frequent itemsets based on a minimum support 
# threshold. Confidence, on the other hand, is used to assess the strength of association rules derived from the frequent 
# itemsets.
frequent_itemsets = apriori(df, min_support=0.75, use_colnames=True)

# Sort the itemsets by support in descending order
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

# we got 1407 frequent itemsets based on min_support of 0.75
print(frequent_itemsets)


# analyzing the results
# first lets look into the itemsets with support of 1; those that appear in every user's set of fixations.
# we can play with the value of the support_threshold as we like.
support_threshold = 1
frequent_itemsets_support_1 = frequent_itemsets[frequent_itemsets['support'] == support_threshold]
print(frequent_itemsets_support_1)
print("There are ", str(len(frequent_itemsets_support_1)), "set of fixations whose support is 1")
print("Meaning that they appear in every user's set of fixation!")



# below, we can further filter the frequent itemsets based on the length of the itemsets (the number of
# fixations it has). To sum up, the first filter we applied was based on the support_threshold and the second
# filter was based on the number of fixation in the itemset.
# we can play with the value of n_itemset as we wish.
n_itemset = 1
frequent_1_itemsets_support_1 = frequent_itemsets_support_1[frequent_itemsets_support_1['itemsets'].apply(lambda x: len(x) == n_itemset)]
print(frequent_1_itemsets_support_1)
# as it can be seen only M,P,Q,R,S itemsets have support of 1 and have only one fixation (1-itemset)


# Finding the maximum_frequent_itemset from the frequent itemsets of support 1
maximum_frequent_itemset = frequent_itemsets_support_1.loc[frequent_itemsets_support_1['itemsets'].apply(len).idxmax(), 'itemsets']
print(maximum_frequent_itemset)


# Finding the maximum_frequent_itemset from the frequent itemsets (those which survived Apriori with min_sup) 
maximum_frequent_itemset = frequent_itemsets.loc[frequent_itemsets['itemsets'].apply(len).idxmax(), 'itemsets']
print(maximum_frequent_itemset)


# testing what happens when min_support is reduced to 0.5. we expect an increase in the number
# of frequent itemsets because it becomes easier to be considered as frequent when the minimum 
# support is 0.5 compared to 0.75
frequent_itemsets_min_sup_05 = apriori(df, min_support=0.50, use_colnames=True)

# Sort the itemsets by support in descending order
frequent_itemsets_min_sup_05 = frequent_itemsets_min_sup_05.sort_values(by='support', ascending=False)

# we got 1407 frequent itemsets based on min_support of 0.75. and we got 4159 for min sup of 0.5
print(frequent_itemsets_min_sup_05)


# Part 2: Clustering

# DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm 
# commonly used in machine learning and data mining. It is designed to discover clusters of data points based 
# on their density distribution in the feature space.

# It has 2 hyper-parameters: Eps (Maximum radius of the neighbourhood) and 
# Minpts (Minimum number of points in an Eps-neighbourhood of that point)


# Data preparation

folder_path = './CNG514-Assignment-3-data/Babylon_fixation_dataset'

# List all files in the folder
files = os.listdir(folder_path)

users = []

# Iterate over each file
for file_name in files:
    # Construct the absolute file path
    file_path = os.path.join(folder_path, file_name)

    # read the file
    with open(file_path, 'r') as file:
        
        # each line represents a single fixation
        fixations = file.readlines()
        
        # the user list holds all the fixations belonging to a user
        user = []
        # skip the first line that holds attribute names
        for fixation in fixations[1:]:
            # Split the fixation information by the tab ('\t') delimiter
            # since the webpage is same for everyone, we can discard that attribute.
            fixation = fixation.strip().split('\t')[:-1]
            user.append(fixation)
    
    # for some users the fixation data is lost, hence, we should skip such users.
    # this can be considered as a step of data preparation.
    if user != []:
        users.append(user)


# note: we are given 38 files that hold user fixation data. however, some have been
# observed to be empty, specifically 2. Hence, there are 36 users with available data
print("There are", str(len(users)), "users.")
print("Not all users have the same number of fixations!")
print(users[0][0])


# We agreed to ignore the stimula name since all the users were looking at the same webpage.
# Now, lets do further feature engineering. 
# First, lets experiment with reducing features and only taking location 
# (x and y coordinates) info into account so that at least we can see it on a 2d plot because 
# if the number of features is more than 2, then its not possible to see the clusters in a 2d plot.
fixations_combined = []
for user in users:
    for fixation in user:
        # only get the coordinate (x,y) values
        filtered_features = [int (fixation[3]), int (fixation[4])]
        fixations_combined.append(filtered_features)
    
print(fixations_combined[0])



# Create a DBSCAN instance
dbscan = DBSCAN(eps=20, min_samples=10)

# Perform DBSCAN clustering
labels = dbscan.fit_predict(fixations_combined)


# Getting only the unique cluster labels
unique_labels = np.unique(labels)

# Printing the unique cluster labels by discarding the outlier label with (-1)
print("The unique clusers are", unique_labels[1:])

# getting the outliers. they are the one with label of -1
outliers = [value for value in labels if value != -1]
print("\nThe number of fixations that are considered as outliers based on hyperparamters of min_samples=10 and eps=20 is", len(outliers))

print("There are in total", len(labels), "fixation points.")
percentage_of_ouliers = round((len(outliers)/len(labels))*100, 1)
print(str(percentage_of_ouliers),"% of them have been labeled as outlies")


# Plot the results
plt.scatter([fixations[0] for fixations in fixations_combined], [fixations[1] for fixations in fixations_combined], c=labels, cmap='viridis')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('DBSCAN Clustering')
# Invert the y-axis
plt.gca().invert_yaxis()
# Invert the x-axis
plt.gca().invert_xaxis()
plt.show()



def grid_search_dbscan(fixations_combined, min_samples_values, eps_values, target_clusters=22):
    # Initialize variables to store the best parameters and metrics
    best_min_samples = None
    best_eps = None
    labels_found = None
    
    # these variables are aimed to be minimized.
    # we want to find as many clusters as it is asked for. note that sometimes we do not know the number of 
    # clusers. in that case the code can be modified to take that into account. 
    best_num_clusters_diff = float('inf')

    # Iterate over each combination of min_samples and eps
    for min_samples in min_samples_values:
        for eps in eps_values:
            # Create a DBSCAN instance with the current parameters
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)

            # Perform DBSCAN clustering
            labels = dbscan.fit_predict(fixations_combined)
            unique_labels, counts = np.unique(labels, return_counts=True)

            # Exclude outliers (label -1) from counts
            counts = counts[unique_labels != -1]

            # Get the number of clusters and outliers
            num_clusters = len(counts)
            num_outliers = counts[0] if len(counts) > 0 else 0

            # Calculate the difference between the actual and target number of clusters
            num_clusters_diff = abs(num_clusters - target_clusters)

            # Check if the current parameters result in the smallest num_clusters_diff
            if num_clusters_diff < best_num_clusters_diff:
                best_min_samples = min_samples
                best_eps = eps
                best_num_clusters_diff = num_clusters_diff
                labels_found = labels

    # Return the best parameters found
    return best_min_samples, best_eps, labels_found



def plot_dbscan_results(fixations_combined, labels):
    # Plot the results
    plt.scatter([fixations[0] for fixations in fixations_combined],
                [fixations[1] for fixations in fixations_combined],
                c=labels,
                cmap='viridis')
    
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('DBSCAN Clustering')
    
    # Invert the y-axis
    plt.gca().invert_yaxis()
    
    # Invert the x-axis
    plt.gca().invert_xaxis()
    
    plt.show()


min_samples_values = [5, 10, 15, 20, 25]
eps_values = [10, 20, 30, 40]

best_min_samples, best_eps, labels = grid_search_dbscan(fixations_combined, min_samples_values, eps_values, target_clusters=22)

print("Best min_samples:", best_min_samples)
print("Best eps:", best_eps)

# Plot the results
plot_dbscan_results(fixations_combined, labels)


best_min_samples, best_eps, labels = grid_search_dbscan(fixations_combined, min_samples_values, eps_values, target_clusters=5)

print("Best min_samples:", best_min_samples)
print("Best eps:", best_eps)

# Plot the results
plot_dbscan_results(fixations_combined, labels)




