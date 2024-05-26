import numpy as np

#1. Save and Writing Files using numpy

array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

np.savetxt('data.txt', array, delimiter=',', fmt='%d')

loaded_array = np.loadtxt('data.txt', delimiter=',', dtype=int)
print(loaded_array)

np.save('data.npy', array)

loaded_array = np.load('data.npy')
print(loaded_array)

array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

np.savez('data.npz', array1=array1, array2=array2)

loaded_data = np.load('data.npz')

array1_loaded = loaded_data['array1']
array2_loaded = loaded_data['array2']

print(array1_loaded)
print(array2_loaded)

#2. Load data from Files using numpy

#Example 1: Loading Data from a Text File with Missing Values

import numpy as np

# Load the array from a text file with missing values

[[ 1  2  3]
 [ 4 -1  6]
 [ 7  8  9]]

loaded_array = np.genfromtxt('data_with_missing.txt', delimiter=',', dtype=int)
print(loaded_array)

#Example 2: Loading Multiple Arrays from a '.npz' File

import numpy as np

# Save example arrays (run this only once to create the file)
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
np.savez('data.npz', array1=array1, array2=array2)

# Load multiple arrays from the .npz file
loaded_data = np.load('data.npz')

# Access the arrays using the keys
array1_loaded = loaded_data['array1']
array2_loaded = loaded_data['array2']

print('Array 1:', array1_loaded)
print('Array 2:', array2_loaded)

#Example 3: Loading Data from a CSV File Using Pandas and Converting to NumPy Array

import pandas as pd

# Load the CSV file using pandas
df = pd.read_csv('data.csv')

# Convert the DataFrame to a NumPy array
array = df.to_numpy()

print(array)







