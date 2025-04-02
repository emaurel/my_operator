from tercen.client import context as ctx
import numpy as np
import json
import pandas as pd


tercenCtx = ctx.TercenContext(
    workflowId="241160841cead65995c0c45d0a011ce8",
    stepId="0ae6a9ae-e827-47f5-8a9b-78152f81112c",
    username="admin", # if using the local Tercen instance
    password="admin", # if using the local Tercen instance
    serviceUri = "http://127.0.0.1:5402/" # if using the local Tercen instance 
)

NB_COLORS = 2
MAX_ITER = 15
PRECISION = 1000

propertyValues = tercenCtx.cubeQuery.toJson()["operatorSettings"]["operatorRef"]["propertyValues"]
NB_COLORS = int(propertyValues[0]["value"])
MAX_ITER = int(propertyValues[1]["value"])
PRECISION = float(propertyValues[2]["value"])  
print(NB_COLORS, MAX_ITER, PRECISION)

data = tercenCtx.select(['.y', '.ci', '.ri'], df_lib="pandas").values
data = np.array(data, dtype=np.uint32)

R = (data[:, 0] >> 16) & 0xFF
G = (data[:, 0] >> 8) & 0xFF
B = data[:, 0] & 0xFF

X = data[:, 1]
Y = data[:, 2]

pixels = np.column_stack((R, G, B, X, Y))

print(pixels)

def k_means(nb_colors, pixels, max_iter) :
    colors = []
    for i in range(nb_colors):
        colors.append(np.random.randint(0, 255, 3))

    new_colors = colors.copy()
    clusters = [[] for i in range(nb_colors)]
    converged = False
    iteration = 0

    while not converged and iteration < max_iter:
        iteration += 1
        print("Iteration", iteration)
        colors = new_colors.copy()
        clusters = [[] for i in range(nb_colors)]
        #pixels is an array of pixels, each pixel is an array of 5 elements (R, G, B, X, Y)
        for pixel in pixels:
            min_distance = 255 * 3
            cluster = 0
            for i in range(nb_colors):
                distance = np.linalg.norm(pixel[:3] - colors[i])
                if distance < min_distance:
                    min_distance = distance
                    cluster = i
            clusters[cluster].append(pixel)
        for i in range(nb_colors):
            mean = 0
            for el in clusters[i] :
                mean += el[:3]
            if len(clusters[i]) != 0:
                mean //= len(clusters[i])
            new_colors[i] = mean

        for color, new_color in zip(colors, new_colors):
            print(np.linalg.norm(color - new_color))
            if np.linalg.norm(color - new_color) < PRECISION:
                converged = True
            else :
                converged = False
                break
    if iteration >= max_iter:
        print("Max iteration reached")

    print("Converged in", iteration, "iterations")
    for i in range(nb_colors):
        for j in range(len(clusters[i])):
            clusters[i][j][:3] = colors[i]
        if clusters[i] == []:
            clusters[i] = np.array([[0, 0, 0, 0, 0]])
            print("Empty cluster")
    #the pixels are sorted by cluster, first regroup all the clusters in one array
    clusters = np.concatenate(clusters)
    sorted_id = np.lexsort((clusters[:, 3], clusters[:, 4]))
    clusters = clusters[sorted_id]

    
    return clusters, colors



clusters, colors = k_means(NB_COLORS, pixels, MAX_ITER)
print("done clustering")

result = clusters.copy()

data = np.array(result, dtype=np.uint32)

# Extract R, G, B, X, Y
R = data[:, 0]
G = data[:, 1]
B = data[:, 2]
X = data[:, 3]
Y = data[:, 4]

# Combine R, G, B into ColorCode (0xRRGGBB)
# Shift R left 16 bits, G left 8 bits, and OR them with B
ColorCode = (R << 16) | (G << 8) | B

# Combine into new array: [[ColorCode, X, Y], ...]
df = np.column_stack((ColorCode, X, Y))
df = np.column_stack((ColorCode, df))

print(df)

dataset = pd.DataFrame(df, columns=['newColors', 'centroids', '.ci', '.ri'])

dataset = dataset.astype({'newColors' : 'float', 'centroid' : 'int32', '.ci': 'int32', '.ri': 'int32'})


print("done")

df = tercenCtx.add_namespace(dataset) 
tercenCtx.save(df)
