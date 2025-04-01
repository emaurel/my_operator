from tercen.client import context as ctx
import numpy as np
import json
import pandas as pd


tercenCtx = ctx.TercenContext(
    workflowId="241160841cead65995c0c45d0a011ce8",
    stepId="6b502ac2-406d-40d5-af29-493a6dccbeef",
    username="admin", # if using the local Tercen instance
    password="admin", # if using the local Tercen instance
    serviceUri = "http://127.0.0.1:5402/" # if using the local Tercen instance 
)

NB_COLORS = 5
MAX_ITER = 15
PRECISION = 1

print(json.dumps(tercenCtx.cubeQuery.toJson()["operatorSettings"], indent = 2))
propertyValues = tercenCtx.cubeQuery.toJson()["operatorSettings"]["operatorRef"]["propertyValues"]
NB_COLORS = int(propertyValues[0]["value"])
MAX_ITER = int(propertyValues[1]["value"])
PRECISION = float(propertyValues[2]["value"])  
print(NB_COLORS, MAX_ITER, PRECISION)

df = tercenCtx.select(['.y', '.ci', '.ri'], df_lib="pandas").values
groups = df.reshape(-1, 3, 3)  # Group every 3 rows
pixels = np.hstack([
    groups[:, :, 0],           # Take all first columns
    groups[:, 0, 1:3]          # Take columns 1 and 2 from first row of each group
])



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
                mean /= len(clusters[i])
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

# Update r, g, b values from clusters where coordinates match

n_groups = pixels.shape[0]  # Number of groups
first_cols = pixels[:, 0:3]  # The three first columns (r values)
extra_cols = pixels[:, 3:5]  # The two extra columns that were from first row

# Step 2: Reconstruct the original 3x3 groups
groups = np.zeros((n_groups, 3, 3))  # Create empty array for groups
groups[:, :, 0] = first_cols  # Put back all first columns
groups[:, 0, 1:3] = extra_cols  # Put extra columns back in first row only
groups[:, :, 1:] = np.tile(extra_cols[:, np.newaxis, :], (1, 3, 1))

# Step 3: Flatten back to original shape
df = groups.reshape(-1, 3)

dataset = pd.DataFrame(df, columns=['centroid', '.ci', '.ri'])

dataset = dataset.astype({'centroid' : 'int32', '.ci': 'int32', '.ri': 'int32'})


print("done")

df = tercenCtx.add_namespace(dataset) 
tercenCtx.save(df)
