import sys, csv, math, random, time, resource

# Record class with arbitrary coordinates
class Record:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.count = len(coordinates)

    def __repr__(self):
        return str(self.coordinates)

# Cluster class with with records, count and centroids
class Cluster:
    def __init__(self, records):
        # Check for empty clusters
        if len(records) == 0:
            raise Exception("Error: Cluster is empty.")
        self.records = records
        self.count = records[0].count
        # Check for same dimensions of records, complain otherwise
        for x in records:
            if x.count != self.count:
                raise Exception('Error: Dimension of records are not equal.')
        # Calculate the centroid for the cluster
        self.centroid = self.calculate_centroid()
    # String representation of record
    def __repr__(self):
        return str(self.records)
    # Assign a new list of records to this cluster and return centroid difference
    def update_centroids(self, dist_func, records):
        old_centroid = self.centroid
        self.records = records
        self.centroid = self.calculate_centroid()
        return calc_distance(dist_func, old_centroid, self.centroid)
    # Calculate the sample mean record (i.e. centroid)
    def calculate_centroid(self):
        centroid_coordinates = []
        for i in range(self.count):
            # Take the average across all Records
            centroid_coordinates.append(0.0)
            for p in self.records:
                centroid_coordinates[i] = centroid_coordinates[i] + p.coordinates[i]
            try:
                centroid_coordinates[i] /= len(self.records)
            except:
                raise ZeroDivisionError("There is a division by zero.")
        return Record(centroid_coordinates)

# Euclidean distance function
# Calculate sum of (a-b)^2 for all values of a and b
# take square root of result
def euclidean_distance(a, b):
    dist = 0.0
    for i in range(a.count):
        dist += pow((a.coordinates[i] - b.coordinates[i]), 2)
    return math.sqrt(dist)

# Manhattan Distance Function
# Calculate sum of |a-b| for all values of a and b
def manhattan_distance(a, b):
    dist = 0.0
    for i in range(a.count):
        dist += abs((a.coordinates[i] - b.coordinates[i]))
    return dist

# Cosine distance function
# Calculate 1 - (sum(a*b)/sum(a^2)^0.5 x sum(b^2)^0.5)
def cosine_distance(a, b):
    numer, denom_a, denom_b = 0.0, 0.0, 0.0
    for i in range(a.count):
        numer += a.coordinates[i] * b.coordinates[i]
        denom_a += pow(a.coordinates[i], 2)
        denom_b += pow(b.coordinates[i], 2)
    return 1 - (numer / (math.sqrt(denom_a) * math.sqrt(denom_b)))

# Function that calls one of the distance functions
def calc_distance(function, a, b):
    if function == "Euclidean": return euclidean_distance(a, b)
    elif function == "Manhattan": return manhattan_distance(a, b)
    elif function == "Cosine": return cosine_distance(a, b)
    else: raise Exception("Error: Please enter correct distance function")

# Read the data from the csv file
def get_data(filename):
    try: data = csv.reader(open(filename, "r"))
    except: raise FileNotFoundError("No such file in directory.")
    return data

# Get the details for each record of the data
def get_details(record):
    return Record([float(d) for d in record])

# Random initialization - randomly select k records
# Obtain the centroids for those records
def rnd_initialization(records, k):
    rnd_samples = random.sample(records, k)
    clusters = [Cluster([s]) for s in rnd_samples]
    centroids = []
    for cluster in clusters:
        centroids.append(cluster.centroid)
    return clusters, centroids

# Farthest-first initialization - randomly select a record, find
# the farthest point that is already not a centroid and has maximum
# distance from the centroids appended so far and appends to the
# list of centroids. Process is repeated until k initial
# centoroids are discovered
def ff_initialization(records, k):
    rnd_sample = random.choice(records)
    max_dist = float('-inf')
    centroids = [Cluster([rnd_sample]).centroid]
    clusters = [Cluster([record]) for record in records]
    # Repeat k - 1 times
    for i in range(2, k + 1):
        for cluster in clusters:
            centroid = cluster.centroid
            if centroid is not None:
                for record in records:
                    dist = euclidean_distance(record, centroid)
                    if dist > max_dist:
                        max_dist = dist
                        centroid = cluster.centroid
                        centroids.append(centroid)
    return clusters, centroids

# Returns the k cluster of records from k-means clustering
# using different initialization methods, distance functions
# for a maximum number of iterations
def kmeans(records, k, dist_func, max_iter):
    # choose one of the initialization methods
    # clusters, centroids = ff_initialization(records, k)
    clusters, centroids = rnd_initialization(records, k)
    total_iteration = 0
    while True:
        if total_iteration > max_iter:
            break
        total_iteration += 1    # Add to iteration count
        # Make a list for every cluster
        lists = [[] for cluster in clusters]
        for rec in records:
            smallest_distance = calc_distance(dist_func, rec, clusters[0].centroid)
            index = 0
            for i in range(len(clusters[1:])):
                distance = calc_distance(dist_func, rec, clusters[i + 1].centroid)
                if distance < smallest_distance:
                    smallest_distance = distance
                    index = i + 1
            # Add this record to that Cluster's corresponding list
            lists[index].append(rec)
        biggest_shift = 0.0
        for i in range(len(clusters)):
            shift = clusters[i].update_centroids(dist_func, lists[i])
            biggest_shift = max(biggest_shift, shift)
        # If the biggest centroid shift is less than the cutoff or
        # interation count is greater than maximum then break out of loop
        if biggest_shift < 0.5 or total_iteration > max_iter:
            break
    print("Total Iteration =", total_iteration)
    return clusters

# Main function where all the function assignments and
# implementation is carried out
def main():
    #
    dist_funcs = ["Euclidean", "Manhattan", "Cosine"]
    ks = [5, 10, 15]
    filenames = ["forestfiresmod1.csv","abalone.data"]
    max_iter = 23
    # k = int(input("Input Number of cluster: "))
    # max_iter = int(input("Maximum iterations: "))
    # Check for args and return relevant usage instruction
    myArgs = len(sys.argv)
    if myArgs != 2:
        print("Usage -> kmeans.py dataset-name")
        exit()
    # Get the correct file name and complain otherwise
    else:
        if sys.argv[1] == "forestfires": filename = filenames[0]
        elif sys.argv[1] == "abalone": filename = filenames[1]
        else:
            print("Usage -> kmeans.py dataset-name")
            print("dataset-name: abalone OR forestfires")
            exit()

    print("File: ", filename)
    start_time = time.time() # Get start time
    start_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    reader = get_data(filename)
    records = [] # Initiate points list
    record_counter = 0
    for k in ks:
        print("Number of Clusters:", k)
        for dist_func in dist_funcs:
            print("-------- Distance function: ", dist_func, "-------------")
            for rec in reader:
                records.append(get_details(rec[1:]))
                record_counter += 1
            clusters = kmeans(records, k, dist_func, max_iter)
            cluster_ = []
            end_time = time.time() - start_time
            end_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            usage_difference = end_usage - start_usage
            for i, c in enumerate(clusters):
                count = 0
                for rec in c.records:
                    # print records for corresponding cluster
                    # print(" Cluster: ", i + 1, "\t Record :", r)
                    count = count + 1
                # print("Records for cluster", i + 1, ":", count)
                cluster_.append(float(((float(count) / float(record_counter)) * 100)))
            print("Time:", repr(round(end_time,8)), "Seconds")
            print("Usage:", usage_difference)
            for id, cp in enumerate(cluster_):
                print(id + 1, ":", round(cp,1), "\%")
            print("\n")

if __name__ == "__main__":
    main()