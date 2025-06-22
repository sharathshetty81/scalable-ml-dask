from dask.distributed import Client, LocalCluster

def start_cluster():
    cluster = LocalCluster()
    client = Client(cluster)
    return client

