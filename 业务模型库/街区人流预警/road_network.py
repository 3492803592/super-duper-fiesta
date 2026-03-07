import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

class SpatialGraphBuilder:
    def __init__(self, osm_path):
      
        self.G = ox.graph_from_xml(osm_path, simplify=True)
        self.G = ox.utils_graph.convert.to_undirected(self.G)#转换成无向图

        self.camera_ids = None
        self.camera_to_node = None
        self.poi_ids = None
        self.poi_to_node = None

    def map_cameras(self, camera_df):
        
        camera_nodes = ox.distance.nearest_nodes(
                self.G,
            X=camera_df["LON"].values,
            Y=camera_df["LAT"].values)
            

        self.camera_ids = camera_df["device_id"].tolist()
        self.camera_to_node = dict(zip(self.camera_ids, camera_nodes))

    def map_poi(self,poi_df):

        poi_nodes=ox.distance.nearest_nodes(
                self.G,
                X=poi_df["LON"].values,
                Y=poi_df["LAT"].values
        )
        self.poi_ids=poi_df["POI_ID"].tolist()
        self.poi_to_node=dict(zip(self.poi_ids,poi_nodes))

    def build_topology_adjacency(self, k=2):

        N = len(self.camera_ids)
        A = np.zeros((N, N))

        for i, cam_i in enumerate(self.camera_ids):
            node_i = self.camera_to_node[cam_i]

            lengths = nx.single_source_shortest_path_length(
                self.G,
                node_i,
                cutoff=k
            )

            for j, cam_j in enumerate(self.camera_ids):
                if self.camera_to_node[cam_j] in lengths:
                    A[i, j] = 1

        return self._normalize(A)
    
    def build_distance_adjacency(self, L=1000, sigma=None):
        
        if sigma is None:
            sigma = L / 3

        N = len(self.camera_ids)
        D = np.full((N, N), np.inf)

        for i, cam_i in enumerate(self.camera_ids):
            node_i = self.camera_to_node[cam_i]

            lengths = nx.single_source_dijkstra_path_length(
                self.G,
                node_i,
                weight="length"
            )

            for j, cam_j in enumerate(self.camera_ids):
                node_j = self.camera_to_node[cam_j]
                if node_j in lengths:
                    D[i, j] = lengths[node_j]

        A = np.exp(-D / sigma)
        A[D > L] = 0

        return self._normalize(A)

    """ def build_poi_similarity_matrix(
    cam_coords,       
    food_pois,         
    scenic_pois,      
    sigma_food=300.0,  
    sigma_scenic=500.0,
    eps=1e-8):
        def poi_influence(cam, poi, sigma):
            if poi is None or len(poi) == 0:
                return np.zeros(cam.shape[0])
            dist = cdist(cam, poi)  
            return np.exp(-(dist ** 2) / (2 * sigma ** 2)).sum(axis=1)

    # 构建功能向量（餐饮 + 景点）
        food_inf = poi_influence(cam_coords, food_pois, sigma_food)
        scenic_inf = poi_influence(cam_coords, scenic_pois, sigma_scenic)

        V = np.stack([food_inf, scenic_inf], axis=1)  

    # 2余弦相似性
        norm = np.linalg.norm(V, axis=1, keepdims=True) + eps
        V_norm = V / norm
        A_poi = V_norm @ V_norm.T  

        return A_poi """
    
    def build_multi_adjacency(self, k=2, L=1000, sigma=None,camera_df=None,poi_df=None):
        A_sets=[]
        self.map_cameras(camera_df)
        #self.map_poi(poi_df)
        A_topo = self.build_topology_adjacency(k=k)
        A_dist = self.build_distance_adjacency(L=L, sigma=sigma)
        #A_sets.append(A_topo)
        #A_sets.append(A_dist)

        return A_dist,A_topo

    def _normalize(self, A):
        A = A + np.eye(A.shape[0])
        D = np.diag(A.sum(axis=1))
        D_inv_sqrt = np.linalg.inv(np.sqrt(D))
        return D_inv_sqrt @ A @ D_inv_sqrt
    
#多矩阵的融合方法：1、静态加权融合  2、多通道融和   3、图注意力网络融合  4、多图卷积   5、可学习邻接融合
    
if __name__=="__main__":
    osm_path="./data(real)/map_road.osm"
    camera_df=pd.read_excel("./data(real)/data.xlsx",sheet_name="detector")
    #poi_df=pd.read_excel("./data(real)/poi.xls")
    #cam_coords=camera_df[["LON","LAT"]].values
    #food_pois=poi_df[poi_df["TYPE"]=="美食"][["LON","LAT"]].values
    #scenic_pois=poi_df[poi_df["TYPE"]=="景点"][["LON","LAT"]].values
    graph_builder=SpatialGraphBuilder(osm_path)
    #graph_builder.map_poi(poi_df)
    A=graph_builder.build_multi_adjacency(k=2,L=1000,sigma=None,camera_df=camera_df,poi_df=None)
    #A_poi=graph_builder.build_poi_similarity_matrix(cam_coords,food_pois,scenic_pois,sigma_food=300.0,sigma_scenic=500.0)
    #np.save("./data_out/adjacency_topology.npy",A)
    #np.save("./data_out/adjacency_distance.npy",D)
    #np.save("./data_out/adjacency_poi.npy",A_poi)



