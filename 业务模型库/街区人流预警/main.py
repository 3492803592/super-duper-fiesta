import torch
from torch import nn
import pandas as pd

from  processing import process_all_data 
from features_tensor_building import build_stgcn_tensors
from road_network import SpatialGraphBuilder
from model import STGCN
from train_model import train_model
from evelate import evelate_test



if __name__=="__main__":
    data_path='./data(real)/data.xlsx'
    osm_path="./data(real)/map_road.osm"
    
    df= pd.read_excel(data_path, sheet_name="count")
    area_df = pd.read_excel(data_path, sheet_name="detector")
    weather_df=pd.read_csv('./data(real)/weather.csv')
    camera_df=pd.read_excel(data_path, sheet_name="detector")
    
    print("开始处理原始数据...")
    merge_df_1m,merge_df_10m=process_all_data (df,area_df,weather_df)
    
    print("开始构建STGCN特征张量...")
    train_dataset,val_dataset,test_dataset,num_F=build_stgcn_tensors(merge_df_10m,density_prefix= "_density",time_col="Time",T_in= 24,T_out = 1,train_ratio= 0.7,val_ratio= 0.15)

    print("开始构建多重邻接矩阵...")
    graphbuild=SpatialGraphBuilder(osm_path)
    A_topo,A_dist=graphbuild.build_multi_adjacency(k=2,L=1000,sigma=None,camera_df=area_df,poi_df=None)
    A_sets=[A_dist,A_topo]
    print("初始化模型...")
    num_nodes=area_df.shape[0]
    num_adj=len(A_sets)
    model=STGCN(num_nodes=num_nodes,in_channels=num_F,hidden_channels=128,num_adj=num_adj,T_in=24,num_blocks=3,dropout=0.2)

    print("开始训练STGCN模型...")
    model,train_losses, val_losses= train_model(model,train_dataset,val_dataset,A_sets,epochs=100,batch_size=64,lr=3e-4,weight_decay=0.01,
    patience=10,
    device="cuda",
    save_dir="./checkpoints",
    plot_loss=True)

   
    print("训练完成，开始评估模型...")
    
    evelate_test(model,dataset=test_dataset, A_sets=A_sets,batch_size=32,device="cuda",save_path='./data_out',node_id=3)

    

    
    