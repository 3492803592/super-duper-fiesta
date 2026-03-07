

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def test_and_collect(model, dataset, device,A_sets,batch_size):
    model.eval()

    preds = []
    trues = []

    mae_list = []
    rmse_list = []
    mape_list = []

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )

    A_sets = [torch.tensor(A, dtype=torch.float32).to(device) for A in A_sets]

    with torch.no_grad():
        for X, Y in test_loader:
            X = X.to(device)           # (B, 6, N, F)
            Y = Y.to(device)           # (B, 1, N, 1)
            Y=Y.squeeze(-1)#(b,1,n)

            Y_hat = model(X,A_sets)           # (B, 1, N)

            preds.append(Y_hat.cpu())
            trues.append(Y.cpu())

            mae_list.append(torch.mean(torch.abs(Y_hat - Y)).item())
            rmse_list.append(torch.sqrt(torch.mean((Y_hat - Y) ** 2)).item())
            mape_list.append(torch.mean(torch.abs((Y_hat - Y) / (Y + 1e-5))).item())

    preds = torch.cat(preds, dim=0)    # (T_test, 1, N）
    trues = torch.cat(trues, dim=0)

    metrics = {
        "MAE": np.mean(mae_list),
        "RMSE": np.mean(rmse_list),
        "MAPE": np.mean(mape_list)
    }
    print(metrics)

    return preds, trues, metrics

def save_predictions(preds, trues, save_path):
    """
    保存预测结果为CSV文件
    """
def save_predictions_combined(preds, trues, save_path):

    # 处理维度
    if preds.dim() == 3:
        preds = preds.squeeze(1)
    if trues.dim() == 3:
        trues = trues.squeeze(1)
    
    preds_np = preds.numpy()
    trues_np = trues.numpy()
    
    time_steps, num_nodes = preds_np.shape
    
    # 创建时间索引
    time_idx = pd.RangeIndex(start=0, stop=time_steps, name='time_step')
    
    # 为每个节点创建对比列
    combined_data = {}
    combined_data['time_step'] = time_idx
    
    for node_id in range(num_nodes):
        # 真实值列
        combined_data[f'node_{node_id}_true'] = trues_np[:, node_id]
        # 预测值列
    # 创建DataFrame
    df = pd.DataFrame(combined_data)
    
    # 保存为CSV
    csv_path = f"{save_path}_combined.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"✅ 对比数据已保存: {csv_path}")
    print(f"   时间步数: {time_steps}")
    print(f"   节点数: {num_nodes}")
    print(f"   总列数: {len(df.columns)}")
    print(f"   前5列示例: {df.columns[:5].tolist()}")
    
    return df


def plot_node_prediction(preds, trues, node_id,save_path):
    """
    preds, trues: (T, 1, N)
    """
    preds = preds.squeeze(1)   # (T, N)
    trues = trues.squeeze(1)

    print(f"绘图时preds形状: {preds.shape}")

    plt.figure(figsize=(15, 4))
    plt.plot(trues[:, node_id], label='Ground Truth', linewidth=2)
    plt.plot(preds[:, node_id], label='Prediction', linestyle='--')
    plt.xlabel('Time step')
    plt.ylabel('Density')
    plt.title(f'Node {node_id} Prediction vs Ground Truth')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show() 


def evelate_test(model,dataset, device,A_sets,batch_size,save_path,node_id):
    preds, trues, _= test_and_collect(model, dataset, device,A_sets,batch_size)
    save_predictions(preds,trues,save_path)
    plot_node_prediction(preds,trues,node_id,save_path)
""" 
if __name__=="__main__":
    import pandas as pd
    from features_tensor_building import build_stgcn_tensors
    from road_network import SpatialGraphBuilder
    from model import STGCN
    
    data_path='./data(real)/data.xlsx'
    osm_path="./data(real)/map_road.osm"
    area_df = pd.read_excel(data_path, sheet_name="detector")
    graphbuild=SpatialGraphBuilder(osm_path)
    A_topo,A_dist=graphbuild.build_multi_adjacency(k=2,L=1000,sigma=None,camera_df=area_df,poi_df=None)
    A_sets=[A_dist,A_topo]
    merge_df_10m=pd.read_csv("./merge_df_10m.csv") 
    model=STGCN(num_nodes=num_nodes,in_channels=num_F,hidden_channels=128,num_adj=num_adj,T_in=24,num_blocks=3,dropout=0.2)
    model=torch.load("./checkpoints/best_model.pt")
    train_dataset,val_dataset,test_dataset,num_F=build_stgcn_tensors(merge_df_10m,density_prefix= "_density",time_col="Time",T_in= 24,T_out = 1,train_ratio= 0.7,val_ratio= 0.15)
    evelate_test(model,dataloader=test_dataset, A_sets=A_sets,device="cuda",save_path='./data_out',node_id=3) """





