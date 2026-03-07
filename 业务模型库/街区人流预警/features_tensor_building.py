import numpy as np
import pandas as pd
import torch


def extract_node_features(df: pd.DataFrame,density_prefix: str = "_density"):
    density_cols = [c for c in df.columns if c.endswith(density_prefix)]
    if len(density_cols) == 0:
        raise ValueError("未找到 density 列，请检查列名前缀")
    density_values = df[density_cols].values.astype(float)
    X_point = density_values[..., np.newaxis]#添加通道数，满足模型输入要求

    return X_point, density_cols#此时X_point的形状为（数据量，节点数，通道数）

def extract_extra_features(df: pd.DataFrame,X_point=None):
    extra_cols=[c for c in df.columns if c not in ["Time"] and not c.endswith("_density")]
    if len(extra_cols)==0:
        raise ValueError("未找到外部特征列，请检查数据")
    extra_values=df[extra_cols].values.astype(float)
    X_extra=extra_values[:,np.newaxis,:]#添加通道数，满足模型输入要求
    X_extra=np.repeat(X_extra,X_point.shape[1],axis=1)#在节点维度上进行重复扩展，使其与节点特征的节点数匹配,X_extra的形状为（数据量，节点数，通道数）
    return X_extra, extra_cols

def combine_features(X_point,X_extra):
    if X_point.shape[0]!=X_extra.shape[0]:
        raise ValueError("节点特征和外部特征的时间步数不匹配")
    X_combined=np.concatenate([X_point,X_extra],axis=-1)#在通道维度上进行拼接

    return X_combined

def make_sliding_windows(X_combined,T_in=6,T_out=1):
    X_list, Y_list = [], []
    T_total = X_combined.shape[0]

    for i in range(T_total - T_in - T_out + 1):
        X_list.append(X_combined[i:i + T_in])#此时X_list的形状为（样本数，T_in，节点数，通道数），在图神经网络中，通道数就等于特征数
        Y_list.append(
            X_combined[i + T_in:i + T_in + T_out, :, 0]#只取第0个通道
        )#此时Y_list的形状为（样本数，T_out，节点数）

    return np.array(X_list), np.array(Y_list)


def split_by_time(X,Y,train_ratio=0.7,val_ratio= 0.15):
    num_samples = X.shape[0]

    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))

    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
    X_test, Y_test = X[val_end:], Y[val_end:]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def build_stgcn_tensors(df,density_prefix= "_density",time_col="Time",T_in= 6,T_out = 1,train_ratio= 0.7,val_ratio= 0.15):
    if time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)

    X_point, node_cols = extract_node_features(df, density_prefix=density_prefix)
    X_extra, extra_cols = extract_extra_features(df, X_point=X_point)
    X_combined = combine_features(X_point, X_extra)
    X, Y = make_sliding_windows(X_combined, T_in=T_in, T_out=T_out)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_by_time(X, Y,train_ratio=train_ratio,val_ratio=val_ratio )
    num_F=X.shape[3]
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_val, dtype=torch.float32)
    )  
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(Y_test, dtype=torch.float32)
    )

    print("===== STGCN 特征张量构造完成 =====")
    print(f"节点数 N = {X.shape[2]}")
    print(f"特征数 F = {X.shape[3]}")
    print(f"输入时间步 T_in = {T_in}")
    print(f"预测时间步 T_out = {T_out}")
    print(f"X_train: {X_train.shape}")
    print(f"Y_train: {Y_train.shape}")
    print(f"X_val:   {X_val.shape}")
    print(f"X_test:  {X_test.shape}")

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        num_F
    )

if __name__=="__main__":
    df=pd.read_csv("merge_df_10m.csv")
    build_stgcn_tensors(df)
