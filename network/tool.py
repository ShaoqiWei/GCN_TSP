import sys
import numpy as np
import torch
import pandas as pd
from pandas.core.frame import DataFrame
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def write_csv_data(data, tour, save_path):
    name = np.asarray(data.name[0])
    x = data.x * (data.data_max - data.data_min) + data.data_min
    x = x.to('cpu')
    x1 = np.asarray(x[:, 0])
    y1 = np.asarray(x[:, 1])
    x2 = np.asarray(x[:, 2])
    y2 = np.asarray(x[:, 3])
    c = {'name': name[tour],
         'x1': x1[tour],
         'y1': y1[tour],
         'x2': x2[tour],
         'y2': y2[tour]}

    xlsx = DataFrame(c)
    xlsx.to_csv(save_path, index=False)


def read_xlsx_data(data_path):
    data = pd.read_excel(data_path, sheet_name=1, header=1)
    data.columns = ['name', 'x1', 'y1', 'x2', 'y2']

    node_name = data['name'].tolist()

    node_feature_matrix = data[['x1', 'y1', 'x2', 'y2']].values.tolist()
    node_feature_matrix = torch.asarray(node_feature_matrix)
    node_feature_matrix_max_ = torch.max(node_feature_matrix)
    node_feature_matrix_min_ = torch.min(node_feature_matrix)
    node_feature_matrix = (node_feature_matrix - node_feature_matrix_min_) / (
            node_feature_matrix_max_ - node_feature_matrix_min_)

    edges_index = []
    node_num = int(node_feature_matrix.shape[1] / 2)
    edges = np.zeros((node_feature_matrix.shape[0], node_feature_matrix.shape[0], node_num))

    def c_dist(x1, y1):
        return ((x1[0] - y1[0]) ** 2 + (x1[1] - y1[1]) ** 2) ** 0.5

    for i, x in enumerate(node_feature_matrix):
        for j, y in enumerate(node_feature_matrix):
            for k in range(node_num):
                idx = k * 2
                d = c_dist((x[idx], x[idx + 1]), (y[idx], y[idx + 1]))
                edges[i][j][k] = d
            edges_index.append((i, j))

    edges_index = torch.LongTensor(edges_index)
    edges_index = edges_index.transpose(dim0=0, dim1=1)

    edges = edges.reshape(-1, node_num)
    data = Data(x=node_feature_matrix,
                edge_index=edges_index,
                edge_attr=torch.from_numpy(edges).float(),
                data_max=node_feature_matrix_max_,
                data_min=node_feature_matrix_min_,
                name=node_name)

    dl = DataLoader([data], batch_size=1)
    return dl


def creat_instance(node_number: int, node_features_number: int, random_seed=None):
    assert node_features_number > 1 and node_number > 10

    def random_tsp(n_nodes, seed=random_seed):
        if seed is None:
            seed = np.random.randint(123456789)
        np.random.seed(seed)
        return np.random.uniform(0, 1, (n_nodes, node_features_number))

    # Calculate the distance matrix
    def c_dist(x1, y1):
        return ((x1[0] - y1[0]) ** 2 + (x1[1] - y1[1]) ** 2) ** 0.5

    # node feature matrix[node_number, 2]
    node_feature_matrix = random_tsp(node_number, random_seed)

    # edge feature matrix[node_number*node_number, 1]

    edges_index = []
    node_num = int(node_features_number / 2)
    edges = np.zeros((node_number, node_number, node_num))

    for i, x in enumerate(node_feature_matrix):
        for j, y in enumerate(node_feature_matrix):
            for k in range(node_num):
                idx = k * 2
                d = c_dist((x[idx], x[idx + 1]), (y[idx], y[idx + 1]))
                edges[i][j][k] = d
            edges_index.append((i, j))
    edges_index = torch.LongTensor(edges_index)
    edges_index = edges_index.transpose(dim0=0, dim1=1)

    edges = edges.reshape(-1, node_num)
    return node_feature_matrix, edges, edges_index


def creat_data(node_number: int, data_number: int, input_node_dim: int, batch_size: int = 1, random_seed: int = None):
    """
    x: Node feature matrix with shape [num_nodes, num_node_features]-----> 归一化后的x,y坐标 [20, 2]
    edge_index: Edge connectivity with shape [2,num_edges]-----> 边的连接信息（无向图） [2, 400]
    edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]-----> 边的长度（节点与节点之间的距离）[400, 1]

    Args:
        random_seed:
        node_number: 节点个数
        data_number: 数据个数
        input_node_dim: 节点特征矩阵的维度
        batch_size:

    Returns: DataLoader

    """

    datas = []
    for i in range(data_number):
        node, edge, edges_index = creat_instance(node_number, input_node_dim, random_seed)
        data = Data(x=torch.from_numpy(node).float(),
                    edge_index=edges_index,
                    edge_attr=torch.from_numpy(edge).float(),
                    data_max=1,
                    data_min=0)
        datas.append(data)

    dl = DataLoader(datas, batch_size=batch_size)
    return dl


def reward_batch(static, tour_indices, node_number, node_features_number, scaler_min=0, scaler_max=1):
    """
    batch的奖励：为路线的总长度

    Args:
        static: Node feature matrix with shape [num_nodes, num_node_features]
        tour_indices:
        node_number:
        node_features_number:
        scaler_max:
        scaler_min:
    Returns:

    """
    # 反归一化
    static = static * (scaler_max - scaler_min) + scaler_min

    # 将数据大小reshape为[batch_size, node_number, num_node_features]
    static = static.reshape(-1, node_number, node_features_number)

    # 将数据大小transpose为[batch_size, node_features_number, node_number]
    static = static.transpose(2, 1)

    # 将路径数据reshape为[batch_size, node_number]
    tour_indices = tour_indices.reshape(-1, node_number)

    # 路径顺序idx --> [batch_size, 1, node_number] --> [batch_size, node_features_number, node_number]
    idx = tour_indices.unsqueeze(1).expand_as(static)

    # 根据idx排序数据
    # [batch_size, node_features_number, node_number] --> [batch_size, node_number, node_features_number]
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)

    # 首尾相连组成闭环 [batch_size, node_number + 1, node_features_number]
    # y = torch.cat((tour, tour[:, :1]), dim=1)

    # 求相邻两个node之间的距离 [batch_size, node_number]
    y_pow = torch.pow(tour[:, :-1] - tour[:, 1:], 2)
    y_pow_ = torch.tensor_split(y_pow, int(node_features_number / 2), dim=2)

    # [batch_size, node_number]
    tour_len = torch.zeros((y_pow.shape[0], y_pow.shape[1]), device=static.device)
    for y_p in y_pow_:
        y_sum = torch.sum(y_p, dim=2)
        tour_len += torch.sqrt(y_sum)

    # 求和输出 [batch_size]
    return tour_len.sum(1).detach()


def reward(static, tour_indices, node_number, node_features_number, scaler_min=0, scaler_max=1):
    """
    奖励：为路线的总长度
    Args:
        static:
        tour_indices:
        node_number:
        node_features_number:
        scaler_min:
        scaler_max:

    Returns:

    """
    static = static * (scaler_max - scaler_min) + scaler_min
    # print("static size:{}  [0]:{}".format(static.size(), static[0]))
    static = static.reshape(-1, node_number, node_features_number)
    # print("static size:{}  [0]:{}".format(static.size(), static[0]))
    static = static.transpose(2, 1)
    # print("static size:{}  [0]:{}".format(static.size(), static[0]))
    idx = tour_indices.unsqueeze(1).expand_as(static)
    # print("idx size:{}  [0]:{}".format(idx.size(), idx[0]))
    tour = torch.gather(static, 2, idx).permute(0, 2, 1)
    # print("tour size:{}  [0]:{}".format(tour.size(), tour[0]))
    # y = torch.cat((tour, tour[:, :1]), dim=1)
    # print("y size:{}  [0]:{}".format(y.size(), y[0]))
    y_pow = torch.pow(tour[:, :-1] - tour[:, 1:], 2)
    # print("y_pow size:{}  [0]:{}".format(y_pow.size(), y_pow[0]))
    y_pow_ = torch.tensor_split(y_pow, int(node_features_number / 2), dim=2)
    # print("y_pow_[0] size:{}  [0]:{}".format(y_pow_[0].size(), y_pow_[0][0]))
    tour_len = torch.zeros((y_pow.shape[0], y_pow.shape[1]), device=static.device)
    # print("tour_len[0] size:{}  [0]:{}".format(tour_len.size(), tour_len[0]))

    for y_p in y_pow_:
        y_sum = torch.sum(y_p, dim=2)
        tour_len += torch.sqrt(y_sum)
    return tour_len.sum(1).detach()


def rollout(model, dataset, node_number, device, node_features_number):
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model.act(bat, 0, node_number, True, False)
            cost = reward(static=bat.x,
                          tour_indices=cost.detach(),
                          node_number=node_number,
                          node_features_number=node_features_number)
        return cost.cpu()

    total_cost = torch.cat([eval_model_bat(bat.to(device)) for bat in dataset], 0)
    return total_cost
