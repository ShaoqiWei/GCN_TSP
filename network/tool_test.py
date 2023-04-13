import os
import numpy as np
import torch
from tool import reward_batch, creat_data, creat_instance, read_xlsx_data, write_svg_for_cvs
import pandas as pd

if __name__ == '__main__':
    # SEED = 0
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)
    #
    # batch_size = 15
    # node_number = 20
    # node_features_number = 4
    #
    # static = torch.randn(batch_size * node_number * node_features_number, device='cuda')
    # tour_indices = torch.randint(low=0, high=20, size=(batch_size * node_number,), device='cuda')
    # #
    # # print(tour_indices.shape)
    # # print(static.shape)
    # #
    # print(static.device)
    # r = reward_batch(static=static,
    #                  tour_indices=tour_indices,
    #                  node_number=node_number,
    #                  node_features_number=node_features_number)
    # print(r)
    # print("r: ", r.size())
    # print(r.get_device())
    #
    # data, edge, index_list = creat_instance(node_number=node_number,
    #                                         node_features_number=node_features_number)
    #
    # print(data.shape, edge.shape, index_list.shape)

    # data = read_xlsx_data("../data/data.xlsx")
    # print(1)
    # data = pd.read_excel("../data/data.xlsx", sheet_name=1, header=1)
    # data.columns = ['name', 'x1', 'y1', 'x2', 'y2']
    # node_name = data['name'].tolist()
    #
    # node_feature_matrix = data[['x1', 'y1', 'x2', 'y2']].values.tolist()
    # node_feature_matrix = torch.asarray(node_feature_matrix)
    # d = node_feature_matrix[1:, :2] - node_feature_matrix[:-1, :2]
    # d = torch.sum(torch.pow(d, 2), dim=1)
    # print(d.size())
    #
    # # 27708.22
    # d = torch.sqrt(d)
    # print(torch.sum(d))
    # 14010.9385 + 13477.5586
    write_svg_for_cvs('../data/result.csv', '../data/result')
    # write_svg_for_cvs('../data/fly_probe1_data.csv.csv', '')
    pass
