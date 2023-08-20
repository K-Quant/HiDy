import torch
import numpy as np
import pandas as pd
import qlib
import datetime
from qlib.config import REG_US, REG_CN

# provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
# provider_uri = "../qlib_data/cn_data"  # target_dir
provider_uri = "../qlib_data/cn_data"
qlib.init(provider_uri=provider_uri, region=REG_CN)
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


class DataLoader:

    def __init__(self, df_feature, df_label, df_market_value, df_stock_index, batch_size=800, pin_memory=True,
                 start_index=0, device=None):

        assert len(df_feature) == len(df_label)

        self.df_feature = df_feature.values
        self.df_label = df_label.values
        self.df_market_value = df_market_value
        self.df_stock_index = df_stock_index
        self.device = device

        if pin_memory:
            self.df_feature = torch.tensor(self.df_feature, dtype=torch.float, device=device)
            self.df_label = torch.tensor(self.df_label, dtype=torch.float, device=device)
            self.df_market_value = torch.tensor(self.df_market_value, dtype=torch.float, device=device)
            self.df_stock_index = torch.tensor(self.df_stock_index, dtype=torch.long, device=device)

        self.index = df_label.index

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.start_index = start_index

        self.daily_count = df_label.groupby(level=0).size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    @property
    def batch_length(self):

        if self.batch_size <= 0:
            # this is the default situation
            return self.daily_length

        return len(self.df_label) // self.batch_size

    @property
    def daily_length(self):
        """
        :return:  number of days in the dataloader
        """
        return len(self.daily_count)

    def iter_batch(self):
        if self.batch_size <= 0:
            yield from self.iter_daily_shuffle()
            return

        indices = np.arange(len(self.df_label))
        np.random.shuffle(indices)

        for i in range(len(indices))[::self.batch_size]:
            if len(indices) - i < self.batch_size:
                break
            yield i, indices[i:i + self.batch_size]  # NOTE: advanced indexing will cause copy

    def iter_daily_shuffle(self):
        indices = np.arange(len(self.daily_count))
        np.random.shuffle(indices)
        for i in indices:
            yield i, slice(self.daily_index[i], self.daily_index[i] + self.daily_count[i])

    def iter_daily(self):
        """
        : yield an index and a slice, that from the day
        """
        indices = np.arange(len(self.daily_count))
        for i in indices:
            yield i, slice(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        # for idx, count in zip(self.daily_index, self.daily_count):
        #     yield slice(idx, idx + count) # NOTE: slice index will not cause copy

    def get(self, slc):
        outs = self.df_feature[slc], self.df_label[slc][:, 0], self.df_market_value[slc], self.df_stock_index[slc]
        # outs = self.df_feature[slc], self.df_label[slc]
        mask = self.padding_mask(outs[0])

        if not self.pin_memory:
            outs = tuple(torch.tensor(x, device=self.device) for x in outs)

        return outs + (self.index[slc], mask, )

    def padding_mask(self, features, max_len=None):
        # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
        lengths = [X.shape[0] for X in features]  # original sequence length for each time series
        if max_len is None:
            max_len = max(lengths)

        padding_masks = self._padding_mask(torch.tensor(lengths, dtype=torch.int16, device=self.device), max_len=max_len)
        # (batch_size, padded_length) boolean tensor, "1" means keep
        return padding_masks

    @staticmethod
    def _padding_mask(lengths, max_len=None):
        """
        Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
        where 1 means keep element at this position (time step)
        """
        batch_size = lengths.numel()
        max_len = max_len or lengths.max_val()
        return (torch.arange(0, max_len, device=lengths.device)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))



def create_test_loaders(args, param_dict,device):
    """
    return a single dataloader for prediction
    """
    start_time = datetime.datetime.strptime(args.test_start_date, '%Y-%m-%d')
    end_time = datetime.datetime.strptime(args.test_end_date, '%Y-%m-%d')
    start_date = args.test_start_date
    end_date = args.test_end_date
    # 此处fit_start_time参照官方文档和代码
    if args.target == 't+0':
        hanlder = {'class': 'Alpha360', 'module_path': 'qlib.contrib.data.handler',
                   'kwargs': {'start_time': start_time, 'end_time': end_time, 'fit_start_time': start_time,
                              'fit_end_time': end_time, 'instruments': param_dict['data_set'], 'infer_processors': [
                           {'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True}},
                           {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}],
                              'learn_processors': [{'class': 'DropnaLabel'},
                                                   {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}}],
                              'label': ['Ref($close, -1) / $close - 1']}}
    else:
        hanlder = {'class': 'Alpha360', 'module_path': 'qlib.contrib.data.handler',
                   'kwargs': {'start_time': start_time, 'end_time': end_time, 'fit_start_time': start_time,
                              'fit_end_time': end_time, 'instruments': param_dict['data_set'], 'infer_processors': [
                           {'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True}},
                           {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}],
                              'learn_processors': [{'class': 'DropnaLabel'},
                                                   {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}}],
                              'label': ['Ref($close, -2) / Ref($close, -1) - 1']}}
    segments = {'test': (start_date, end_date)}
    dataset = DatasetH(hanlder, segments)
    # prepare return a list of df, df_test is the first one
    df_test = dataset.prepare(["test"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L, )[0]
    # ----------------------------------------
    # import pickle5 as pickle
    import pickle
    # only HIST need this
    with open(param_dict['market_value_path'], "rb") as fh:
        # load market value
        df_market_value = pickle.load(fh)
        # the df_market_value save
    df_market_value = df_market_value / 1000000000
    stock_index = np.load(param_dict['stock_index'], allow_pickle=True).item()
    # stock_index is a dict and stock is the key, index is the value
    start_index = 0

    slc = slice(pd.Timestamp(start_date), pd.Timestamp(end_date))
    df_test['market_value'] = df_market_value[slc]
    df_test['market_value'] = df_test['market_value'].fillna(df_test['market_value'].mean())
    df_test['stock_index'] = 733
    df_test['stock_index'] = df_test.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    start_index += len(df_test.groupby(level=0).size())

    test_loader = DataLoader(df_test["feature"], df_test["label"], df_test['market_value'], df_test['stock_index'],
                             pin_memory=True, start_index=start_index, device=device)
    return test_loader
