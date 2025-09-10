"""Gateway notebook for https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/"""

import os
from pathlib import Path

import pandas as pd
import polars as pl

import kaggle_evaluation.core.base_gateway
import kaggle_evaluation.core.templates


class MitsuiGateway(kaggle_evaluation.core.templates.Gateway):
    def __init__(self, data_paths: tuple[str] | None = None):
        super().__init__(data_paths, file_share_dir=None)
        self.data_paths = data_paths
        self.row_id_column_name = 'date_id'
        self.set_response_timeout_seconds(60 * 5)

    def unpack_data_paths(self):
        if not self.data_paths:
            self.competition_data_dir = '/kaggle/input/mitsui-commodity-prediction-challenge/'
        else:
            self.competition_data_dir = self.data_paths[0]
        self.competition_data_dir = Path(self.competition_data_dir)

    def generate_data_batches(self):
        """按日滚动喂数，传入“≤当日”的历史窗口，保证训练-推理同构。

        - test_hist: 截止当日的完整可见历史（用于构造滚动特征，避免前视）
        - label_lags_k_batch: 当日的列名，用于确定每个 lag 的目标列输出顺序
        """
        test = pl.read_csv(self.competition_data_dir / 'test.csv')

        label_lag_dir = self.competition_data_dir / 'lagged_test_labels'
        label_lags_1 = pl.read_csv(label_lag_dir / 'test_labels_lag_1.csv')
        label_lags_2 = pl.read_csv(label_lag_dir / 'test_labels_lag_2.csv')
        label_lags_3 = pl.read_csv(label_lag_dir / 'test_labels_lag_3.csv')
        label_lags_4 = pl.read_csv(label_lag_dir / 'test_labels_lag_4.csv')

        date_ids = test['date_id'].unique(maintain_order=True).to_list()
        for date_id in date_ids:
            test_hist = test.filter(pl.col('date_id') <= date_id)
            label_lags_1_batch = label_lags_1.filter(pl.col('date_id') == date_id)
            label_lags_2_batch = label_lags_2.filter(pl.col('date_id') == date_id)
            label_lags_3_batch = label_lags_3.filter(pl.col('date_id') == date_id)
            label_lags_4_batch = label_lags_4.filter(pl.col('date_id') == date_id)

            yield (
                (test_hist, label_lags_1_batch, label_lags_2_batch, label_lags_3_batch, label_lags_4_batch),
                date_id,
            )

    def competition_specific_validation(self, prediction, row_ids, data_batch) -> None:
        assert isinstance(prediction, (pd.DataFrame, pl.DataFrame))
        assert len(prediction) == 1
        assert 'date_id' not in prediction.columns
        provided_label_lags = pl.concat([i.drop(['date_id', 'label_date_id']) for i in data_batch[1:]], how='horizontal')
        assert len(prediction.columns) == len(provided_label_lags.columns)


if __name__ == '__main__':
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        gateway = MitsuiGateway()
        # Relies on valid default data paths
        gateway.run()
    else:
        print('Skipping run for now')
