from typing import Optional
from .strategy_funcs import process_dataframe, prepare_data, ensure_utc
import faiss
import numpy as np
import pandas as pd
import traceback

class FaissKNNClassifier:
    """A multiclass exact KNN classifier implemented using the FAISS library."""

    def __init__(
        self, n_neighbors: int, n_classes: Optional[int] = None, device: str = "cpu"
    ) -> None:
        """Instantiate a faiss KNN Classifier.

        Args:
            n_neighbors: number of KNN neighbors
            n_classes: (optional) number of dataset classes
                (otherwise derive from the data)
            device: a torch device, e.g. cpu, cuda, cuda:0, etc.
        """
        self.n_neighbors = n_neighbors
        self.n_classes = n_classes

        if device == "cpu":
            self.cuda = False
            self.device = None
        else:
            self.cuda = True
            if ":" in device:
                self.device = int(device.split(":")[-1])
            else:
                self.device = 0

    def create_index(self, d: int) -> None:
        """Create the faiss index.

        Args:
            d: feature dimension
        """
        if self.cuda:
            self.res = faiss.StandardGpuResources()
            self.config = faiss.GpuIndexFlatConfig()
            self.config.device = self.device
            self.index = faiss.GpuIndexFlatL2(self.res, d, self.config)
        else:
            self.index = faiss.IndexFlatL2(d)

    def fit(self, X: np.ndarray, y: np.ndarray) -> object:
        """Store train X and y.

        Args:
            X: input features (N, d)
            y: input labels (N, ...)

        Returns:
            self
        """
        X = np.atleast_2d(X).astype(np.float32)
        X = np.ascontiguousarray(X)
        self.create_index(X.shape[-1])
        self.index.add(X)
        self.y = y.astype(int)
        if self.n_classes is None:
            self.n_classes = len(np.unique(y))
        return self

    def __del__(self) -> None:
        """Cleanup helpers."""
        if hasattr(self, "index"):
            self.index.reset()
            del self.index
        if hasattr(self, "res"):
            self.res.noTempMemory()
            del self.res

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict int labels given X.

        Args:
            X: input features (N, d)

        Returns:
            preds: int predicted labels (N,)
        """
        X = np.atleast_2d(X).astype(np.float32)
        _, idx = self.index.search(X, self.n_neighbors)
        class_idx = self.y[idx]
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_classes),
            axis=1,
            arr=class_idx.astype(np.int16),
        )
        preds = np.argmax(counts, axis=1)
        return preds, idx

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict float probabilities for labels given X.

        Args:
            X: input features (N, d)

        Returns:
            preds_proba: float probas per labels (N, c)
        """
        X = np.atleast_2d(X).astype(np.float32)
        _, idx = self.index.search(X, self.n_neighbors)
        class_idx = self.y[idx]
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_classes),
            axis=1,
            arr=class_idx.astype(np.int16),
        )

        preds_proba = counts / self.n_neighbors
        return preds_proba, idx


class KNNClassifier:
    def __init__(self, data_path:str, threshold:float, n_neighbors:int, window_size:int=28, n_points:int=2500, update_interval:int=24):
        self.knn_clf = FaissKNNClassifier(n_neighbors=n_neighbors)
        self.data_path = data_path
        self.window_size = window_size
        self.threshold = threshold
        self.n_points = n_points
        self.update_interval = update_interval
        try:
            self.profits_df = pd.read_csv(data_path)
            ensure_utc(self.profits_df, ['startDate', 'endDate'])
        except FileNotFoundError:
            raise ValueError(f"Dataset file not found at {data_path}")
        except Exception as e:
            raise ValueError(f"Error loading dataset from {data_path}: {e}")
        

    def get_n_closest(self, startDate, n=2500, starting_count=0):
        """
        Return n closest points in time to startDate discarding first starting_count  
        """
        try:
            data = self.profits_df.drop(columns='predictions')
            previous_data = data[data['endDate'] < startDate - pd.Timedelta(hours=starting_count)]
            res = previous_data.head(n)
            drop_cols = ['startDate', 'endDate', 'profits', 'profit_curado']
            X_train, y_train = res.drop(columns=drop_cols), res['profit_curado']
            return np.array(X_train), np.array(y_train), np.array(res.index)
        except KeyError as e:
            raise ValueError(f"Missing column in profits_df: {e}")
        except Exception as e:
            raise ValueError(f"Error in get_n_closest method: {e}")
    
    def update_profits_df(self, current_data, cols):
        """
        Update profits dataframe with new data every update_interval hours
        """
        try:
            cols = ['date_1h'] + cols
            ensure_utc(current_data, 'date_1h')
            renamed_cols = [x.split('_')[0] for x in cols]
            current_data = current_data.loc[:, cols].copy()
            current_data.columns = renamed_cols
            last_profits_date = self.profits_df['endDate'].max()
            current_date = current_data['date'].max()
            time_diff = current_date - last_profits_date

            print(f"Current date: {current_date}, Last profits date: {last_profits_date}")
            print(f"Time difference between current and last profits date: {time_diff}")

            if time_diff >= pd.Timedelta(hours=self.update_interval):
                updated_profits = prepare_data(current_data, window_size=self.window_size)
                updated_profits['predictions'] = np.nan
                combined_df = pd.concat([updated_profits, self.profits_df])
                combined_df = combined_df.drop_duplicates(subset='startDate', keep='last')  
                combined_df = combined_df.sort_values(by='endDate', ascending=False)
                previous_length = len(self.profits_df)
                current_length = len(combined_df)

                max_rows = self.n_points + 1000 + 48 
                if len(combined_df) > max_rows:
                    combined_df = combined_df.head(max_rows)  # Keep only the most recent max_rows rows
                combined_df.to_csv(self.data_path, index=False)  

                self.profits_df = combined_df  

                print(f"Updated dataframe with {current_length - previous_length} new rows.")
            else:
                self.profits_df = self.profits_df
        except KeyError as e:
            raise ValueError(f"Error updating profits_df: Missing column {e}")
        except Exception as e:
            raise ValueError(f"Error updating profits_df: {e}")

    def check_prediction(self, date):
        """
        Check if a prediction for a given date already exists in dataframe
        """
        prediction = self.profits_df.loc[self.profits_df.startDate == date, 'predictions']
        if len(prediction) == 0:
            return None
        return prediction.values[0]

    
    def predict(self, dataframe, cols):
        try:
            dataframe['date'] = pd.to_datetime(dataframe['date'])
            self.update_profits_df(dataframe, cols)
            data = process_dataframe(dataframe, lags=self.window_size, cols=cols)
            dates = data.index
            predictions = []
            n_precomputed = 0
            n_computed = 0

            print(f'Computing {len(dates)} predictions')
            for date in dates:
                # Check if the prediction already exists
                existing_prediction = self.check_prediction(date)
                
                if pd.notna(existing_prediction):
                    predictions.append(existing_prediction)
                    n_precomputed += 1
                    continue
                try:
                    X, y, _ = self.get_n_closest(date, n=self.n_points, starting_count=1)
                    current_data = np.array(data.loc[date])

                    if len(y) == self.n_points:
                        self.knn_clf.fit(X, y)
                        pred_proba, _ = self.knn_clf.predict_proba(current_data)
                        pred = pred_proba[0, 1] >= self.threshold
                        pred = int(pred)
                        self.profits_df.loc[self.profits_df.startDate == date, 'predictions'] = pred
                        predictions.append(pred)
                        n_computed += 1
                    else:
                        print(f"Warning: Not enough neighbors found for date {date}. Found: {len(y)}, Expected: {self.n_points}")
                except Exception as e:
                    print(f"Error during prediction for date {date}: {e}")
                    traceback.print_exc()
                    predictions.append(None)  # Append None or any fallback value in case of error
            
            print(f'{n_precomputed} precomputed predictions and {n_computed} new predictions.')
            #self.profits_df.to_csv('profits_28h_curados.csv', index=False) # Save updated dataframe to disk
            return predictions
        except Exception as e:
            raise ValueError(f"Error in predict method: {e}")