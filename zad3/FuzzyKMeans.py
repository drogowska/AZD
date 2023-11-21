import numpy as np
from sklearn.cluster import KMeans
from typing import Any, Dict
from OutlierMarker import detector




class KMeansDetector(detector):

    def detect(self, params: Dict[str, Any]) -> None:
        filtered_params = {
            key: value
            for key, value in params.items()
            if key != "fraction_threshold"
        }

        k_means: KMeans = KMeans(**filtered_params, n_init=10)
        k_means.random_state = detector.RANDOM_STATE_VALUE
        self.y_pred = k_means.fit_predict(self.X).astype(np.float32)
        self._mark_outliers(params["fraction_threshold"])
