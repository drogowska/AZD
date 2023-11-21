from typing import Any, Dict
from fcmeans import FCM
from OutlierMarker import detector


class FuzzyCMeansDetector(detector):
    def detect(self, params: Dict[str, Any]) -> None:
        fcm = FCM(**params)
        fcm.random_state = detector.RANDOM_STATE_VALUE
        fcm.fit(self.X)
        self.y_pred = fcm.predict(self.X)
        self._mark_outliers(params["fraction_threshold"])
