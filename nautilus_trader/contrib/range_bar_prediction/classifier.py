# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2023 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

"""
Directional classifier for range bar prediction.

This module provides a logistic regression classifier with isotonic calibration
for predicting the direction of the next range bar.
"""

from typing import Optional
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning

from nautilus_trader.contrib.range_bar_prediction.data import BarFeatures


class DirectionalClassifier:
    """
    Logistic regression classifier for bar direction prediction.

    Uses isotonic calibration to improve probability estimates.
    Tracks out-of-sample performance metrics for honest evaluation.

    The classifier predicts P(next bar is up | features), where "up"
    means direction = +1.

    Parameters
    ----------
    penalty : str, default "l2"
        Regularization penalty ('l1', 'l2', 'elasticnet', or 'none').
    C : float, default 1.0
        Inverse regularization strength. Smaller values = stronger regularization.
    calibrate : bool, default True
        Whether to use isotonic calibration for probability estimates.
    cv_folds : int, default 5
        Number of cross-validation folds for calibration.
    random_state : int, default 42
        Random seed for reproducibility.

    Examples
    --------
    >>> classifier = DirectionalClassifier()
    >>> success = classifier.fit(training_features, min_samples=500)
    >>> if success:
    ...     p_up = classifier.predict_proba(new_features)
    ...     metrics = classifier.evaluate(validation_features)
    """

    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1.0,
        calibrate: bool = True,
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        self.penalty = penalty
        self.C = C
        self.calibrate = calibrate
        self.cv_folds = cv_folds
        self.random_state = random_state

        self.model: Optional[CalibratedClassifierCV | LogisticRegression] = None
        self.is_fitted: bool = False
        self.feature_names: list[str] = BarFeatures.get_feature_names()

        # Training metadata
        self.n_training_samples: int = 0
        self.training_class_balance: Optional[float] = None

    def fit(
        self,
        features: list[BarFeatures],
        min_samples: int = 500,
    ) -> bool:
        """
        Train the classifier on labeled feature vectors.

        Parameters
        ----------
        features : list[BarFeatures]
            List of BarFeatures with next_direction filled in.
        min_samples : int, default 500
            Minimum samples required to fit (prevents overfitting on small data).

        Returns
        -------
        bool
            True if fitting succeeded, False if insufficient data.
        """
        # Filter to features with known targets
        labeled = [f for f in features if f.next_direction is not None]

        if len(labeled) < min_samples:
            return False

        X = np.array([f.to_array() for f in labeled])
        y = np.array([1 if f.next_direction == 1 else 0 for f in labeled])

        # Store training metadata
        self.n_training_samples = len(labeled)
        self.training_class_balance = np.mean(y)

        # Suppress convergence warnings during fitting
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

            base_model = LogisticRegression(
                penalty=self.penalty,
                C=self.C,
                max_iter=1000,
                random_state=self.random_state,
                solver="lbfgs" if self.penalty == "l2" else "saga",
            )

            if self.calibrate and len(labeled) >= self.cv_folds * 20:
                # Use calibrated classifier for better probability estimates
                self.model = CalibratedClassifierCV(
                    estimator=base_model,
                    method="isotonic",
                    cv=self.cv_folds,
                )
            else:
                # Fall back to uncalibrated if not enough data for CV
                self.model = base_model

            self.model.fit(X, y)

        self.is_fitted = True
        return True

    def predict_proba(self, features: BarFeatures) -> float:
        """
        Predict probability that next bar will be +1 (up).

        Parameters
        ----------
        features : BarFeatures
            Feature vector to predict on.

        Returns
        -------
        float
            Probability in [0, 1] that next bar is up.
            Returns 0.5 if model not fitted (no information).
        """
        if not self.is_fitted:
            return 0.5

        X = features.to_array().reshape(1, -1)
        proba = self.model.predict_proba(X)[0, 1]  # P(class=1) = P(up)

        return float(proba)

    def predict_probas(self, features: list[BarFeatures]) -> np.ndarray:
        """
        Predict probabilities for a batch of feature vectors.

        Parameters
        ----------
        features : list[BarFeatures]
            List of feature vectors to predict on.

        Returns
        -------
        np.ndarray
            Array of probabilities, each in [0, 1].
        """
        if not self.is_fitted:
            return np.full(len(features), 0.5)

        X = np.array([f.to_array() for f in features])
        probas = self.model.predict_proba(X)[:, 1]

        return probas

    def predict_direction(
        self,
        features: BarFeatures,
        threshold: float = 0.5,
    ) -> tuple[int, float]:
        """
        Predict direction and confidence.

        Parameters
        ----------
        features : BarFeatures
            Feature vector to predict on.
        threshold : float, default 0.5
            Probability threshold for direction.

        Returns
        -------
        tuple[int, float]
            (predicted_direction, confidence)
            direction is +1 or -1, confidence is in [0.5, 1.0].
        """
        p_up = self.predict_proba(features)

        if p_up >= threshold:
            return (1, p_up)
        else:
            return (-1, 1 - p_up)

    def evaluate(self, features: list[BarFeatures]) -> dict:
        """
        Compute out-of-sample performance metrics.

        Parameters
        ----------
        features : list[BarFeatures]
            List of features with next_direction filled in.

        Returns
        -------
        dict
            Dictionary containing:
            - brier_score: Lower is better, 0.25 is no-skill baseline
            - accuracy: Fraction correctly predicted
            - edge: Accuracy - 0.5 (positive means better than chance)
            - n_samples: Number of samples evaluated
            - class_balance: Fraction of up bars in validation set
        """
        labeled = [f for f in features if f.next_direction is not None]

        if len(labeled) == 0:
            return {
                "brier_score": None,
                "accuracy": None,
                "edge": None,
                "n_samples": 0,
                "class_balance": None,
            }

        predictions = []
        actuals = []

        for f in labeled:
            p_up = self.predict_proba(f)
            actual = 1 if f.next_direction == 1 else 0
            predictions.append(p_up)
            actuals.append(actual)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Brier score: mean squared error of probability predictions
        brier = float(np.mean((predictions - actuals) ** 2))

        # Accuracy: did we predict the right direction?
        predicted_direction = (predictions > 0.5).astype(int)
        accuracy = float(np.mean(predicted_direction == actuals))

        # Log loss (cross-entropy)
        eps = 1e-15
        predictions_clipped = np.clip(predictions, eps, 1 - eps)
        log_loss = float(-np.mean(
            actuals * np.log(predictions_clipped)
            + (1 - actuals) * np.log(1 - predictions_clipped)
        ))

        return {
            "brier_score": brier,
            "accuracy": accuracy,
            "edge": accuracy - 0.5,
            "log_loss": log_loss,
            "n_samples": len(labeled),
            "class_balance": float(np.mean(actuals)),
        }

    def get_feature_importance(self) -> Optional[dict[str, float]]:
        """
        Get feature importance scores from the model coefficients.

        Returns
        -------
        Optional[dict[str, float]]
            Dictionary mapping feature names to importance scores.
            Returns None if model not fitted or not accessible.
        """
        if not self.is_fitted:
            return None

        try:
            # Try to get coefficients from base model
            if hasattr(self.model, "coef_"):
                coefs = self.model.coef_[0]
            elif hasattr(self.model, "estimator") and hasattr(
                self.model.estimator, "coef_"
            ):
                coefs = self.model.estimator.coef_[0]
            elif hasattr(self.model, "calibrated_classifiers_"):
                # For CalibratedClassifierCV, average across calibrators
                coefs_list = []
                for clf in self.model.calibrated_classifiers_:
                    if hasattr(clf.estimator, "coef_"):
                        coefs_list.append(clf.estimator.coef_[0])
                if coefs_list:
                    coefs = np.mean(coefs_list, axis=0)
                else:
                    return None
            else:
                return None

            # Return normalized absolute importance
            abs_coefs = np.abs(coefs)
            importance = abs_coefs / np.sum(abs_coefs)

            return dict(zip(self.feature_names, importance))

        except Exception:
            return None

    def reset(self) -> None:
        """Reset the classifier to unfitted state."""
        self.model = None
        self.is_fitted = False
        self.n_training_samples = 0
        self.training_class_balance = None
