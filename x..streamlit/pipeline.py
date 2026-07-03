"""
=========================================================
DataPilot AI
Machine Learning Pipeline
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DataPilotPipeline:

    cleaner=None

    feature_engineer=None

    scaler=None

    encoder=None

    selector=None

    model=None

    target=None

    feature_names=None

    problem_type=None

    metadata=Noneclass DataPilotPipeline:

    ...

    def summary(self):

        return {

            "Problem Type":

                self.problem_type,

            "Target":

                self.target,

            "Features":

                len(self.feature_names)
                if self.feature_names
                else 0,

            "Model":

                type(self.model).__name__
                if self.model
                else None

        }

      def predict(self, X):

        return self.model.predict(X)

    def predict_proba(self, X):

        if hasattr(

            self.model,

            "predict_proba"

        ):

            return self.model.predict_proba(X)

        return None

      def feature_importance(self):

        if hasattr(

            self.model,

            "feature_importances_"

        ):

            return self.model.feature_importances_

        return None

import joblib


def save_pipeline(

        pipeline,

        filename

):

    joblib.dump(

        pipeline,

        filename

    )


def load_pipeline(

        filename

):

    return joblib.load(

        filename

    )
