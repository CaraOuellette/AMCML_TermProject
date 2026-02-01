"""
Automated Feature Selection Tool

This module provides functions for various feature selection methods
and an automated selector that combines them using majority voting.

This module is dataset-agnostic and intended for use AFTER train/test split.
It supports classification problems and should be used on preprocessed data.
"""

import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


def cor_selector(X: pd.DataFrame, y: pd.Series, num_feats: int) -> Tuple[List[bool], List[str]]:
    """
    Select features based on Pearson correlation with the target.

    Args:
        X: Feature matrix
        y: Target vector
        num_feats: Number of features to select

    Returns:
        Tuple of (support mask, selected feature names)
    """
    cor_list = []
    for col in X.columns:
        cor = np.corrcoef(X[col], y)[0, 1]
        cor_list.append(0 if np.isnan(cor) else cor)

    top_indices = np.argsort(np.abs(cor_list))[-num_feats:]
    selected_features = X.columns[top_indices].tolist()
    support = [col in selected_features for col in X.columns]
    return support, selected_features


def chi_squared_selector(X: pd.DataFrame, y: pd.Series, num_feats: int) -> Tuple[List[bool], List[str]]:
    """
    Select features using Chi-Squared test.

    Args:
        X: Feature matrix (should be non-negative)
        y: Target vector
        num_feats: Number of features to select

    Returns:
        Tuple of (support mask, selected feature names)
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    selector = SelectKBest(chi2, k=num_feats)
    selector.fit(X_scaled, y)
    support = selector.get_support().tolist()
    features = X.columns[support].tolist()
    return support, features


def rfe_selector(X: pd.DataFrame, y: pd.Series, num_feats: int) -> Tuple[List[bool], List[str]]:
    """
    Select features using Recursive Feature Elimination with Logistic Regression.

    Args:
        X: Feature matrix
        y: Target vector
        num_feats: Number of features to select

    Returns:
        Tuple of (support mask, selected feature names)
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(random_state=42, max_iter=1000)
    rfe = RFE(model, n_features_to_select=num_feats)
    rfe.fit(X_scaled, y)
    support = rfe.support_.tolist()
    features = X.columns[support].tolist()
    return support, features


def embedded_log_reg_selector(X, y, num_feats):
    """
    Embedded feature selection using Logistic Regression with L1 regularization.
    Universal across sklearn versions and supports binary & multiclass targets.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import SelectFromModel
    from sklearn.preprocessing import StandardScaler

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine number of classes
    n_classes = y.nunique()

    # Choose solver safely (no multi_class argument, use l2 for multiclass)
    if n_classes > 2:
        model = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            random_state=42,
            max_iter=3000
        )
    else:
        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            random_state=42,
            max_iter=1000
        )

    selector = SelectFromModel(model, max_features=num_feats)
    selector.fit(X_scaled, y)

    support = selector.get_support().tolist()
    features = X.columns[support].tolist()

    return support, features

def embedded_rf_selector(X: pd.DataFrame, y: pd.Series, num_feats: int) -> Tuple[List[bool], List[str]]:
    """
    Select features using embedded method with Random Forest.

    Args:
        X: Feature matrix
        y: Target vector
        num_feats: Number of features to select

    Returns:
        Tuple of (support mask, selected feature names)
    """
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    selector = SelectFromModel(model, threshold=-np.inf, max_features=num_feats)
    selector.fit(X, y)
    support = selector.get_support().tolist()
    features = X.columns[support].tolist()
    return support, features


def embedded_lgbm_selector(X: pd.DataFrame, y: pd.Series, num_feats: int) -> Tuple[List[bool], List[str]]:
    """
    Select features using embedded method with LightGBM.

    Args:
        X: Feature matrix
        y: Target vector
        num_feats: Number of features to select

    Returns:
        Tuple of (support mask, selected feature names)
    """
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        verbosity=-1  # Suppress warnings
    )
    selector = SelectFromModel(model, threshold=-np.inf, max_features=num_feats)
    selector.fit(X, y)
    support = selector.get_support().tolist()
    features = X.columns[support].tolist()
    return support, features


def autoFeatureSelector(X: pd.DataFrame, y: pd.Series, num_feats: int, methods: List[str]) -> List[str]:
    """
    Automatically select the best features using multiple methods and majority voting.

    Args:
        X: Feature matrix
        y: Target vector
        num_feats: Number of features to select
        methods: List of method names to use

    Returns:
        List of selected feature names (up to num_feats)
    """
    # Define available methods
    method_dict = {
        'pearson': cor_selector,
        'chi-square': chi_squared_selector,
        'rfe': rfe_selector,
        'log-reg': embedded_log_reg_selector,
        'rf': embedded_rf_selector,
        'lgbm': embedded_lgbm_selector
    }

    # Collect features from selected methods
    all_selected_features = []
    for method in methods:
        if method in method_dict:
            _, features = method_dict[method](X, y, num_feats)
            all_selected_features.extend(features)

    if not all_selected_features:
        return []

    # Majority voting
    feature_counts = pd.Series(all_selected_features).value_counts()
    best_features = feature_counts.head(num_feats).index.tolist()

    return best_features