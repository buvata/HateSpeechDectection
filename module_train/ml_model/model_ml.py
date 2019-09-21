from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgbm


def model_svm():
    params = {
            'kernel': ['linear'],
            'C': [0.5, 1, 10]
        }

    model = SVC()
    name = "svm_model.pkl"
    return model, params, name


def model_random_forest():
    params = {
        "criterion": ["gini", "entropy"],
        "n_estimators": [80],
    }

    model = RandomForestClassifier()
    name = "random_forest_model.pkl"
    return model, params, name


def model_xgboost():
    params = {
        'max_depth': [30, 40],
        'n_estimators': [100, 150],
        'colsample_bytree': [0.2, 0.3],
        'subsample': [0.8]
    }
    model = xgb.XGBClassifier()
    name = "xgboost_model.pkl"
    return model, params, name


def model_lgbm():
    params = {
        'boosting_type': ['gbdt', 'dart'],
        'objective': ['binary'],
        'num_leaves': [30, 40],
        'n_estimators': [100, 150],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.2, 0.3]
    }
    model = lgbm.LGBMClassifier()
    name = "lgbm_model.pkl"
    return model, params, name


def model_gradient_boosting():
    params = {
        "criterion": ["friedman_mse",  "mae"],
        'n_estimators': [80],
        'max_depth': [50],
        'learning_rate': [0.05],
    }

    model = GradientBoostingClassifier()
    name = "graident_boosting_model.pkl"
    return model, params, name
