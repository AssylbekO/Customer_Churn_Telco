from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

contract_order = ["Month-to-month", "One year", "Two year"]

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, num_var, onehot_var, ordinal_var):
        self.num_var = num_var
        self.onehot_var = onehot_var
        self.ordinal_var = ordinal_var
        self.preprocessor = None

    def fit(self, X, y=None):
        numeric_transformer = StandardScaler()
        onehot_transformer = OneHotEncoder()
        ordinal_transformer = OrdinalEncoder(
            categories=[contract_order]
        )

        self.preprocessor = ColumnTransformer([
            ("num", numeric_transformer, self.num_var),
            ("onehot", onehot_transformer, self.onehot_var),
            ("ordinal", ordinal_transformer, self.ordinal_var)
        ], remainder="passthrough", force_int_remainder_cols=False, 
           verbose_feature_names_out=False)

        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        return self.preprocessor.transform(X)

    def get_feature_names(self, input_features=None):
        return self.preprocessor.get_feature_names_out(input_features)
