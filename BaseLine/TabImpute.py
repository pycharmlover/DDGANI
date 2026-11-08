import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tabimpute.interface import TabImpute
from calc_source.flops import compute_mcpfn_flops


def get_TabImpute_filled(
    nan_data,
    categorical_cols,
    values=None,
    device="cpu"
):
    """
    TabImpute baseline（含 categorical 恢复）

    Returns:
        imputed_X: np.ndarray（categorical 已恢复）
        encoders: Dict[key -> LabelEncoder]
    """

    # ---------- numpy -> DataFrame ----------
    df = pd.DataFrame(nan_data.copy(), columns=values)

    # ---------- categorical encoding ----------
    encoders = {}
    for col_idx in categorical_cols:
        le = LabelEncoder()
        col_name = df.columns[col_idx]

        mask = df[col_name].notna()
        if mask.any():
            df.loc[mask, col_name] = le.fit_transform(
                df.loc[mask, col_name].astype(str)
            )

        key = values[col_idx] if values is not None else col_idx
        encoders[key] = le

    # ---------- DataFrame -> numpy ----------
    X = df.values.astype(float)   # NaN 保留

    # ---------- TabImpute ----------
    imputer = TabImpute(device=device)

    # Calculate FLOPS
    input_dim = X.shape[1]  # number of features
    flops, params = compute_mcpfn_flops(imputer.model, input_dim, device)
    # For TabImpute, we assume 1 forward pass per imputation
    total_flops = flops * 1  # Simplified, could be adjusted based on usage
    with open('calc_source/flops.txt', 'a') as file:
        file.write(f"TabImpute FLOPS: {total_flops}, Params: {params}\n")

    imputed_X = imputer.impute(X)

    # ---------- categorical recovery ----------
    imputed_df = pd.DataFrame(imputed_X, columns=df.columns)

    for col_idx in categorical_cols:
        key = values[col_idx] if values is not None else col_idx
        le = encoders[key]
        col_name = df.columns[col_idx]

        if len(le.classes_) == 0:
            continue  # 全 NaN 列

        # round + clip
        vals = np.round(imputed_df[col_name].values)
        vals = np.clip(vals, 0, len(le.classes_) - 1)

        # inverse transform
        imputed_df[col_name] = le.inverse_transform(vals.astype(int))

    return imputed_df.values
