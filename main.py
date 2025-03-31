from tercen.client import context as ctx
import numpy as np

tercenCtx = ctx.TercenContext()

df = (
    tercenCtx
    .select(['.y', '.ci', '.ri'], df_lib="pandas")
    .groupby(['.ci','.ri'], as_index=False)
    .mean()
    .multiply(2)
    .rename(columns={".y":"mean"})
    .astype({".ci": np.int32, ".ri": np.int32})
)

df = tercenCtx.add_namespace(df) 
tercenCtx.save(df)
