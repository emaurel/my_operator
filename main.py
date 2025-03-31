from tercen.client import context as ctx
import numpy as np

tercenCtx = ctx.TercenContext(
    workflowId="03efcb35db925880b87a09a004013815",
    stepId="64f5120a-9c2f-41ea-8cd4-5eec8bccb792",
    username="admin", # if using the local Tercen instance
    password="admin", # if using the local Tercen instance
    serviceUri = "http://tercen:5400/" # if using the local Tercen instance
)

df = (
    tercenCtx
    .select(['.y', '.ci', '.ri'], df_lib="pandas")
    .groupby(['.ci','.ri'], as_index=False)
    .mean()
    .rename(columns={".y":"mean"})
    .astype({".ci": np.int32, ".ri": np.int32})
)

df = tercenCtx.add_namespace(df) 
tercenCtx.save(df)
