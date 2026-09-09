import pandas as pd
import plotly.graph_objects as go
from io import StringIO

# Your data
data = """epoch,x,y,z,vx,vy,vz
2024-01-01 00:18:14.850428+00:00,-2269.9061393560305,4364.939005635997,4677.538094850088,-4.565674284955432,-5.44898997068868,2.8655454217147645
"""

# Read CSV data
df = pd.read_csv(StringIO(data))

# Create 3D plot
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=df["x"],
    y=df["y"],
    z=df["z"],
    mode="markers",
    marker=dict(
        size=8,
        color="red"
    ),
    text=df["epoch"],
    hovertemplate=(
        "Epoch: %{text}<br>"
        "X: %{x:.2f}<br>"
        "Y: %{y:.2f}<br>"
        "Z: %{z:.2f}"
        "<extra></extra>"
    )
))

fig.update_layout(
    title="3D Position",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode="data"
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()
