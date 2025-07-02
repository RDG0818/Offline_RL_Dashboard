import plotly.graph_objs as go

def example_plot():
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=[1, 3, 2, 4], mode='lines+markers', name='Sample'))
    fig.update_layout(title="Example Plot", xaxis_title="Step", yaxis_title="Value")
    return fig
