import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import os

class TSNEVisualization:
    def __init__(self, df, title="T-SNE Embedding Visualization"):
        """Initializes the T-SNE visualization class."""
        self.df = self._perform_tsne(df)
        self.title = title
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
        self._build_layout()
    
    def _perform_tsne(self, df):
        """Applies T-SNE for dimensionality reduction."""
        embeddings = np.array(df["abstract_embeddings"].tolist())
        tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings)
        tsne_3d = TSNE(n_components=3, perplexity=30, random_state=42).fit_transform(embeddings)
        df["tsne_x_2d"], df["tsne_y_2d"] = tsne_2d[:, 0], tsne_2d[:, 1]
        df["tsne_x_3d"], df["tsne_y_3d"], df["tsne_z_3d"] = tsne_3d[:, 0], tsne_3d[:, 1], tsne_3d[:, 2]
        return df
    
    def _build_layout(self):
        """Constructs the Dash layout."""
        self.app.layout = dbc.Container([
            html.H1(self.title, className="text-center mt-4 mb-4"),
            dcc.Dropdown(
                id="venue-selector",
                options=[{"label": v, "value": v} for v in sorted(self.df["venue"].unique())],
                multi=True,
                placeholder="Select venues to display",
                className="mb-4"
            ),
            dbc.Row([
                dbc.Col(dcc.Graph(id="tsne-2d-plot"), width=6),
                dbc.Col(dcc.Graph(id="tsne-3d-plot"), width=6)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="heatmap-plot"), width=12)
            ]),
            dbc.Button("Download HTML Report", id="download-button", className="mt-4 mb-4", color="primary"),
            dcc.Download(id="download-html")
        ], fluid=True)
        
        @self.app.callback(
            [Output("tsne-2d-plot", "figure"), Output("tsne-3d-plot", "figure"), Output("heatmap-plot", "figure")],
            [Input("venue-selector", "value")]
        )
        def update_plots(selected_venues):
            """Updates the plots based on selected venues."""
            filtered_df = self.df[self.df["venue"].isin(selected_venues)] if selected_venues else self.df
            
            fig_2d = px.scatter(
                filtered_df, x="tsne_x_2d", y="tsne_y_2d", color="venue",
                title="T-SNE 2D Visualization", hover_data=["venue"], template="plotly_white"
            )
            
            fig_3d = px.scatter_3d(
                filtered_df, x="tsne_x_3d", y="tsne_y_3d", z="tsne_z_3d", color="venue",
                title="T-SNE 3D Visualization", hover_data=["venue"], template="plotly_white"
            )
            
            fig_heatmap = go.Figure(
                data=go.Histogram2dContour(
                    x=filtered_df["tsne_x_2d"], y=filtered_df["tsne_y_2d"],
                    colorscale="Viridis", contours=dict(showlabels=True)
                )
            )
            fig_heatmap.update_layout(
                title="T-SNE Heatmap of Abstract Embeddings",
                xaxis_title="TSNE X", yaxis_title="TSNE Y", template="plotly_white"
            )
            
            return fig_2d, fig_3d, fig_heatmap
        
        @self.app.callback(
            Output("download-html", "data"),
            Input("download-button", "n_clicks"),
            prevent_initial_call=True
        )
        def download_html(n_clicks):
            """Exports the visualization as an HTML file."""
            file_path = "tsne_visualization.html"
            self.app.server.run("dash_app", port=8050, debug=False)
            return dcc.send_file(file_path)
    
    def run(self):
        """Runs the Dash app."""
        self.app.run_server(debug=True)
