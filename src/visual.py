import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import json
import logging
import sys
from tqdm import tqdm
from sklearn.decomposition import PCA
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc


class PCAVisualization:
    def __init__(self, df, title="PCA Embedding Visualization"):
        if df["abstract_embeddings"].apply(lambda x: isinstance(x, str)).any():
            logging.info("Converting string embeddings to list.")
            df["abstract_embeddings"] = [
                json.loads(x) if isinstance(x, str) else x
                for x in tqdm(df["abstract_embeddings"])
            ]
        logging.info("Performing PCA.")
        self.df = self._perform_pca(df)
        self.title = title
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
        self._build_layout()
    
    def _perform_pca(self, df):
        embeddings = np.array(df["abstract_embeddings"].tolist())
        logging.info("Computing 2D PCA with shape %s", embeddings.shape)
        pca_2d = PCA(n_components=2).fit_transform(embeddings)
        logging.info("2D PCA done. Computing 3D PCA.")
        pca_3d = PCA(n_components=3).fit_transform(embeddings)
        logging.info("3D PCA done.")
        df["pca_x_2d"], df["pca_y_2d"] = pca_2d[:, 0], pca_2d[:, 1]
        df["pca_x_3d"], df["pca_y_3d"], df["pca_z_3d"] = pca_3d[:, 0], pca_3d[:, 1], pca_3d[:, 2]
        return df
    
    def _build_layout(self):
        self.app.layout = dbc.Container([
            html.H1(self.title, className="text-center mt-4 mb-4"),
            dcc.Dropdown(
                id="venue-selector",
                options=[{"label": v, "value": v} for v in sorted(self.df["venue_numeric"].unique())],
                multi=True,
                placeholder="Select venues to display",
                className="mb-4"
            ),
            dbc.Row([
                dbc.Col(dcc.Graph(id="pca-2d-plot"), width=6),
                dbc.Col(dcc.Graph(id="pca-3d-plot"), width=6)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="heatmap-plot"), width=12)
            ]),
            dbc.Button("Download HTML Report", id="download-button", className="mt-4 mb-4", color="primary"),
            dcc.Download(id="download-html")
        ], fluid=True)
        
        @self.app.callback(
            [
                Output("pca-2d-plot", "figure"),
                Output("pca-3d-plot", "figure"),
                Output("heatmap-plot", "figure")
            ],
            [Input("venue-selector", "value")]
        )
        def update_plots(selected_venues):
            logging.info("Updating plots.")
            filtered_df = self.df[self.df["venue_numeric"].isin(selected_venues)] if selected_venues else self.df
            fig_2d = px.scatter(
                filtered_df, x="pca_x_2d", y="pca_y_2d", color="venue_numeric",
                title="PCA 2D Visualization", hover_data=["venue_numeric"], template="plotly_white"
            )
            fig_3d = px.scatter_3d(
                filtered_df, x="pca_x_3d", y="pca_y_3d", z="pca_z_3d", color="venue_numeric",
                title="PCA 3D Visualization", hover_data=["venue_numeric"], template="plotly_white"
            )
            fig_heatmap = go.Figure(
                data=go.Histogram2dContour(
                    x=filtered_df["pca_x_2d"], y=filtered_df["pca_y_2d"],
                    colorscale="Viridis", contours=dict(showlabels=True)
                )
            )
            fig_heatmap.update_layout(
                title="PCA Heatmap of Abstract Embeddings",
                xaxis_title="PCA X",
                yaxis_title="PCA Y",
                template="plotly_white"
            )
            return fig_2d, fig_3d, fig_heatmap
        
        @self.app.callback(
            Output("download-html", "data"),
            [Input("download-button", "n_clicks")],
            prevent_initial_call=True
        )
        def download_html(n_clicks):
            logging.info("Download button clicked.")
            file_path = "pca_visualization.html"
            self.app.server.run("dash_app", port=8050, debug=False)
            return dcc.send_file(file_path)
    
    def run(self):
        logging.info("Running Dash app.")
        self.app.run_server(debug=True)

