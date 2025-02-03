import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import torch
from datetime import datetime, timedelta
import wandb
from pathlib import Path
import json
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import HTML, display

class NeuroFluxVisualizer:
    """
    Visualization tools implementation from Section 5.3
    Provides interactive dashboards and analysis tools
    """
    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for metrics
        self.metrics_cache = {}
        
        # Style configuration
        self.colors = px.colors.qualitative.Set3
        self.theme = 'plotly_white'
        
    def create_training_dashboard(
        self,
        metrics: Dict[str, List[float]],
        step_size: int = 100
    ) -> go.Figure:
        """Create interactive training progress dashboard"""
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Loss Curve', 'Learning Rate',
                'Throughput', 'Memory Usage',
                'GPU Utilization', 'Temperature'
            ),
            vertical_spacing=0.12
        )
        
        # Add loss curve
        steps = list(range(0, len(metrics['loss']), step_size))
        smoothed_loss = self._smooth_curve(metrics['loss'])
        
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=smoothed_loss[::step_size],
                name='Loss',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Add learning rate
        if 'learning_rate' in metrics:
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=metrics['learning_rate'][::step_size],
                    name='LR',
                    line=dict(color='green')
                ),
                row=1, col=2
            )
        
        # Add throughput
        if 'throughput' in metrics:
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=metrics['throughput'][::step_size],
                    name='Throughput',
                    line=dict(color='purple')
                ),
                row=2, col=1
            )
        
        # Add memory usage
        if 'gpu_memory' in metrics:
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=metrics['gpu_memory'][::step_size],
                    name='GPU Memory',
                    line=dict(color='orange')
                ),
                row=2, col=2
            )
        
        # Add GPU utilization
        if 'gpu_util' in metrics:
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=metrics['gpu_util'][::step_size],
                    name='GPU Util',
                    line=dict(color='red')
                ),
                row=3, col=1
            )
        
        # Add temperature
        if 'gpu_temp' in metrics:
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=metrics['gpu_temp'][::step_size],
                    name='GPU Temp',
                    line=dict(color='brown')
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=900,
            width=1200,
            showlegend=True,
            title_text="Training Progress Dashboard",
            template=self.theme
        )
        
        # Save dashboard
        fig.write_html(self.save_dir / "training_dashboard.html")
        return fig
    
    def create_resource_heatmap(
        self,
        monitor_data: Dict[str, List[float]],
        interval_minutes: int = 5
    ) -> go.Figure:
        """Create resource utilization heatmap"""
        # Prepare data
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=24),
            end=datetime.now(),
            freq=f'{interval_minutes}min'
        )
        
        metrics = ['gpu_util', 'gpu_memory', 'cpu_util', 'memory_util']
        data = np.zeros((len(metrics), len(timestamps)))
        
        for i, metric in enumerate(metrics):
            if metric in monitor_data:
                # Resample data to match timestamps
                values = monitor_data[metric]
                data[i, :] = np.interp(
                    np.linspace(0, 1, len(timestamps)),
                    np.linspace(0, 1, len(values)),
                    values
                )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=timestamps,
            y=metrics,
            colorscale='Viridis',
            zmin=0,
            zmax=1
        ))
        
        fig.update_layout(
            title='Resource Utilization Heatmap',
            xaxis_title='Time',
            yaxis_title='Metric',
            height=400
        )
        
        # Save heatmap
        fig.write_html(self.save_dir / "resource_heatmap.html")
        return fig
    
    def create_performance_analysis(
        self,
        training_metrics: Dict[str, List[float]],
        baseline_metrics: Optional[Dict[str, float]] = None
    ) -> go.Figure:
        """Create performance analysis visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Performance Distribution',
                'Correlation Matrix',
                'Throughput vs. Batch Size',
                'Memory vs. Sequence Length'
            )
        )
        
        # Performance distribution
        for i, (metric, values) in enumerate(training_metrics.items()):
            fig.add_trace(
                go.Histogram(
                    x=values,
                    name=metric,
                    nbinsx=30,
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # Correlation matrix
        df = pd.DataFrame(training_metrics)
        corr = df.corr()
        
        fig.add_trace(
            go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='RdBu'
            ),
            row=1, col=2
        )
        
        # Throughput vs. Batch Size
        if all(k in training_metrics for k in ['throughput', 'batch_size']):
            fig.add_trace(
                go.Scatter(
                    x=training_metrics['batch_size'],
                    y=training_metrics['throughput'],
                    mode='markers',
                    name='Throughput'
                ),
                row=2, col=1
            )
        
        # Memory vs. Sequence Length
        if all(k in training_metrics for k in ['gpu_memory', 'seq_length']):
            fig.add_trace(
                go.Scatter(
                    x=training_metrics['seq_length'],
                    y=training_metrics['gpu_memory'],
                    mode='markers',
                    name='Memory'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            width=1200,
            showlegend=True,
            title_text="Performance Analysis"
        )
        
        # Save analysis
        fig.write_html(self.save_dir / "performance_analysis.html")
        return fig
    
    def create_interactive_report(
        self,
        training_data: Dict,
        monitor_data: Dict,
        save_path: Optional[str] = None
    ) -> HTML:
        """Create interactive HTML report"""
        # Generate all visualizations
        dashboard = self.create_training_dashboard(training_data)
        heatmap = self.create_resource_heatmap(monitor_data)
        analysis = self.create_performance_analysis(training_data)
        
        # Create HTML report
        html_content = f"""
        <html>
        <head>
            <title>NeuroFlux Training Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .container {{ max-width: 1200px; margin: auto; }}
                .section {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>NeuroFlux Training Report</h1>
                <div class="section">
                    <h2>Training Progress</h2>
                    {dashboard.to_html(full_html=False)}
                </div>
                <div class="section">
                    <h2>Resource Utilization</h2>
                    {heatmap.to_html(full_html=False)}
                </div>
                <div class="section">
                    <h2>Performance Analysis</h2>
                    {analysis.to_html(full_html=False)}
                </div>
            </div>
        </body>
        </html>
        """
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(html_content)
        
        return HTML(html_content)
    
    def _smooth_curve(
        self,
        values: List[float],
        weight: float = 0.6
    ) -> List[float]:
        """Apply exponential smoothing to curve"""
        smoothed = []
        last = values[0]
        for point in values:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    
    def export_wandb_artifacts(
        self,
        run_id: Optional[str] = None
    ):
        """Export visualizations to W&B"""
        if wandb.run is None and run_id is None:
            raise ValueError("No active W&B run found")
            
        # Log all HTML artifacts
        for html_file in self.save_dir.glob("*.html"):
            artifact = wandb.Artifact(
                name=f"visualization_{html_file.stem}",
                type="visualization"
            )
            artifact.add_file(str(html_file))
            wandb.log_artifact(artifact) 