import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file generation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)


def _ensure_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Defensively add temporal columns if they are absent."""
    df = df.copy()
    if "DateTime" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["DateTime"]):
        df["DateTime"] = pd.to_datetime(df["DateTime"])
    if "Hour" not in df.columns and "DateTime" in df.columns:
        df["Hour"] = df["DateTime"].dt.hour
    if "DayOfWeek" not in df.columns and "DateTime" in df.columns:
        df["DayOfWeek"] = df["DateTime"].dt.dayofweek
    if "IsWeekend" not in df.columns and "DayOfWeek" in df.columns:
        df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    return df

def setup_reports_directory():
    """Ensure that the directory for figures exists."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(base_dir, "reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir

def plot_hourly_traffic(df: pd.DataFrame, save: bool = False) -> go.Figure:
    """Plot average traffic volume by hour for each junction."""
    df = _ensure_temporal(df)
    # Compute aggregate
    hourly_avg = df.groupby(["Junction", "Hour"])["Vehicles"].mean().reset_index()
    hourly_avg["Junction"] = hourly_avg["Junction"].astype(str)
    
    fig = px.line(
        hourly_avg,
        x="Hour",
        y="Vehicles",
        color="Junction",
        title="Average Hourly Traffic Volume by Junction",
        labels={"Vehicles": "Average Vehicles", "Hour": "Hour of Day (0-23)"},
        markers=True,
        template="plotly_white"
    )
    fig.update_layout(hovermode="x unified", title_x=0.5)
    
    if save:
        fig_dir = setup_reports_directory()
        # Use matplotlib to save static png
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="Hour", y="Vehicles", hue="Junction", palette="tab10", marker="o")
        plt.title("Average Hourly Traffic Volume by Junction")
        plt.xlabel("Hour of Day")
        plt.ylabel("Average Vehicles")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "hourly_traffic.png"), dpi=150)
        plt.close()
        
    return fig

def plot_daily_traffic(df: pd.DataFrame, save: bool = False) -> go.Figure:
    """Plot average traffic volume by day of the week for each junction."""
    df = _ensure_temporal(df)
    # Day Names
    day_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    df_temp = df.copy()
    df_temp["DayName"] = df_temp["DayOfWeek"].map(day_map)
    
    daily_avg = df_temp.groupby(["Junction", "DayOfWeek", "DayName"])["Vehicles"].mean().reset_index()
    daily_avg.sort_values(by="DayOfWeek", inplace=True)
    daily_avg["Junction"] = daily_avg["Junction"].astype(str)
    
    fig = px.bar(
        daily_avg,
        x="DayName",
        y="Vehicles",
        color="Junction",
        barmode="group",
        title="Average Daily Traffic Volume by Junction",
        labels={"Vehicles": "Average Vehicles", "DayName": "Day of Week"},
        template="plotly_white"
    )
    fig.update_layout(title_x=0.5)
    
    if save:
        fig_dir = setup_reports_directory()
        plt.figure(figsize=(10, 6))
        # Ensure chronological order
        sns.barplot(data=df_temp, x="DayName", y="Vehicles", hue="Junction", palette="tab10",
                    order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        plt.title("Average Daily Traffic Volume by Junction")
        plt.xlabel("Day of Week")
        plt.ylabel("Average Vehicles")
        plt.xticks(rotation=15)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "daily_traffic.png"), dpi=150)
        plt.close()
        
    return fig

def plot_weekday_vs_weekend(df: pd.DataFrame, save: bool = False) -> go.Figure:
    """Compare weekday vs weekend traffic volumes."""
    df = _ensure_temporal(df)
    df_temp = df.copy()
    df_temp["Period"] = df_temp["IsWeekend"].map({0: "Weekday", 1: "Weekend"})
    
    avg_traffic = df_temp.groupby(["Junction", "Period"])["Vehicles"].mean().reset_index()
    avg_traffic["Junction"] = avg_traffic["Junction"].astype(str)
    
    fig = px.bar(
        avg_traffic,
        x="Junction",
        y="Vehicles",
        color="Period",
        barmode="group",
        title="Traffic Comparison: Weekday vs Weekend",
        labels={"Vehicles": "Average Vehicles", "Junction": "Junction ID"},
        color_discrete_map={"Weekday": "#3B82F6", "Weekend": "#EF4444"},
        template="plotly_white"
    )
    fig.update_layout(title_x=0.5)
    
    if save:
        fig_dir = setup_reports_directory()
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_temp, x="Junction", y="Vehicles", hue="Period", palette="Set1")
        plt.title("Traffic Comparison: Weekday vs Weekend")
        plt.xlabel("Junction")
        plt.ylabel("Average Vehicles")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "weekday_vs_weekend.png"), dpi=150)
        plt.close()
        
    return fig

def plot_traffic_heatmap(df: pd.DataFrame, save: bool = False) -> go.Figure:
    """Create a junction vs hour traffic volume heatmap."""
    df = _ensure_temporal(df)
    pivot_df = df.groupby(["Junction", "Hour"])["Vehicles"].mean().reset_index()
    pivot_df = pivot_df.pivot(index="Junction", columns="Hour", values="Vehicles")
    
    # Plotly heatmap
    fig = px.imshow(
        pivot_df,
        labels=dict(x="Hour of Day", y="Junction ID", color="Average Vehicles"),
        x=pivot_df.columns,
        y=[f"Junction {i}" for i in pivot_df.index],
        title="Congestion Heat Map (Junction vs Hour)",
        color_continuous_scale="Viridis",
        template="plotly_white"
    )
    fig.update_layout(title_x=0.5)
    
    if save:
        fig_dir = setup_reports_directory()
        plt.figure(figsize=(12, 5))
        sns.heatmap(pivot_df, cmap="viridis", annot=True, fmt=".1f", cbar_kws={'label': 'Average Vehicles'})
        plt.title("Congestion Heat Map: Junction vs Hour")
        plt.xlabel("Hour of Day")
        plt.ylabel("Junction ID")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "congestion_heatmap.png"), dpi=150)
        plt.close()
        
    return fig

def plot_correlation_matrix(df: pd.DataFrame, save: bool = False) -> go.Figure:
    """Generate and plot correlation matrix of features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Drop columns that are ids or constant
    cols_to_drop = [col for col in ["ID", "id"] if col in numeric_cols]
    corr_df = df[numeric_cols].drop(columns=cols_to_drop, errors="ignore").corr()
    
    fig = px.imshow(
        corr_df,
        labels=dict(color="Correlation"),
        x=corr_df.columns,
        y=corr_df.columns,
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu",
        zmin=-1.0,
        zmax=1.0,
        template="plotly_white"
    )
    fig.update_layout(title_x=0.5)
    
    if save:
        fig_dir = setup_reports_directory()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1.0, vmax=1.0, square=True)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "correlation_matrix.png"), dpi=150)
        plt.close()
        
    return fig

def plot_actual_vs_predicted(actual: np.ndarray, predicted: np.ndarray, model_name: str, save: bool = False) -> go.Figure:
    """Plot actual vs predicted values."""
    eval_df = pd.DataFrame({"Actual": actual, "Predicted": predicted})
    # Sample down to 1000 points if huge for cleaner rendering
    if len(eval_df) > 1000:
        eval_df_sample = eval_df.sample(n=1000, random_state=42).sort_index()
    else:
        eval_df_sample = eval_df
        
    fig = px.scatter(
        eval_df_sample,
        x="Actual",
        y="Predicted",
        opacity=0.6,
        trendline="ols",
        trendline_color_override="red",
        title=f"Actual vs Predicted Traffic Volume - {model_name}",
        labels={"Actual": "Actual Vehicles", "Predicted": "Predicted Vehicles"},
        template="plotly_white"
    )
    # Add 45 degree baseline reference line
    min_val = min(eval_df["Actual"].min(), eval_df["Predicted"].min())
    max_val = max(eval_df["Actual"].max(), eval_df["Predicted"].max())
    fig.add_shape(
        type="line",
        line=dict(dash="dash", color="blue", width=2),
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val
    )
    fig.update_layout(title_x=0.5)
    
    if save:
        fig_dir = setup_reports_directory()
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=eval_df_sample, x="Actual", y="Predicted", alpha=0.6)
        plt.plot([min_val, max_val], [min_val, max_val], color="blue", linestyle="--", label="Ideal Line")
        plt.title(f"Actual vs Predicted Traffic Volume: {model_name}")
        plt.xlabel("Actual Vehicles")
        plt.ylabel("Predicted Vehicles")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "actual_vs_predicted.png"), dpi=150)
        plt.close()
        
    return fig

def plot_residuals(actual: np.ndarray, predicted: np.ndarray, model_name: str, save: bool = False) -> go.Figure:
    """Plot the distribution of prediction residuals (errors)."""
    residuals = actual - predicted
    eval_df = pd.DataFrame({"Residuals": residuals})
    
    fig = px.histogram(
        eval_df,
        x="Residuals",
        nbins=50,
        marginal="box",
        title=f"Residuals (Errors) Distribution - {model_name}",
        labels={"Residuals": "Residual (Actual - Predicted)"},
        color_discrete_sequence=["#10B981"],
        template="plotly_white"
    )
    fig.update_layout(title_x=0.5)
    
    if save:
        fig_dir = setup_reports_directory()
        plt.figure(figsize=(8, 5))
        sns.histplot(residuals, bins=50, kde=True, color="teal")
        plt.axvline(0, color="red", linestyle="--", linewidth=1.5)
        plt.title(f"Residuals Distribution: {model_name}")
        plt.xlabel("Residual (Actual - Predicted)")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "residuals_distribution.png"), dpi=150)
        plt.close()
        
    return fig

def plot_feature_importance(feature_names: list, importances: np.ndarray, model_name: str, save: bool = False) -> go.Figure:
    """Plot feature importances."""
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_df = feat_df.sort_values(by="Importance", ascending=True)
    
    fig = px.bar(
        feat_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Global Feature Importance - {model_name}",
        labels={"Importance": "Relative Importance", "Feature": "Model Feature"},
        template="plotly_white",
        color="Importance",
        color_continuous_scale="Blues"
    )
    fig.update_layout(title_x=0.5, coloraxis_showscale=False)
    
    if save:
        fig_dir = setup_reports_directory()
        plt.figure(figsize=(10, 6))
        # Re-sort descending for matplotlib barplot
        feat_df_desc = feat_df.sort_values(by="Importance", ascending=False)
        sns.barplot(data=feat_df_desc, x="Importance", y="Feature",
                    hue="Feature", palette="Blues_r", legend=False)
        plt.title(f"Feature Importance: {model_name}")
        plt.xlabel("Relative Importance")
        plt.ylabel("Feature Name")
        plt.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "feature_importance.png"), dpi=150)
        plt.close()
        
    return fig
