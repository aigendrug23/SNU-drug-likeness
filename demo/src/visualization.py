from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.embed import components
from sklearn.manifold import TSNE
import pandas as pd


# Assume that the LAST ELEMENT of 'df' is the instance we want to infer
def infer_dimension_reduction(
    model,
    df,
    perplexity=10,
    n_iter=1000,
    metric="cosine",
    learning_rate=200,
    n_neighbors=15,
    min_dist=0.1,
):
    # Selecting columns '0' to '511'
    data = df.loc[:, "0":"511"]

    # model : TSNE or UMAP or PCA
    if model == "TSNE":
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            n_iter=n_iter,
            metric=metric,
            learning_rate=learning_rate,
        )
        results = tsne.fit_transform(data)
    elif model == "UMAP":
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            metric=metric,
            min_dist=min_dist,
            n_components=2,
            random_state=42,
        )
        results = reducer.fit_transform(data)
    elif model == "PCA":
        # Standardizing the features (important for PCA)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        # Applying PCA
        pca = PCA(
            n_components=2, random_state=42
        )  # n_components is the number of dimensions to reduce to
        results = pca.fit_transform(data_scaled)
    else:
        print("model should be TSNE or UMAP or PCA")
        return

    # Creating a new DataFrame with the reduced results
    reduced_df = pd.DataFrame(results, columns=["1", "2"])

    # Concatenating the t-SNE results with the original DataFrame's 'SMILES' column and 'isDrug' column
    reduced_df = pd.concat([reduced_df, df[["SMILES", "isDrug"]]], axis=1)
    reduced_df["isDrug_str"] = reduced_df["isDrug"].astype(str)
    reduced_df["image_url"] = [
        "images/" + str(idx) + ".png" for idx in reduced_df.index
    ]

    # change 'isDrug' column to int for color mapping
    reduced_df["isDrug"] = [
        0 if isDrug == False else 1 for isDrug in reduced_df["isDrug"]
    ]
    reduced_df["color"] = reduced_df["isDrug"].map({1: "red", 0: "blue"})
    # change the color of the last element to green
    reduced_df["color"].iloc[-1] = "lime"

    return reduced_df


def infer_plot(df):
    source = ColumnDataSource(
        data=dict(
            x=df[:-1]["1"],  # Exclude the last point for now
            y=df[:-1]["2"],
            imgs=df[:-1]["image_url"],
            color=df[:-1]["color"],
        )
    )
    last_point_source = ColumnDataSource(
        data=dict(
            x=[df.iloc[-1]["1"]],
            y=[df.iloc[-1]["2"]],
            imgs=[df.iloc[-1]["image_url"]],
            color=[df.iloc[-1]["color"]],
        )
    )

    # Create a figure
    p = figure(
        width=800,
        height=800,
        title="Visualization of Chemical Space",
        tools="pan,wheel_zoom,box_zoom,reset",
        active_scroll="wheel_zoom",
    )

    # Add a hover tool to display images
    hover = HoverTool(
        tooltips="""
        <div>
            <div>
                <img
                    src="@imgs" height="150" alt="@imgs" width="150"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>
            </div>
        </div>
        """
    )

    p.add_tools(hover)

    # Color mapping
    # color_mapper = linear_cmap(field_name='isDrug', palette=['blue', 'red'], low=0, high=1)

    # Plot the points
    p.circle("x", "y", source=source, size=7, color="color", fill_alpha=0.6)
    p.circle("x", "y", source=last_point_source, size=20, color="color", fill_alpha=1)

    # Show the plot
    script, div = components(p)
    return script, div
