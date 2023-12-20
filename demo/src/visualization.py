from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.embed import components


def plot(df):
    source = ColumnDataSource(
        data=dict(
            x=df["1"],
            y=df["2"],
            imgs=df["image_url"],  # URLs of the images
            # isDrug=tsne_df['isDrug_str'],
            color=df["color"],
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

    # Show the plot
    script, div = components(p)
    return script, div
