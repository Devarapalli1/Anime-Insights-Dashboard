import plotly.express as px20
from collections import Counter
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tabulate import tabulate

#----------------Rading and printing first 10 rows of both datasets------------
anime_df = pd.read_excel("datasets/cleaned_anime_dataset.xlsx")
rating_df = pd.read_excel("datasets/cleaned_rating_dataset.xlsx")

# First 10 rows of anime and rating dataset
print("\nFirst 10 rows of the anime data:\n",tabulate(anime_df.head(10), headers = 'keys', tablefmt='psql'))
print("\nFirst 10 rows of the rating data:\n",tabulate(rating_df.head(10), headers = 'keys', tablefmt='psql'))


#-------------Both Dataset information-----------------------------
def info_table(df):
    return pd.DataFrame({
        'Column': df.columns,
        'Non-Null Count': df.notnull().sum(),
        'Null Count': df.isnull().sum(),
        'Dtype': df.dtypes.astype(str)
    }).reset_index(drop=True)


print("Anime Dataset Info:")
print(info_table(anime_df).to_string(index=False))

print("\nRating Dataset Info:")
print(info_table(rating_df).to_string(index=False))

print("\n\n")

print(f"Shape of The Rating Dataset : {rating_df.shape}")
print(f"Shape of The Anime Dataset : {anime_df.shape}")


#------------------Summary of the each dataset------------------------
print("Summary of The anime Dataset :")
summary = anime_df.describe().T
print(tabulate(summary, headers='keys', tablefmt='psql'))


print("\n\nSummary of The rating Dataset :")
summary1 = rating_df.describe().T
print(tabulate(summary1, headers='keys', tablefmt='psql'))


#-------------------Merging two datasets into one----------------------

# Merging the datasets
fulldata = pd.merge(anime_df, rating_df, on="anime_id", suffixes=[None, "_user"])
fulldata = fulldata.rename(columns={"rating_user": "user_rating"})

# Shape of the merged dataset
print(f"Shape of The Merged Dataset : {fulldata.shape}")

# Printing first 10 rows of the merged dataset.
print("\nGlimpse of The Merged Dataset :")
print(tabulate(fulldata.head(10), headers="keys", tablefmt="grid"))




#--------------------Visualisations---------------------------------


#----------####**1.)  Top 15 anime names based on members count**------------------

top_members_df = anime_df.sort_values(by="members", ascending=False).head(15)
fig = px20.bar(top_members_df,x="name",y="members",title="Top 15 Most Watched Anime by Members",labels={"name": "Anime Name", "members": "Member Count"},)
fig.update_layout(xaxis_tickangle=-30)
fig.write_html("topanime_members.html",include_plotlyjs='cdn', full_html=False)



#----------####**2.)  Top 15 anime names based on rating**-----------------------

top_rated_df = anime_df.sort_values(by="rating", ascending=False).head(15)
fig1 = px20.bar(
    top_rated_df,
    x="rating",
    y="name",
    orientation="h",
    title="Top 15 Rated Anime",
    labels={"rating": "Average Rating", "name": "Anime Name"},
    hover_data=["name", "rating"],
    color="rating", 
    color_continuous_scale=px20.colors.sequential.Tealgrn
)
fig1.update_yaxes(categoryorder="total ascending")
fig1.write_html("top_rated.html")


#----------####**3.)  Distribution of each anime type**----------------

type_counts = anime_df["type"].value_counts().reset_index()
type_counts.columns = ["type", "count"]
fig2 = px20.bar(type_counts,x="type",y="count",color="type",color_discrete_sequence=px20.colors.qualitative.Plotly,
labels={"type": "Anime Type", "count": "Count"},hover_data=["type", "count"]
)
fig2.update_layout(plot_bgcolor="white",paper_bgcolor="white",font_color="black",title_font_size=20,
                  legend=dict(
                        title='Anime Type',
                        x=1,  
                        y=0.5,  
                        traceorder='normal',
                        orientation='v',
                        bgcolor='rgba(255, 255, 255, 0)', 
                        borderwidth=0
                    )
                  )
fig2.write_html("animetypedist.html",include_plotlyjs='cdn', full_html=False)

#-----------####**4.)  Distribution of each anime type Ratings**------------

fig3 = px20.box(anime_df,x="type",y="rating",
              title="Distribution of Anime Ratings by Type",
              labels={"type": "Anime Type", "rating": "Average Rating"})
fig3.write_html("animetyperatingsdist.html",include_plotlyjs='cdn', full_html=False)


#--------------####**5.)  Genre Frequency Treemap**----------------------------------------

genre_list = anime_df['genre'].dropna().str.split(', ')
all_genres = [genre for sublist in genre_list for genre in sublist]
genre_counts = Counter(all_genres)
genre_data = pd.DataFrame(genre_counts.items(), columns=["genre", "count"])

fig4 = go.Figure(go.Treemap(
    labels=genre_data['genre'],
    parents=[""] * len(genre_data),
    values=genre_data['count'],
    textinfo="label+value"  
))

fig4.update_layout(title="Genre Frequency Treemap")
fig4.write_html("Genrefrequency.html",include_plotlyjs='cdn', full_html=False)


#-------------####**6.)  Interactive Scatter Plot: Rating vs. Popularity**-------------
fig5 = px20.scatter(
    anime_df, x='rating', y='members',
    color='type', hover_data=['name', 'genre'],
    title='Anime Rating vs. Popularity'
)
fig5.write_html("Scatter.html",include_plotlyjs='cdn', full_html=False)



#---------------####**7.)  User Ratings Distribution Across Top 10 Anime Titles**-----------------

filtered_df = fulldata[fulldata['user_rating'] != -1]
top_n = 10
top_titles = (filtered_df['name'].value_counts().head(top_n).index)
filtered_top = filtered_df[filtered_df['name'].isin(top_titles)]

fig6 = px20.histogram(
    filtered_top,
    x='user_rating',
    facet_col='name',
    facet_col_wrap=3,
    category_orders={'name': list(top_titles)},
    title='User Ratings Distribution Across Top 10 Anime Titles',
    labels={'user_rating': 'User Rating', 'name': 'Anime Title'},
    color_discrete_sequence=['#636EFA'],
    nbins=10
)

fig6.update_layout(
    xaxis=dict(dtick=1),
    yaxis_title='Number of Users',
    bargap=0.2,
    height=600
)
fig6.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig6.write_html("Userratingdist.html",include_plotlyjs='cdn', full_html=False)


#-----------------####**8.)  Category wise Anime Ratings Distribution**-----------------

palette = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff']  # Define your color palette
categories = ['TV', 'OVA', 'Movie', 'Special', 'ONA', 'Music']
palette_indices = [0, 1, 2, 3, 4, 5] 
edge_color = '#000000'
fulldata = pd.merge(anime_df, rating_df, on="anime_id", suffixes=[None, "_user"])
fulldata = fulldata.rename(columns={"rating_user": "user_rating"})
data_dict = {cat: fulldata[fulldata["type"] == cat] for cat in categories}
fig7 = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        "Anime's Average Ratings Distribution",
        "Users Anime Ratings Distribution"
    ]
)

for idx, (category, color_idx) in enumerate(zip(categories, palette_indices)):
    visible = idx == 0 
    fig7.add_trace(
        go.Histogram(
            x=data_dict[category]["rating"],
            nbinsx=20,
            marker=dict(color=palette[color_idx], line=dict(color=edge_color, width=1)),
            name=f"{category} Anime Ratings",
            visible=visible
        ),
        row=1, col=1
    )
    
    fig7.add_trace(
        go.Histogram(
            x=data_dict[category]["user_rating"],
            nbinsx=20,
            marker=dict(color=palette[color_idx], line=dict(color=edge_color, width=1)),
            name=f"{category} User Ratings",
            visible=visible
        ),
        row=1, col=2
    )

# Create dropdown buttons
buttons = []
for idx, category in enumerate(categories):
    visibility = [False] * (len(categories)*2)
    visibility[idx*2] = True  # Anime ratings trace
    visibility[idx*2 + 1] = True  # User ratings trace
    
    buttons.append({
        "label": category,
        "method": "update",
        "args": [
            {"visible": visibility},
            {"title": f"Category: {category}"}
        ]
    })

# Update layout with dropdown
fig7.update_layout(
    updatemenus=[{
        "buttons": buttons,
        "direction": "down",
        "showactive": True,
        "x": 0.5,
        "xanchor": "left",
        "y": 1.2,
        "yanchor": "top"
    }],
    title="Anime Ratings Distribution by Category",
    width=1200,
    height=500,
    template="simple_white"
)

# Set axis labels
fig7.update_xaxes(title_text="Rating", row=1, col=1)
fig7.update_yaxes(title_text="Total", row=1, col=1)
fig7.update_xaxes(title_text="Rating", row=1, col=2)
fig7.update_yaxes(title_text="Total", row=1, col=2)
fig7.write_html("catwiseanimeratingdist.html", include_plotlyjs='cdn', full_html=False)

#-----------------####**9.)  Anime Watch Count by Type (TV, Movie, OVA, etc.) **-----------------

type_counts = fulldata.groupby('type')['user_id'].count().reset_index()
type_counts.columns = ['type', 'watch_count']
fig8 = px20.pie(
    type_counts,
    names='type',
    values='watch_count',
    title='Anime Watch Count by Type',
    color='type',
    color_discrete_sequence=px20.colors.qualitative.Set3
)
fig.update_layout(
    legend=dict(
        title='Anime Type',
        x=0.5, 
        y=0.5,  
        traceorder='normal',
        orientation='v',
        bgcolor='rgba(255, 255, 255, 0)',
        borderwidth=0
    )
)
fig8.write_html("watchcountbycategory.html", include_plotlyjs='cdn', full_html=False)



#---------- K means--------------------------

fulldata['genre'] = fulldata['genre'].fillna('')
genre_split = fulldata['genre'].apply(lambda x: x.split(', '))

mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(genre_split)
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)


fulldata['episodes'] = pd.to_numeric(fulldata['episodes'], errors='coerce')
episodes_median = fulldata['episodes'].median()
fulldata['episodes'] = fulldata['episodes'].fillna(episodes_median)
features = pd.concat([genre_df, fulldata[['rating', 'episodes', 'members']]], axis=1)

# Normalize
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 2: KMeans clustering
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
fulldata['cluster'] = kmeans.fit_predict(features_scaled)

# Step 3: PCA for 2D reduction
pca = PCA(n_components=2)
components = pca.fit_transform(features_scaled)
fulldata['pca1'] = components[:, 0]
fulldata['pca2'] = components[:, 1]

# Step 4: Interactive Plotly visualization
fig9 = px20.scatter(
    fulldata,
    x='pca1',
    y='pca2',
    color=fulldata['cluster'].astype(str),
    hover_data=['name', 'genre', 'rating', 'episodes', 'members'],
    title='Anime Clusters (via KMeans + PCA)',
    labels={'cluster': 'Cluster'}
)
fig9.write_html("anime_clusters_user_rating_plotly.html",include_plotlyjs='cdn', full_html=False)

