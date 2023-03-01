import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import json
import requests
import time
from streamlit_lottie import st_lottie
from sklearn.neighbors import NearestNeighbors


# Définir la largeur de la page
st.set_page_config(layout="wide")

# Charger le dataset
data = pd.read_csv('data/df_OK.csv')


############################ Définition de la fonction Page d'accueil ############################
def home():
    
        # Diviser la page en deux colonnes
    col1, col2 = st.columns([3, 1]) #(colonne de droite plus étroite)

    # Première colonne titre + machine learning
    with col1:
        
        # Titre de la page
        st.title("Bienvenue sur la page d'accueil !")
        
        # transformer tous les titres en minuscule
        data['title'] = data['title'].str.lower()

        # Réduire la pondération de runtimeMinutes en divisant sa valeur par 10
        data['runtimeMinutes'] = data['runtimeMinutes']/10

        # Réduire la pondération de startYear en divisant sa valeur par 3
        data['startYear'] = data['startYear']/3

        # les caractéristiques X
        X = data[['startYear','averageRating', 'runtimeMinutes', 'genre__Action', 'genre__Adult', 'genre__Adventure',
                'genre__Animation', 'genre__Biography', 'genre__Comedy', 'genre__Crime',
                'genre__Documentary', 'genre__Drama', 'genre__Family', 'genre__Fantasy',
                'genre__Film-Noir', 'genre__History', 'genre__Horror', 'genre__Music',
                'genre__Musical', 'genre__Mystery', 'genre__News', 'genre__Reality-TV',
                'genre__Romance', 'genre__Sci-Fi', 'genre__Short', 'genre__Sport',
                'genre__Thriller', 'genre__War', 'genre__Western']]

        # initialiser le modèle
        nn = NearestNeighbors(n_neighbors=11, algorithm='auto')
        nn.fit(X)

        # demander à l'utilisateur de saisir le nom d'un film
        movie_name = st.empty().text_input("Entre le nom du film que tu as aimé :")

        # afficher la select box avec l'autocomplétion seulement si l'utilisateur a commencé à taper un nom de film
    if movie_name:
        movie_name = movie_name.lower()
        movies = data[data['title'].str.contains(movie_name)]['title'].tolist()
        selected_movie = st.selectbox("Sélectionne un film:", movies)

        # chercher le film le plus proche
        if selected_movie:
            movie_index = data[data['title'] == selected_movie.lower()].index
            if len(movie_index) == 0:
                st.write("Désolé, ce film n'a pas été trouvé dans notre base de données.")
            else:
                # ajouter une barre de progression pour montrer à l'utilisateur que nous sommes en train de chercher des recommandations
                with st.spinner('Recherche des recommandations en cours...'):
                    progress_bar = st.progress(0)
                    for percent_complete in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(percent_complete + 1)

                    distances, indices = nn.kneighbors(X.iloc[movie_index])
                    
                st.write(f"Si vous avez adoré {selected_movie}, vous devriez apprécier : ")
                for i in indices[0][1:]:
                    st.write("- " + data.iloc[i]['title'])


        # insertion du gif dans la deuxième colonne
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    lottie_hello = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_qtiaeavi.json")
    
    with col2:
        st_lottie(lottie_hello,
                speed=1,
                reverse=False,
                loop=True,
                quality="medium", # medium ; high ; low
                height=None,
                width=None,
                key=None,
                )
    
    
  
############################ Définition de la focntion KPI ############################

def kpi():
    
    st.title("KPI")
    st.text(' ')
    st.write("Ceci est la page des KPI de l'application.")

    ## Charger le dataset
    data = pd.read_csv('data/df_OK.csv')

    # Diviser la page en 2 parties égales
    col1, col2 = st.columns([2, 2])

    # Ajouter chaque KPI à une partie de la page
    
    with col1:
        
        # Code pour afficher le camembert des genres
        st.subheader('Répartition des 6 genres les plus représentés')
        
        # Top 5 des genres les plus populaires
        top_genres = data.iloc[:, 18:].sum().sort_values(ascending=False)[:6]

        # Créer un graphique en camembert avec des couleurs plus joyeuses
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#f9ca24', '#6c5ce7', '#ff7675', '#fd79a8', '#74b9ff', '#a29bfe'] # couleurs jaune, violet, rose
        ax.pie(top_genres, labels=top_genres.index, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.axis('equal')  # Pour s'assurer que le camembert est parfaitement circulaire
        st.pyplot(fig)



    with col2:
        # Code pour afficher l'évolution de la durée moyenne des films
        st.subheader('Évolution de la durée moyenne des films')

        # Retirer les films de genre 'Short'
        data = data.loc[data['genre__Short'] == 0]

        # Calculer la durée moyenne des films pour chaque année
        avg_runtime_by_year = data.groupby('startYear')['runtimeMinutes'].mean().reset_index()

        # Créer un graphique en ligne
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(avg_runtime_by_year['startYear'], avg_runtime_by_year['runtimeMinutes'], color='blue')
        ax.set_xlabel('Année')
        ax.set_ylabel('Durée moyenne des films')
        st.pyplot(fig)

    
    # Ajouter un saut de ligne pour séparer les deux blocs de KPI
    st.write('\n')

    # Diviser la page en 2 parties égales
    col3, col4 = st.columns(2)
        
    # Ajouter chaque KPI à une partie de la page
        
    with col3:
        st.subheader('Répartition des films par années de sortie')

        # Convertir les dates en années et grouper par tranche de 10 ans
        data['year'] = pd.to_datetime(data['date']).dt.year // 10 * 10
        date = pd.DataFrame(data['year'].value_counts())

        # Définir la couleur personnalisée et l'épaisseur des barres
        colors = ['#a29bfe']
        bar_width = 7

        # Créer le graphique en barres avec la couleur personnalisée et l'épaisseur des barres
        fig, ax = plt.subplots()
        ax.bar(date.index, date['year'], color=colors, width=bar_width)

        # Afficher le graphique
        st.pyplot(fig)



    with col4:
        
        st.subheader("Les 10 réalisateurs qui ont réalisé le plus de films")
        df_ok_name = pd.read_csv('data/df_OK_name-2.csv')

        # Compter le nombre de films par réalisateur
        director_count = df_ok_name['primaryName_x'].value_counts().sort_values(ascending=False)[:10]

        # Définir la palette de couleurs personnalisée
        colors = ['#f9ca24', '#6c5ce7', '#ff7675', '#fd79a8', '#74b9ff', '#a29bfe', '#fdcb6e', '#81ecec', '#d63031', '#e84393']

        # Afficher le résultat sous forme de bar chart avec la palette de couleurs personnalisée
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(x=director_count.values, y=director_count.index, ax=ax, palette=sns.color_palette(colors))
        ax.set_xlabel('Nombre de films')
        ax.set_ylabel('Réalisateur')
        ax.set_title('Top 10 des films les mieux notés par genre', color="black")

        # Afficher le graphique
        st.pyplot(fig)
    
    
    # Ajouter un saut de ligne pour séparer les deux blocs de KPI
    st.write('\n')


    # Diviser la page en 2 parties égales
    col5, col6 = st.columns(2)
    
    # Ajouter chaque KPI à une partie de la page

    with col5:
    
        # Code pour afficher le widget interactif et le tableau des 10 films les mieux notés avec filtre
        st.subheader('Top 10 des films les mieux notés')
        
    # Créer un widget interactif pour filtrer les genres
        genre_options = list(data.columns[18:])
        selected_genres = st.multiselect('Select genres', genre_options)

    # Filtrer les données en fonction des genres sélectionnés
        filtered_data = data[data[selected_genres].sum(axis=1) > 0]

    # Calculer les 10 films les mieux notés
        top_movies = filtered_data.nlargest(10, 'averageRating')

    # Afficher le tableau des 10 films les mieux notés
        st.write(top_movies[['primaryTitle', 'startYear', 'genres', 'averageRating', 'numVotes']])


    with col6:
        # Code pour afficher le tableau des 10 films les mieux notés
        st.subheader('Top 10 des films les mieux notés')
        top_10 = data[['title', 'averageRating']].sort_values('averageRating', ascending=False).head(10)
        st.table(top_10)
        




############################ Configuration de la barre latérale #############################
pages = {
    "Page d'accueil": home,
    "KPI": kpi
}

# Création de l'application Streamlit
def app():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Aller à", list(pages.keys()))

    # Exécute la fonction de la page sélectionnée
    page = pages[selection]
    page()

if __name__ == '__main__':
    app()  #permet de s'assurer que ce code est exécuté seulement si le script est exécuté directement (et non pas importé dans un autre script). C'est une bonne pratique en Python.
