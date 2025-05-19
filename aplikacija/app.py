import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import hashlib
import os

# ===========================
# Naloži podatke
# ===========================
@st.cache_data
def load_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))  # lokacija app.py
    movies_path = os.path.join(dir_path, "ml-latest-small", "movies.csv")
    ratings_path = os.path.join(dir_path, "ml-latest-small", "ratings.csv")

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    rating_stats = ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index()

    merged = pd.merge(movies, rating_stats, on="movieId")
    merged["year"] = merged["title"].str.extract(r'\((\d{4})\)').astype("Int64")
    merged["clean_title"] = merged["title"].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
    return merged, ratings

df, ratings = load_data()

# ===========================
# Inicializacija datotek
# ===========================
USERS_FILE = "users.csv"
USER_RATINGS_FILE = "user_ratings.csv"

if not os.path.exists(USERS_FILE):
    pd.DataFrame(columns=["username", "password"]).to_csv(USERS_FILE, index=False)
if not os.path.exists(USER_RATINGS_FILE):
    pd.DataFrame(columns=["username", "movieId", "rating"]).to_csv(USER_RATINGS_FILE, index=False)

# ===========================
# Funkcije za uporabnike
# ===========================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    users = pd.read_csv(USERS_FILE)
    if username in users["username"].values or not username or not password:
        return False
    new_user = pd.DataFrame({"username": [username], "password": [hash_password(password)]})
    new_user.to_csv(USERS_FILE, mode='a', header=False, index=False)
    return True

def login_user(username, password):
    users = pd.read_csv(USERS_FILE)
    hashed = hash_password(password)
    return ((users["username"] == username) & (users["password"] == hashed)).any()

# ===========================
# Inicializacija session_state
# ===========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ===========================
# Sidebar: Prijava / Registracija
# ===========================
st.sidebar.subheader("🔐 Prijava / Registracija")
auth_mode = st.sidebar.radio("Izberi možnost", ["Prijava", "Registracija"])
username = st.sidebar.text_input("Uporabniško ime", key="username_input")
password = st.sidebar.text_input("Geslo", type="password", key="password_input")

if st.sidebar.button("Potrdi"):
    if auth_mode == "Registracija":
        if register_user(username, password):
            st.sidebar.success("Registracija uspešna. Prijavi se.")
        else:
            st.sidebar.error("Uporabniško ime že obstaja ali manjkajo podatki.")
    else:
        if login_user(username, password):
            st.sidebar.success("Prijava uspešna.")
            st.session_state.logged_in = True
            st.session_state.username = username
        else:
            st.sidebar.error("Napačno uporabniško ime ali geslo.")

# Gumb za odjavo, če je prijavljen
if st.session_state.logged_in:
    if st.sidebar.button("Odjava"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.experimental_rerun()

# ===========================
# Naslov aplikacije
# ===========================
st.title("🎬 Analiza najboljših filmov (MovieLens)")

# ===========================
# Sidebar filtri
# ===========================
st.sidebar.header("🎛️ Filtri")
min_ratings = st.sidebar.slider("Minimalno število ocen", min_value=1, max_value=1000, value=50, step=10)
all_genres = sorted({genre for sublist in df["genres"].str.split('|') for genre in sublist})
selected_genre = st.sidebar.selectbox("Izberi žanr", ["Vsi"] + all_genres)
years = sorted(df["year"].dropna().unique())
selected_year = st.sidebar.selectbox("Izberi leto", ["Vsa leta"] + list(years))

# ===========================
# Filtriranje
# ===========================
filtered_df = df[df["rating_count"] >= min_ratings]
if selected_genre != "Vsi":
    filtered_df = filtered_df[filtered_df["genres"].str.contains(selected_genre)]
if selected_year != "Vsa leta":
    filtered_df = filtered_df[filtered_df["year"] == selected_year]

# ===========================
# Top 10 filmov
# ===========================
st.subheader("🔝 Top 10 filmov glede na povprečno oceno")
top10 = filtered_df.sort_values(by="avg_rating", ascending=False).head(10)
st.write(top10[["clean_title", "avg_rating", "rating_count", "genres", "year"]].rename(columns={
    "clean_title": "Naslov",
    "avg_rating": "Povprečna ocena",
    "rating_count": "Št. ocen",
    "genres": "Žanri",
    "year": "Leto"
}))

# ===========================
# Primerjava dveh filmov
# ===========================
st.subheader("🎭 Primerjava dveh filmov")
film_options = df["clean_title"].sort_values().unique()
col1, col2 = st.columns(2)
with col1:
    film1 = st.selectbox("Izberi prvi film", film_options, key="film1")
with col2:
    film2 = st.selectbox("Izberi drugi film", film_options, key="film2")

id1 = df[df["clean_title"] == film1]["movieId"].values[0]
id2 = df[df["clean_title"] == film2]["movieId"].values[0]
r1 = ratings[ratings["movieId"] == id1]["rating"]
r2 = ratings[ratings["movieId"] == id2]["rating"]

st.write("### 📈 Statistika ocen")
stats_df = pd.DataFrame({
    "Film": [film1, film2],
    "Povprečna ocena": [r1.mean(), r2.mean()],
    "Št. ocen": [r1.count(), r2.count()],
    "Std. odklon": [r1.std(), r2.std()]
})
st.write(stats_df)

st.write("### 🧮 Histogram ocen")
fig, ax = plt.subplots()
ax.hist(r1, bins=5, alpha=0.5, label=film1)
ax.hist(r2, bins=5, alpha=0.5, label=film2)
ax.set_xlabel("Ocena")
ax.set_ylabel("Število ocen")
ax.legend()
st.pyplot(fig)

st.write("### 🕒 Povprečna letna ocena")
ratings["year"] = pd.to_datetime(ratings["timestamp"], unit='s').dt.year
avg_yearly_1 = ratings[ratings["movieId"] == id1].groupby("year")["rating"].mean()
avg_yearly_2 = ratings[ratings["movieId"] == id2].groupby("year")["rating"].mean()
fig2, ax2 = plt.subplots()
avg_yearly_1.plot(label=film1, ax=ax2)
avg_yearly_2.plot(label=film2, ax=ax2)
ax2.set_ylabel("Povprečna ocena")
ax2.set_xlabel("Leto")
ax2.legend()
st.pyplot(fig2)

st.write("### 📊 Število ocen na leto")
count_yearly_1 = ratings[ratings["movieId"] == id1].groupby("year")["rating"].count()
count_yearly_2 = ratings[ratings["movieId"] == id2].groupby("year")["rating"].count()
fig3, ax3 = plt.subplots()
count_yearly_1.plot(label=film1, ax=ax3)
count_yearly_2.plot(label=film2, ax=ax3)
ax3.set_ylabel("Število ocen")
ax3.set_xlabel("Leto")
ax3.legend()
st.pyplot(fig3)

# ===========================
# Ocenjevanje in priporočila (samo za prijavljene)
# ===========================
if st.session_state.logged_in:
    st.subheader(f"🎟️ Ocenjevanje filmov - {st.session_state.username}")

    # Naloži ocene uporabnikov
    user_ratings = pd.read_csv(USER_RATINGS_FILE, dtype={"movieId": int})
    rated_movie_ids = user_ratings[user_ratings["username"] == st.session_state.username]["movieId"].astype(int).values
    unrated = df[~df["movieId"].isin(rated_movie_ids)]

    # Koliko filmov je uporabnik že ocenil
    num_rated = len(rated_movie_ids)
    min_ratings_required = 10
    num_left = max(0, min_ratings_required - num_rated)

    if num_left > 0:
        st.info(f"Za priporočila morate oceniti še vsaj {num_left} filmov.")

        # Uporabnik izbere film za ocenjevanje
        unrated_titles = unrated["clean_title"] + " (" + unrated["year"].astype(str) + ")"
        film_selection = st.selectbox("Izberi film za ocenjevanje", unrated_titles if not unrated.empty else ["Ni več neocenjenih filmov"])

        if not unrated.empty:
            selected_row = unrated[unrated_titles == film_selection].iloc[0]
            st.markdown(f"**{selected_row['clean_title']} ({selected_row['year']})**  \nŽanri: {selected_row['genres']}")
            rating_key = f"rating_{selected_row['movieId']}"
            submit_key = f"submit_{selected_row['movieId']}"

            if rating_key not in st.session_state:
                st.session_state[rating_key] = 2.5

            rating = st.slider("Oceni film", min_value=0.0, max_value=5.0, step=0.5, key=rating_key, label_visibility="visible", disabled=False)

            if st.button("Oddaj oceno", key=submit_key):
                new_row = pd.DataFrame({
                    "username": [st.session_state.username],
                    "movieId": [selected_row["movieId"]],
                    "rating": [rating]
                })
                user_ratings = pd.concat([user_ratings, new_row], ignore_index=True)
                user_ratings.to_csv(USER_RATINGS_FILE, index=False)
                st.success(f"Ocena {rating} oddana za film {selected_row['clean_title']}")
                st.rerun()
    else:
        # Priporočila uporabniške kolaborativne filtracije
        st.subheader("⭐ Priporočila za vas (na osnovi ocen drugih uporabnikov)")

        # Priprava matrike uporabnikov x filmov
        all_ratings = pd.concat([
            ratings[["userId", "movieId", "rating"]],
            user_ratings.rename(columns={"username": "userId"})[["userId", "movieId", "rating"]]
        ])
        user_movie_matrix = all_ratings.pivot_table(index="userId", columns="movieId", values="rating")

        if st.session_state.username in user_movie_matrix.index:
            user_corr = user_movie_matrix.T.corrwith(user_movie_matrix.loc[st.session_state.username], method='pearson').dropna()
            user_corr = user_corr.drop(labels=[st.session_state.username], errors='ignore')
            user_corr = user_corr[user_corr > 0]

            if not user_corr.empty:
                similar_users = user_corr.sort_values(ascending=False).head(5).index

                # Zberi ocene podobnih uporabnikov
                sim_users_ratings = user_movie_matrix.loc[similar_users]

                # Povprečna ocena za filme, ki jih še niste ocenjevali
                unrated_for_user = user_movie_matrix.columns.difference(user_movie_matrix.loc[st.session_state.username].dropna().index)
                pred_ratings = sim_users_ratings[unrated_for_user].mean(axis=0).dropna()
                pred_ratings = pred_ratings.sort_values(ascending=False).head(10)

                if not pred_ratings.empty:
                    st.write("### Priporočeni filmi:")
                    recs = df[df["movieId"].isin(pred_ratings.index)][["clean_title", "genres", "year"]].copy()
                    recs["Predvidena ocena"] = pred_ratings.values
                    recs = recs.sort_values(by="Predvidena ocena", ascending=False)
                    st.dataframe(recs.rename(columns={"clean_title": "Naslov", "genres": "Žanri", "year": "Leto"}))
                else:
                    st.info("Ni dovolj podatkov za priporočila.")
            else:
                st.info("Ni dovolj podobnih uporabnikov za priporočila.")
        else:
            st.info("Za priporočila najprej ocenite nekaj filmov.")
