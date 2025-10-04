import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

gold = yf.download("GC=F", start="1980-01-01", end="2025-09-01", auto_adjust=False)
gold = gold.reset_index()

if 'index' in gold.columns:
    gold = gold.rename(columns={'index': 'Date'})

gold.head()

gold['Daily Change'] = gold['Close'].diff()
gold['Daily Change %'] = gold['Close'].pct_change() * 100
gold[['Date', 'Close', 'Daily Change', 'Daily Change %']].head()

gold['Shock_2w'] = gold['Close'].shift(-10) - gold['Close']
gold['Shock_2w %'] = (gold['Close'].shift(-10) - gold['Close']) / gold['Close'] * 100
gold[['Date', 'Close', 'Daily Change', 'Daily Change %', 'Shock_2w', 'Shock_2w %']].tail(10)

gold['Volatility_30d'] = gold['Daily Change %'].rolling(window=30).std()
gold[['Date', 'Daily Change %', 'Volatility_30d']].tail(10)

gold['Seuil_variable'] = 2 * gold['Volatility_30d']
gold['Target_Shock'] = (gold['Shock_2w %'].abs() >= gold['Seuil_variable']).astype(int)
gold[['Date', 'Close', 'Shock_2w %', 'Seuil_variable', 'Target_Shock']].tail(10)

gold['MeanGrowth_30d'] = gold['Daily Change %'].abs().rolling(window=30).mean()
gold[['Date', 'Daily Change %', 'MeanGrowth_30d']].tail(10)

gold['Seuil_growth'] = 2 * gold['MeanGrowth_30d']
gold['Target_Shock_growth'] = (gold['Shock_2w %'].abs() >= gold['Seuil_growth']).astype(int)
gold[['Date', 'Close', 'Shock_2w %', 'Seuil_growth', 'Target_Shock_growth']].tail(50)

gold['Bollinger_MA20'] = gold['Close'].rolling(window=20).mean()
gold['Bollinger_STD20'] = gold['Close'].rolling(window=20).std()
gold['Bollinger_Upper'] = gold['Bollinger_MA20'] + 2 * gold['Bollinger_STD20']
gold['Bollinger_Lower'] = gold['Bollinger_MA20'] - 2 * gold['Bollinger_STD20']
gold[['Date', 'Close', 'Bollinger_MA20', 'Bollinger_Upper', 'Bollinger_Lower']].tail(10)

plt.figure(figsize=(14, 7))
plt.plot(gold['Close'], label='Prix de clôture')
plt.plot(gold['Bollinger_MA20'], label='Moyenne mobile 20j', color='orange')
plt.plot(gold['Bollinger_Upper'], label='Bollinger Upper', linestyle='--', color='green')
plt.plot(gold['Bollinger_Lower'], label='Bollinger Lower', linestyle='--', color='red')
plt.title("Prix de l'or et bandes de Bollinger")
plt.xlabel("Date")
plt.ylabel("Prix")
plt.legend()
plt.tight_layout()
plt.show()

# Calcul du RSI sur 14 jours
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

gold['RSI_14'] = compute_rsi(gold['Close'], window=14)
gold[['Date', 'Close', 'RSI_14']].tail(10)

plt.figure(figsize=(14, 4))
plt.plot(gold['RSI_14'], label='RSI 14 jours', color='purple')
plt.axhline(70, color='red', linestyle='--', label='Suracheté (70)')
plt.axhline(30, color='green', linestyle='--', label='Survendu (30)')
plt.title("RSI 14 jours du prix de l'or")
plt.xlabel("Date")
plt.ylabel("RSI")
plt.legend()
plt.tight_layout()
plt.show()

from gdeltdoc import GdeltDoc, Filters, near, repeat
import pandas as pd

def get_multiple_batches_gold(num_batches):
    gd = GdeltDoc()
    all_articles = []
    filters_used = []

    from datetime import datetime, timedelta
    import time

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 9, 20)
    total_days = (end_date - start_date).days
    days_per_batch = total_days // num_batches

    current_date = start_date

    for i in range(num_batches):
        if i == num_batches - 1:
            period_end = end_date
        else:
            period_end = current_date + timedelta(days=days_per_batch)

        f = Filters(
            start_date=current_date.strftime("%Y-%m-%d"),
            end_date=period_end.strftime("%Y-%m-%d"),
            num_records=250,
            language="english",
            keyword=["Gold price", "Gold investment", "Gold market"]
        )
        filters_used.append(f)

        try:
            df_batch = gd.article_search(f)

            if not df_batch.empty:
                all_articles.append(df_batch)
                print(f"Batch {i+1} ({current_date.strftime('%Y-%m-%d')} à {period_end.strftime('%Y-%m-%d')}): {len(df_batch)} articles")
        except Exception as e:
            print(f"Erreur batch {i+1}: {e}")

        current_date = period_end
        time.sleep(1)

    if all_articles:
        final_df = pd.concat(all_articles, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['url'], keep='first')
        print(f"Total final après suppression doublons: {len(final_df)} articles")
        return gd, filters_used, final_df

    return None, [], pd.DataFrame()


gd, filters_list, df = get_multiple_batches_gold(10)

# Exemple : prendre le filtre global (ou le dernier)
f = filters_list [-1]

articles = gd.article_search(f)
timeline = gd.timeline_search("timelinevol", f)



from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# On suppose que le texte des articles est dans la colonne 'title'
df['sentiment_score'] = df['title'].astype(str).apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Afficher les 10 premiers articles avec leur score de sentiment
df[['title', 'url', 'sentiment_score']].head(10)

mean_sentiment = df['sentiment_score'].mean()
print(f"Sentiment moyen des articles : {mean_sentiment:.2f}")


# Convertir la colonne 'seendate' en format datetime si ce n'est pas déjà fait
df['seendate'] = pd.to_datetime(df['seendate'], errors='coerce')

# Grouper par jour et calculer la moyenne du score de sentiment
daily_sentiment = df.groupby(df['seendate'].dt.date)['sentiment_score'].mean()

# Afficher les 10 premiers jours avec leur score moyen
print(daily_sentiment.head(10))

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
daily_sentiment.plot()
plt.title("Sentiment moyen des articles sur l'or par jour")
plt.xlabel("Date")
plt.ylabel("Score de sentiment moyen")
plt.tight_layout()
plt.show()


df = gold[[ 'Date', 'Open', 'Close', 'Volatility_30d', 'Daily Change', 'Daily Change %', 'Shock_2w', 'Shock_2w %', 'Seuil_variable', 'Target_Shock','Seuil_growth', 'Target_Shock_growth', 'RSI_14','Bollinger_MA20', 'Bollinger_Upper', 'Bollinger_Lower']]
df


# --- Analyse PCA et classification K-means sur les variables quantitatives du gold ---

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns

# Sélection des variables quantitatives pertinentes
quant_vars = [
    'Close', 'Volatility_30d', 'Daily Change', 'Daily Change %',
    'Shock_2w', 'Shock_2w %', 'Seuil_variable', 'Target_Shock',
    'Seuil_growth', 'Target_Shock_growth', 'RSI_14',
    'Bollinger_MA20', 'Bollinger_Upper', 'Bollinger_Lower'
]

# Nettoyage des données (suppression des lignes avec valeurs manquantes)
X = gold[quant_vars].dropna()

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA (2 composantes principales)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Ajout des composantes au DataFrame
gold.loc[X.index, 'PCA1'] = X_pca[:, 0]
gold.loc[X.index, 'PCA2'] = X_pca[:, 1]

# Classification K-means (3 clusters, adapte si besoin)
kmeans = KMeans(n_clusters=3, random_state=42)
gold.loc[X.index, 'cluster'] = kmeans.fit_predict(X_pca)

# Visualisation des clusters sur les composantes principales
plt.figure(figsize=(8,6))
sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data=gold.loc[X.index], palette='Set1')
plt.title("K-means sur les composantes PCA du gold")
plt.tight_layout()
plt.show()



from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

Z = linkage(X_pca, method='ward')
gold.loc[X.index, 'hclust'] = fcluster(Z, t=3, criterion='maxclust')

plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Dendrogramme Hclust")
plt.show()





import mariadb
import pandas as pd
import sqlalchemy

import os

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "gold_db")
}


# ...existing code...
try:
    conn = mariadb.connect(**DB_CONFIG)
    print("Connexion réussie à MariaDB.")
except mariadb.Error as e:
    print(f"Erreur lors de la connexion à MariaDB: {e}")
# ...existing code...

from sqlalchemy import create_engine

# Connexion : user=root, mdp=root, base=panel_db
db_host = os.getenv("DB_HOST", "localhost")
db_user = os.getenv("DB_USER", "root")
db_password = os.getenv("DB_PASSWORD", "")
db_name = os.getenv("DB_NAME", "gold_db")

engine = create_engine(f"mariadb+mariadbconnector://{db_user}:{db_password}@{db_host}/{db_name}")



# Supprimer les lignes contenant au moins un NaN
df_clean = df.dropna()

# Arrondir toutes les colonnes numériques à 2 décimales
numeric_columns = df_clean.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
df_clean = df_clean.copy()
df_clean[numeric_columns] = df_clean[numeric_columns].round(2)

# Insère dans la table 'gold_events'
try:
    df_clean.to_sql('gold_events', con=engine, if_exists='replace', index=False)
    print(f"{len(df_clean)} lignes insérées dans la table gold_events.")
except Exception as e:
    print("Erreur lors de l'insertion :", e)
# ...existing code...