import pandas as pd
from elasticsearch import Elasticsearch, helpers
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import json

# =====================
# 1. ETL Y LIMPIEZA DE DATOS BASE
# =====================
df = pd.read_csv("C:/Users/Luis Diaz/OneDrive/Documentos/Escritorio/proyectoe/messi_con_lat_long_F.csv")

# Quitar filas sin lat/lon ni minuto
df = df.dropna(subset=['Latitud', 'Longitud', 'Minute'])

# Limpieza general
df.drop(columns=['Unnamed: 13', 'Season', 'Result', 'At_score'], inplace=True)
df['Matchday'] = df['Matchday'].apply(lambda x: "partido de liga" if str(x).isdigit() else x)
df['Club'] = df['Club'].replace({
    "PSG": "Paris Saint-Germain",
    "Paris Saint Germain": "Paris Saint-Germain"
})
df = df[~df['Latitud'].astype(str).str.contains(r"\?", na=False)]
df = df[~df['Longitud'].astype(str).str.contains(r"\?", na=False)]

# Campo geo_point
df['location'] = df.apply(lambda row: {'lat': float(row['Latitud']), 'lon': float(row['Longitud'])}, axis=1)
# =====================
# 2. GUARDAR JSON PARA KIBANA (GEO)
# =====================
geo_json = "messi_geo.json"
df.to_json(geo_json, orient='records', lines=True, force_ascii=False)

# =====================
# 3. CREAR MODELO DE CLASIFICACIÓN
# =====================

df_clas = df[['Club', 'Opponent', 'Venue', 'Type', 'Minute','location']].dropna()
df_clas['Minute'] = pd.to_numeric(df_clas['Minute'], errors='coerce')
df_clas = df_clas.dropna(subset=['Minute'])
df_clas['gol_segundo_tiempo'] = df_clas['Minute'].apply(lambda x: 1 if x > 45 else 0)

df_train, df_test = train_test_split(df_clas, test_size=0.3, random_state=42)

X_train = pd.get_dummies(df_train[['Club', 'Opponent', 'Venue', 'Type']], drop_first=True)
X_test = pd.get_dummies(df_test[['Club', 'Opponent', 'Venue', 'Type']], drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

y_train = df_train['gol_segundo_tiempo']
y_test = df_test['gol_segundo_tiempo']

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Crear un DataFrame final con las columnas categóricas y location
df_test_resultado = df_test[['Club', 'Opponent', 'Venue', 'Type', 'Minute', 'location']].copy()
df_test_resultado['gol_segundo_tiempo'] = y_test.values
df_test_resultado['prediccion_segundo_tiempo'] = y_pred

clas_csv = "messi_clasificacion_resultado.csv"
df_test_resultado.to_csv(clas_csv, index=False)

# =====================
# 4. CARGAR A ELASTICSEARCH
# =====================
es = Elasticsearch("http://localhost:9200", verify_certs=False)

# --- Índice messi_geo
index_geo = "messi_geo"
if es.indices.exists(index=index_geo):
    es.indices.delete(index=index_geo)
es.indices.create(index=index_geo, body={"mappings": {"properties": {"location": {"type": "geo_point"}}}})

with open(geo_json, "r", encoding="utf-8") as file:
    data_geo = [json.loads(line) for line in file if line.strip()]
actions_geo = [{"_index": index_geo, "_source": doc} for doc in data_geo]

try:
    helpers.bulk(es, actions_geo)
    print(f"✅ Datos geográficos cargados en índice '{index_geo}'")
except helpers.BulkIndexError as e:
    print("❌ Errores al cargar geo:")
    for err in e.errors: print(err)

# --- Índice messi_clasificacion
index_clas = "messi_clasificacion"
if es.indices.exists(index=index_clas):
    es.indices.delete(index=index_clas)
es.indices.create(index=index_clas, body={
    "mappings": {
        "properties": {
            "location": {"type": "geo_point"}
        }
    }
})

df_test_resultado.reset_index(drop=True, inplace=True)
data_clas = df_test_resultado.to_dict(orient='records')
actions_clas = [{"_index": index_clas, "_source": row} for row in data_clas]

try:
    helpers.bulk(es, actions_clas)
    print(f"✅ Datos clasificados cargados en índice '{index_clas}'")
except helpers.BulkIndexError as e:
    print("❌ Errores al cargar clasificación:")
    for err in e.errors: print(err)
print("Ejemplo de documento para messi_clasificacion:")
print(data_clas[0])
