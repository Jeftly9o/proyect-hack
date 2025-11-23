import pandas as pd
from transformers import pipeline
from pathlib import Path

current_dir = Path(__file__).parent
csv_path = current_dir.parent / 'database' / 'customer-Emalis_Español_50000.xlsx'

def clasificar_excel(csv_path):
    
    try:
        df = pd.read_excel(csv_path)
    except FileNotFoundError:
        print("El archivo no fue encontrado.")
        return
    df['texto_completo'] = "Asunto: " + df['asunto'].astype(str) + ". Contenido: " + df['contenido'].astype(str)

    print("Cargando modelo... esto puede tardar un poco la primera vez.")
    
    clasificador = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    etiquetas_candidatas = ["Facturación", "Soporte Técnico", "Recursos Humanos", "Spam", "Urgente"]

    print("Analizando filas...")

    def aplicar_clasificacion(texto):

        resultado = clasificador(texto, candidate_labels=etiquetas_candidatas)

        mejor_etiqueta = resultado['labels'][0]
        confianza = resultado['scores'][0]
        return pd.Series([mejor_etiqueta, confianza])


    df[['categoria_predicha', 'confianza']] = df['texto_completo'].apply(aplicar_clasificacion)


    print("\n--- Resultados (Primeras 5 filas) ---")
    print(df[['asunto', 'categoria_predicha', 'confianza']].head())


    df.to_excel("resultado_clasificado.xlsx", index=False)
    print("\nArchivo guardado como 'resultado_clasificado.xlsx'")


if __name__ == "__main__":

    clasificar_excel("mis_correos.xlsx")