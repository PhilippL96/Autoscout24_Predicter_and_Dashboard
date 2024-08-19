# AutoScout24 Autopreisanalyse und Vorhersage

Dieses Projekt analysiert Autopreise in Deutschland mithilfe eines [AutoScout24-Datensatzes](https://www.kaggle.com/datasets/ander289386/cars-germany) von Kaggle. Das Projekt besteht aus zwei Hauptteilen:

## 1. Tableau Dashboards
- **Allgemeine Trends Dashboard**: Visualisiert allgemeine Trends im Automarkt, wie Durchschnittspreise nach Automarke, Modell und Jahr.
- **Spezifische Trends Dashboard**: Fokussiert auf spezifische Faktoren, die die Autopreise beeinflussen, wie Kilometerstand, Kraftstoffart und Motorgröße.

## 2. Streamlit App
- **Autopreis-Vorhersage**: Eine Streamlit-Anwendung, die es Nutzern ermöglicht, spezifische Daten zu einem Auto einzugeben (z.B. Marke, Modell, Baujahr, Kilometerstand, etc.). Die App verwendet einen trainierten Random Forest Regressor, um den Preis des Autos vorherzusagen.

---

## Installation und Nutzung

1. **Tableau Dashboards**: Die Tableau-Dashboards können direkt in Tableau geöffnet und untersucht werden.

2. **Streamlit App**:
   - Installiere die benötigten Python-Abhängigkeiten mit `pip install -r requirements.txt`.
   - Starte die Streamlit App mit dem Befehl `streamlit run Predicter_App.py`.

---

## Datenquelle

Die Daten stammen aus dem [AutoScout24-Datensatz](https://www.kaggle.com/datasets/ander289386/cars-germany) auf Kaggle.
