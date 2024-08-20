# AutoScout24 Autopreisanalyse und Vorhersage

Dieses Projekt analysiert Autopreise in Deutschland mithilfe eines [AutoScout24-Datensatzes](https://www.kaggle.com/datasets/ander289386/cars-germany) von Kaggle. Das Projekt besteht aus zwei Hauptteilen:

## 1. Tableau Dashboards
- **Allgemeine Trends Dashboard**: Visualisiert allgemeine Trends im Automarkt, wie Durchschnittspreise nach Automarke, Modell und Jahr.
- **Spezifische Trends Dashboard**: Fokussiert auf spezifische Faktoren, die die Autopreise beeinflussen, wie Kilometerstand, Kraftstoffart und Motorgröße.

**Hinweis**: Um die Dashboards in der Datei `Dashboard_Tableau.twb` nutzen zu können, musst du die Datei zunächst mit der Datei `autoscout_bereinigt.csv` verknüpfen. Gehe dazu wie folgt vor:

1. **Öffne Tableau Desktop** und lade die Datei `Dashboard_Tableau.twb`.
2. **Verknüpfe die Datenquelle**:
   - Gehe zu **Daten** im Menü.
   - Wähle **Datenquelle ändern** oder **Verbindung zu Datenquelle herstellen**.
   - Wähle die Datei `autoscout_bereinigt.csv` aus deinem Projektordner und stelle die Verbindung her.
3. **Aktualisiere die Verbindungen** und überprüfe, ob die Daten korrekt geladen wurden.

## 2. Streamlit App
- **Autopreis-Vorhersage**: Eine Streamlit-Anwendung, die es Nutzern ermöglicht, spezifische Daten zu einem Auto einzugeben (z.B. Marke, Modell, Baujahr, Kilometerstand, etc.). Die App verwendet einen trainierten Random Forest Regressor, um den Preis des Autos vorherzusagen.

---

## Installation und Nutzung

1. **Tableau Dashboards**: Die Tableau-Dashboards können direkt in Tableau geöffnet und untersucht werden, nachdem du die oben beschriebenen Schritte zur Verknüpfung der Datenquelle durchgeführt hast.

2. **Streamlit App**:
   - Installiere die benötigten Python-Abhängigkeiten mit `pip install -r requirements.txt`.
   - Starte die Streamlit App mit dem Befehl `streamlit run Predicter_App.py`.

---

## Datenquelle

Die Daten stammen aus dem [AutoScout24-Datensatz](https://www.kaggle.com/datasets/ander289386/cars-germany) auf Kaggle.
