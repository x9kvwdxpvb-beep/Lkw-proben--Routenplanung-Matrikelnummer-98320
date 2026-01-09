# LKW-Proben-Routenplanung (Neckar-Odenwald-Kreis)

Dieses Python-Projekt berechnet optimale Rundtouren für bis zu fünf Baustellen
im Neckar-Odenwald-Kreis.

## Funktionen
- Fester Start- und Endpunkt (Depot)
- Interaktive Eingabe der Baustellenadressen
- Zwei Optimierungsziele:
  - schnellste Route (Fahrzeit + Probenzeit)
  - kürzeste Route (Kilometer)
- Ausgabe als:
  - Textreport
  - Interaktive HTML-Karte

## Voraussetzungen
- Python 3.10 oder neuer

## Installation
```bash
git clone <REPOSITORY-URL>
cd lkw-routing
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Ausführung

Nach der Installation kann das Programm wie folgt gestartet werden:

```bash
python3 lkw_routing.py
Nach der Berechnung werden die Ergebnisse im Projektverzeichnis gespeichert:

- **out_report.txt**  
  → Textdatei mit der berechneten optimalen Route (Reihenfolge, Zeiten, Distanzen)

- **out_map.html**  
  → Interaktive Karte  
  Diese Datei kann mit einem Webbrowser geöffnet werden (z. B. Doppelklick oder Rechtsklick → „Im Browser öffnen“).
