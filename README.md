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

