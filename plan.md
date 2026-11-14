 Timebox ist je ein bis zwei Wochen, damit wärst du etwa im Februar bei einer belastbaren Forschungsfrage.

| Blatt | Kernfrage | Code | Lesen |
|---|---|---|---|
| 1 | Kann das MLP den Schnitt rein aus Daten lernen? | Daten-Baseline, Entdimensionalisierung | Zhang & Zhao 2021 (2D) |
| 2 | Passen die JHTDB-Daten, und wie falsch ist 2D? | Ausschnitt laden, virtuelle Lidar, Divergenz-Check | JHTDB-wind-Paper |
| 3 | Rekonstruiert Lidar + NS das Feld vor einer Turbine? | PINN mit $(x, y, t)$, erste Ablation | Tian et al. 2024 |
| 4 | Was bringt der Turbinen-Term im Nachlauf? | Actuator-Disk-Kraft, volle Ablation | Digital Twin 2.4, Wang et al. 2024 |
| 5 | Lohnt sich 3D? | kleines 3D-PINN auf demselben Ausschnitt | Zhang & Zhao 2021 (3D), Chen et al. 2025 |

## Blatt 1: Daten-Baseline (aktueller Code)
- **Code:** `pde_weight = 0`, einige tausend Epochen, Outputs und Koordinaten mit $D$ und $U_\infty$ entdimensionalisieren.
- **Done:** 3×3-Plot und Rel. L2 für zwei oder drei Werte von `n_train`, drei Stichpunkte in `plan.md`.
- **Verboten:** PDE-Term anfassen, Architektur tunen.

## Blatt 2: JHTDB-Daten und virtuelle Lidar
- **Code:** Einen kleinen Ausschnitt laden und lokal speichern: ein bis zwei Turbinen, Nabenhöhenebene plus ein paar Ebenen darüber und darunter, wenige Minuten. Daraus Lidar-Messungen nach Gl. 4 des Papers erzeugen.
- **Divergenz-Check:** $u_x + v_y$ direkt aus den Daten per finiten Differenzen berechnen und mit $w_z$ vergleichen. Das braucht kein Netz und zeigt dir früh, wie stark die 2D-Annahme verletzt ist, getrennt für Anströmung und Nachlauf.
- **Done:** gespeicherter Ausschnitt, Plot der Lidar-Punkte über dem wahren Feld, Plot der Divergenz.

## Blatt 3: 2D-PINN vor der Turbine
- **Code:** Das einfachste Setting aus dem 2021er-Paper: Zeit als dritter Input, Lidar-Loss plus NS-Residuum, noch keine Turbinenkraft.
- **Ablation:** Derselbe Lauf nur mit dem Lidar-Loss. Das ist genau die Baseline, die im Digital-Twin-Paper fehlt.
- **Done:** Fehler für $u$ und $v$ mit und ohne Physik, Loss-Kurven der einzelnen Terme.

## Blatt 4: Turbinen-Term und Nachlauf
- **Code:** Gebiet auf den Nachlauf erweitern, Kraftterm aus den Schubdaten der Datenbank bauen.
- **Ablation:** nur Lidar, Lidar + NS, Lidar + NS + Turbine. Dazu ein Lauf mit Rauschen auf den Lidar-Daten.
- **Done:** Tabelle der Fehler für alle Varianten.

## Blatt 5: 2D gegen 3D
- **Code:** Dasselbe Setup als kleines 3D-PINN, mit Lidar-Daten aus einem horizontalen und einem vertikalen Schnitt.
- **Done:** Fehler auf der Nabenhöhenebene im Vergleich zu Blatt 4, dazu die Trainingszeit.

## Parallel: Operatoren
Lies ohne Zeitdruck etwa ein Paper pro Blatt: DeepONet, physik-informiertes DeepONet, dann der Vergleich von Lu et al. 2022. Wenn du es anfassen willst, bau ein Mini-DeepONet für Burgers mit variabler Anfangsbedingung. Das Problem kennst du schon aus deiner PINN-Übung.

## Entscheidungspunkt nach Blatt 5
Dann hast du Zahlen zu den drei Kandidaten für die Forschungsfrage: Ablation und Rauschen, 2D gegen 3D, Generalisierung über Operatoren. Damit gehst du zum Betreuer und legst das Thema fest.

## Dein Test mit beiden Datensätzen
Als Test, ob der Code auf beiden läuft, ist das gut. Als Test für Generalisierung über den Yaw-Winkel taugt es kaum, weil sich die beiden Datensätze auch in Layout (6 × 10 gegen 2 × 4) und Atmosphäre (neutral gegen Tagesgang) unterscheiden. Welche Yaw-Winkel sie enthalten, habe ich nicht geprüft.

## Kaltstart für die nächste Session
In `train.py` `pde_weight` auf 0 setzen, Epochen auf 5000, starten.

Soll ich den Fahrplan als Notiz im Projekt „wake modeling“ ablegen? In deine `plan.md` kann ich gerade nicht schreiben, die Verbindung zu deinem Rechner ist im Moment weg.