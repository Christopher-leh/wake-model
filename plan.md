# Research Log

## ⚡ KALTSTART (Nächster mechanischer Schritt < 15 Min)
<!-- 
PRINZIP: Die Feierabend-Regel gegen die Morgen-Paralyse.
Du darfst den PC abends oder nach einer Session nur zuklappen, wenn hier eine einzige, rein mechanische Start-Aufgabe für das nächste Mal steht. 
Keine Denkaufgabe ("Über Loss nachdenken"), sondern eine konkrete Handlung ("Datei X öffnen, Funktion Y mit Dummy-Tensor aufrufen").
-->
- [ ] **Nächster Schritt:** 

---

## 🎯 AKTUELLES ÜBUNGSBLATT: Blatt #[Nr.]
**Zeitraum:** [Datum bis Datum] | **Timebox:** max. [X] Stunden

### 1. Die Kernfrage (Das "Übungsblatt")
<!-- 
PRINZIP: Künstliche Zielfunktion statt offenem Suchen.
Formuliere 1–2 präzise Fragen, die am Ende des Zyklus mit Ja, Nein, einem Plot oder einer konkreten Erkenntnis beantwortet werden können. Trenne strikt zwischen Code-Experiment und Literatur-Recherche.
-->
* 

### 2. Ignorier-Liste (Künstliche Randbedingungen)
<!-- 
PRINZIP: Verkleinerung des Suchraums.
Schreibe explizit auf, was in diesem Zyklus VERBOTEN ist (z. B. kein Hyperparameter-Tuning, keine neue Architektur bauen, maximal X Paper herunterladen). Alles, was hier steht, darfst du diese Woche ohne schlechtes Gewissen ignorieren.
-->
* 

### 3. Definition of Done (Abbruchkriterium)
<!-- 
PRINZIP: Bewertung nach Input/Erkenntnis statt nach Modell-Perfektion.
Wann gilt das Blatt als abgegeben? Definiere greifbare Artefakte (z. B. "Plot ist gespeichert", "3 Stichpunkte pro Paper sind notiert"), völlig unabhängig davon, ob das Modell schon gut konvergiert oder das Experiment fehlschlägt.
-->
- [ ] 

---

## 📝 KURZ-NOTIZEN & ERGEBNISSE (Aktuelles Blatt)
<!-- 
PRINZIP: Sofortige Dokumentation während der Arbeit.
Keine Romane. Nur knappe Beobachtungen aus den Experimenten (Was ist kaputtgegangen? Wo liegt der Flaschenhals?) oder die Kern-Takeaways aus gelesenen Papern (Problem / Methodischer Trick / Nutzen für mich).
-->
* 

---

## 🅿️ IDEEN-PARKPLATZ (Unter der Woche NICHT anfassen!)
<!-- 
PRINZIP: Schutz vor dem Rabbit-Hole-Drift.
Jede neue Idee, jeder Zweifel ("Müsste ich nicht eigentlich Methode Z nutzen?") und jedes spannende Paper am Wegesrand wird hier sofort als Einzeiler geparkt und ignoriert. Dieser Bereich darf erst am Ende des Zyklus beim Erstellen des nächsten Übungsblatts gelesen werden.
-->
* 

---

## 🗄️ ARCHIV (Abgegebene Blätter)
<!-- 
PRINZIP: Sichtbarer Fortschritt gegen das Gefühl des Stillstands.
Am Ende der Timebox wird das "Aktuelle Übungsblatt" samt Notizen unverändert hier nach unten verschoben – egal ob die Hypothese geklappt hat oder widerlegt wurde. Ein abgeschlossenes Negativ-Ergebnis ist ein erfolgreich gelöstes Übungsblatt.
-->




R

Das Repository in [`wake-model`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model) ist jetzt sauber aufgeräumt und der erste Commit steht (`1b37b0c`):

### Was erledigt wurde
1. **Ordnerstruktur eingerichtet**:
   - Alle bisherigen `.png`- und `.html`-Plots liegen jetzt gesammelt in `wake-model/outputs/`.
   - Dein Notebook liegt jetzt unter [`notebooks/data_test.ipynb`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model/notebooks/data_test.ipynb) (die 9 MB an eingebetteten Plot-Rohdaten im Notebook wurden geleert, sodass die Datei im Git nur 6,6 KB groß ist – deine gespeicherten Plots liegen ja weiterhin in `outputs/`).
2. **[`.gitignore`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model/.gitignore) erweitert**:
   - Ignoriert jetzt `outputs/`, `*.png`, `*.html`, `*.pt`/`*.pth`, `data/` und `checkpoints/`.
3. **Erster sauberer Git-Commit**:
   - `git status` ist komplett sauber (`working tree clean`).

---

### Dein Fahrplan zum Selber-Coden (2D-Wake-Schnitt)

Damit du die Best Practices jetzt direkt beim Coden anwenden kannst, hier der empfohlene Ablauf in 4 kleinen Schritten:

#### Schritt 1: Übungsblatt #1 in [`plan.md`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model/plan.md) ausfüllen & Branch starten
Trage in [`plan.md`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model/plan.md) kurz deine Kernfrage (z. B. *„Kann mein MLP aus [`models.py`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model/src/models.py) den 2D-Schnitt auf Nabenhöhe `z_index = 18` von `meanvelocity` und `meanpressure` rekonstruieren?“*) und dein Abbruchkriterium ein.
Optional im Terminal (`cd wake-model`):
```powershell
git checkout -b exp/2d-wake-slice
```

#### Schritt 2: Daten-Loader in `src/data.py` anlegen
Nimm deinen HDF5-Code aus [`notebooks/data_test.ipynb`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model/notebooks/data_test.ipynb) und baue daraus eine saubere Funktion in einer neuen Datei `src/data.py`, z. B.:
- Lädt `meanvelocity.h5` `(25, 30, 301, 3)` und `meanpressure.h5` `(25, 30, 301, 1)` sowie `xcoor`, `ycoor`, `zcoor` `(301, 30, 25)`.
- **Wichtiger Stolperstein bei deinen HDF5-Daten**: In deiner letzten Zelle von [`data_test.ipynb`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model/notebooks/data_test.ipynb) hast du schon notiert, dass die Felder die Form `(Nz, Ny, Nx, C)` = `(25, 30, 301, ...)` haben, während `xcoor, ycoor, zcoor` die Form `(301, 30, 25)` haben! Achte beim Schneiden bei `z_index = 18` darauf, dass Koordinaten und Feldwerte dieselbe Achsenreihenfolge haben, bevor du `.ravel()` aufrufst.
- **Koordinaten-Skalierung**: Deine $x, y$-Koordinaten sind in Metern (großer Wertebereich). Ein MLP mit `Tanh()` trainiert deutlich besser, wenn die Eingabekoordinaten grob auf $[-1, 1]$ oder $[0, 1]$ normiert sind.

👉 *Danach erster kleiner Commit:* `git add src/data.py && git commit -m "feat: add 2D wake slice loader from h5"`

#### Schritt 3: Problem-Klasse in [`src/problems.py`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model/src/problems.py)
- Da du mit [`BaseProblem`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model/src/problems.py#L6-L14) schon das passende Interface hast, kannst du entweder in [`NS2DPINN`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model/src/problems.py#L17-L77) `pde_weight=0.0` für eine reine Data-Driven-Baseline setzen (oder den PDE-Teil bei `self.pde_weight == 0` überspringen, um dir die teuren Autograd-Ableitungen zu sparen) oder eine schlanke `Wake2DDataProblem(BaseProblem)`-Klasse schreiben.

#### Schritt 4: [`train.py`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model/train.py) & [`src/visualizer.py`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model/src/visualizer.py)
- In [`visualizer.py`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model/src/visualizer.py) kannst du die Standard-Speicherpfade auf `outputs/loss_curve.png` usw. setzen (`set_aspect("equal")` bietet sich beim langgestreckten Wake-Gebiet in [`plot_comparison_and_residuals`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model/src/visualizer.py#L23-L71) auch an).

---

Leg gern einfach los! Wenn du deinen Entwurf in `src/data.py` oder [`train.py`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model/train.py) geschrieben hast, sag Bescheid – dann mache ich dir gern ein **Code-Review** (auf Stil, Shapes, PyTorch-Fallstricke und Struktur) oder helfe dir beim Debuggen.


Das ist eine super wichtige Frage – gerade im Machine Learning werden hier oft zwei Begriffe vermischt.

---

### 1. Was sind die „Checkpoints“? (Wichtige Unterscheidung!)

In deinem Projekt begegnet dir das Wort an zwei völlig verschiedenen Stellen:

1. **Der Ordner `checkpoints/` (`best_model.pt`) $\rightarrow$ PyTorch (NICHT Git!)**
   - Wenn dein Modell in [`trainer.py`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model/src/trainer.py#L69-L80) für 3000 Epochen trainiert, verändern sich Millionen Kommazahlen (die Gewichte des Netzes).
   - Sobald das Modell auf den Validierungsdaten einen neuen Bestwert erreicht, speichert PyTorch diese Gewichte in `checkpoints/best_model.pt`.
   - Falls das Training in Epoche 2500 plötzlich schlechter wird oder abstürzt, lädst du am Ende mit `trainer.load_checkpoint("best_model.pt")` einfach den besten Stand des Netzes wieder.
   - **Wichtig:** Diese `.pt`-Dateien sind groß und ändern sich bei jedem Trainingslauf. Deshalb stehen sie in der [`.gitignore`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model/.gitignore) und werden **niemals** zu Git/GitHub hochgeladen.

2. **Git-Commits $\rightarrow$ Speicherstände für deinen Code**
   - Git kümmert sich ausschließlich um deinen **geschriebenen Code** (`.py`, `.md`, `.ipynb`).
   - Denk an einen Git-Commit wie an **Schnellspeichern (Quicksave) in einem Videospiel**.

---

### 2. Warum „Abends einmal alles hochladen“ dir das Leben schwer macht

Wenn du Git nur abends wie Google Drive als Backup nutzt, verlierst du den größten Vorteil von Git:
- *Szenario:* Um 11:00 Uhr funktioniert dein neuer HDF5-Data-Loader perfekt. Um 14:00 Uhr baust du die Loss-Funktion um und änderst dabei auch den Data-Loader. Um 16:00 Uhr wirft alles nur noch Dimensions-Fehler und du weißt nicht mehr, wie der funktionierende Stand von 11:00 Uhr aussah.
- Hättest du um 11:00 Uhr einen Commit gemacht, könntest du mit einem Klick exakt zu diesem funktionierenden Stand zurückspringen.

Deshalb trennt man zwei Dinge:
1. **Commit (Lokal speichern):** Machst du **mehrmals am Tag** (immer wenn ein kleiner Teilabschnitt klappt). Das passiert nur auf deiner Festplatte und dauert 2 Sekunden.
2. **Push / Sync (Auf GitHub hochladen):** Machst du **am Ende des Tages** (oder nach einem fertigen Block), damit dein Stand online gesichert ist.

---

### 3. Wann committe ich was? (Die 4 Faustregeln)

Du kannst weiterhin bequem die **Git-Seitenleiste in VS Code** (das verzweigte Icon links, `Strg+Shift+G`) nutzen!

#### Regel 1: „Ein Gedanke = Ein Commit“
Immer wenn du sagen kannst: *„So, dieser kleine Baustein klappt jetzt“*, machst du einen Commit.
Typische Zeitpunkte an einem normalen Tag:
1. Du hast `src/data.py` geschrieben und kurz getestet, dass die Tensoren die richtige Shape haben $\rightarrow$ **Commit!**
2. Du hast in [`problems.py`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model/src/problems.py) das Problem für den 2D-Schnitt ergänzt $\rightarrow$ **Commit!**
3. Du hast in [`train.py`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model/train.py) einen Bug bei der Koordinaten-Normierung gefixt $\rightarrow$ **Commit!**
4. Du trägst abends deinen „Kaltstart“-Schritt für morgen in [`plan.md`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model/plan.md) ein $\rightarrow$ **Commit & Push auf GitHub!**

#### Regel 2: Vor jedem riskanten Umbau committen
Bevor du sagst: *„Ich probiere jetzt mal aus, ob ich die ganze Architektur auf Fourier-Features umbaue“*, machst du vorher einen Commit vom funktionierenden Stand.
- Wenn das Experiment komplett schiefgeht, klickst du in VS Code bei der Datei auf den kleinen Rückgängig-Pfeil (*„Discard Changes“*) und bist sofort wieder beim funktionierenden Stand.

#### Regel 3: Kurz reingucken vor dem Klick auf „Commit“
In VS Code siehst du unter **Changes** alle veränderten Dateien:
- Klick einmal kurz auf die Datei: VS Code zeigt dir links (rot) den alten Code und rechts (grün) deinen neuen Code.
- So siehst du sofort, ob du aus Versehen noch ein `print(" HALLO TEST 123 ")` dringelassen hast.
- Klicke auf das `+` neben den Dateien, die zusammengehören (*Stage Changes*), tippe oben eine kurze Nachricht ein und klicke auf **Commit**.

#### Regel 4: Kurze, sprechende Commit-Nachrichten
Statt `"update"` oder `"feierabend"` schreibst du eine Zeile, was passiert ist. Ein kurzes Wort am Anfang hilft beim Wiederfinden:
- `feat: load 2D wake slice at z=18` *(neues Feature / neuer Code)*
- `fix: transpose x and y coordinates to match h5 shape` *(Fehler behoben)*
- `exp: test hidden_dim 128 on wake slice` *(Experiment / Hyperparameter)*
- `docs: fill out sheet 1 in plan.md` *(Notizen / Doku)*

---

### 4. Wenn du diesen Ordner wieder mit GitHub verbinden willst

Da dein [`wake-model`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model)-Ordner aktuell ein rein lokales Git-Repository ist (ohne verknüpftes GitHub-Repo), kannst du in der VS Code Git-Seitenleiste jederzeit auf **„Publish Branch“** klicken:
- VS Code fragt dich dann, ob es ein **privates** GitHub-Repository namens `wake-model` für dich anlegen soll.
- Danach erscheint an derselben Stelle der bekannte **„Sync Changes / Push“**-Button für den Feierabend.

Ah, **Checkouts** (`git checkout`)! Jetzt ergibt die Frage noch viel mehr Sinn.

Ein **Checkout** bedeutet in Git einfach: **„Wechsle meine aktuelle Ansicht (die Dateien im Ordner) auf einen anderen Stand oder eine andere Zeitlinie (Branch).“**

---

### Die Idee dahinter: Hauptstraße vs. Teststrecke (Branches)

Stell dir dein Projekt wie eine Straße vor:
- **`master` (deine Hauptstraße):** Hier liegt immer nur Code, der **funktioniert** und sauber durchläuft (so wie dein Kovasznay-Stand jetzt gerade).
- **Ein neuer Branch (deine Teststrecke):** Jetzt willst du für Übungsblatt #1 alles auf den 2D-Wake-Schnitt aus den HDF5-Daten umbauen. Dabei wirst du [`train.py`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model/train.py) und [`problems.py`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model/src/problems.py) stark verändern.

Statt direkt auf der Hauptstraße (`master`) herumzuschrauben, machst du einen **Checkout auf einen neuen Branch**:
```powershell
git checkout -b exp/2d-wake
```
*(Das `-b` heißt: „Erstelle einen neuen **B**ranch namens `exp/2d-wake` und wechsle direkt dorthin“).*

---

### Was bringt dir das konkret im Alltag?

#### 1. Du kannst jederzeit zwischen Ständen hin- und herspringen
Angenommen, du bist mitten im Umbau auf `exp/2d-wake`, nichts läuft gerade, und dein Betreuer fragt spontan: *„Kannst du mir nochmal kurz den Plot vom Kovasznay-Flow zeigen?“*
- Du speicherst (committest) deinen Zwischenstand auf `exp/2d-wake`.
- Du machst **Checkout zurück auf `master`**:
  ```powershell
  git checkout master
  ```
- **Magie:** Im selben Moment sehen alle deine Dateien im Ordner wieder exakt so aus wie beim funktionierenden Kovasznay-Stand!
- Danach wechselst du mit `git checkout exp/2d-wake` wieder zurück und arbeitest nahtlos an deinem Wake-Modell weiter.

#### 2. Sackgassen tun nicht mehr weh
Angenommen, du probierst 2 Tage lang eine komplizierte neue Loss-Funktion auf einem Branch `exp/komischer-loss` aus und merkst: *„Bringt überhaupt nichts, macht alles schlechter.“*
- Früher musstest du mühsam in 5 Dateien per Hand `Strg+Z` drücken oder Code auskommentieren.
- Mit Branches machst du einfach `git checkout master` – und lässt den fehlgeschlagenen Branch links liegen (oder löschst ihn). Deine Hauptstraße ist komplett sauber geblieben.

#### 3. Wenn das Experiment klappt $\rightarrow$ Zusammenführen (`merge`)
Wenn dein 2D-Wake-Modell auf dem Branch `exp/2d-wake` super läuft und das Übungsblatt fertig ist, holst du es auf die Hauptstraße:
```powershell
git checkout master          # 1. Zurück auf die Hauptstraße wechseln
git merge exp/2d-wake        # 2. Den Code von der Teststrecke übernehmen
```

---

### So machst du Checkouts in VS Code (ganz ohne Terminal)

Du musst dir die Befehle gar nicht zwingend merken, in VS Code geht das mit zwei Klicks:

1. **Wo sehe ich, wo ich bin?**
   Schau ganz **unten links** in die Leiste von VS Code: Dort steht neben einem kleinen Zweig-Symbol aktuell **`master`**.
2. **Neuen Branch starten (`checkout -b`):**
   Klicke unten links auf `master` $\rightarrow$ wähle oben im Menü **„+ Create new branch...“** $\rightarrow$ tippe z. B. `exp/2d-wake` ein $\rightarrow$ Enter. Unten links steht jetzt `exp/2d-wake`.
3. **Zwischen Branches wechseln (`checkout`):**
   Klicke wieder unten links auf den Branch-Namen und wähle in der Liste einfach `master` oder `exp/2d-wake` aus. *(Wichtig: Vor dem Wechseln immer kurz deine offenen Änderungen committen!)*
4. **Am Ende zusammenführen (`merge`):**
   Wechsle unten links auf `master` $\rightarrow$ klicke in der Git-Seitenleiste oben auf die drei Punkte `...` $\rightarrow$ **Branch** $\rightarrow$ **Merge Branch...** $\rightarrow$ wähle `exp/2d-wake`.

---

**Kurzfassung für deinen Workflow:**
- **Neues Übungsblatt in [`plan.md`](file:///c:/Users/Christopher/sync/Masterarbeit/wake-model/plan.md) starten** $\rightarrow$ Unten links neuen Branch auschecken (`exp/blatt-1`).
- **Unterwegs kleine Etappenziele erreicht** $\rightarrow$ `Commit` in der Seitenleiste.
- **Abends sichern** $\rightarrow$ `Push / Sync` auf GitHub.
- **Übungsblatt fertig** $\rightarrow$ Zurück auf `master` wechseln und `Merge`.