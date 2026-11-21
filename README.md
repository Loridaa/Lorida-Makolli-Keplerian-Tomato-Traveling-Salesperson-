# Lorida Makolli – SpOC-4 Keplerian Tomato Traveling Salesperson


Projekt për lëndën **Dizajni dhe Analiza e Algoritmeve**.
Sfida: <https://optimise.esa.int/challenge/spoc-4-keplerian-tomato-traveling-salesperson/About>
Të dhënat zyrtare: <https://github.com/esa/SpOC4> (dosja *Challenge 2*).

**I njëjti kod** zgjidh të tre nivelet. Ndryshon vetëm skedari hyrës:

```bash
python solve.py --level easy   --time 900     # 49 domate
python solve.py --level medium --time 1800    # 181 domate
python solve.py --level hard   --time 3600    # 1051 domate
```

## Rezultatet (mission time, ditë – sa më e vogël aq më mirë)

| Niveli | Domate | Rezultati ynë | Skedari për dorëzim |
|---|---|---|---|
| easy   | 49   | **114.947**   | `submissions/easy_submission.json` |
| medium | 181  | **169.992** | `submissions/medium_submission.json` |
| hard   | 1051 | **319.771**   | `submissions/hard_submission.json` |

Të gjitha zgjidhjet janë kontrolluar me `validate.py` (ΔV ≤ 100 m/s për çdo transfer,
maksimum 5 përjashtime ≤ 600 m/s, renditje kronologjike, permutacion i plotë).

## Problemi shkurt

* Çdo domate është në një orbitë Kepleriane rreth Hënës (a, e, i, Ω, ω, M).
* Duhet t'i vizitojmë **të gjitha** (një herë secilën) në kohën më të shkurtër.
* Transferi `i → j` që niset në kohën `t` dhe zgjat `tof` kushton
  ΔV = min mbi zgjidhjet e problemit të Lambert-it (të dy drejtimet, deri në 20 rrotullime).
* Çdo transfer duhet ΔV ≤ 100 m/s, përveç maksimum **5** transfereve me ΔV ≤ 600 m/s.
* Vektori i vendimit: `x = [t_1..t_{N-1}, tof_1..tof_{N-1}, π_1..π_N]`,
  objektivi `f = t_{N-1} + tof_{N-1}`.

Struktura e instancave (e zbuluar nga analiza e të dhënave): dy “shtresa” orbitash
(a ≈ 15 000 km dhe a ≈ 3 740 km), grupe (cluster) me inklinacion ≈ 0°, 90°, 180°
dhe një “shkallë” (ladder) domatesh në orbitën e lartë që lidh 0° → 90° → 180°.
Kalimi mes shtresave kushton ≈ 515 m/s (Hohmann) ⇒ duhen përjashtimet.

## Algoritmi (kodi në `kttsp/`)

1. **Zgjidhësi Lambert** (`lambert.py`) – port i vektorizuar në NumPy i algoritmit të Izzo (2015),
   identik me `pykep.lambert_problem` që përdor vlerësuesi zyrtar. Llogarit me qindra mijëra
   transfere në sekondë, pa pasur nevojë për pykep.
2. **Earliest-arrival search** (`transfer.py`) – për një çift `i → j` dhe kohë `t` gjen
   `(t_dep ≥ t, tof)` me kohën më të hershme të mbërritjes që plotëson ΔV ≤ limit:
   rrjetë e trashë 2-D (hapi ∝ periudha orbitale) në dritare kohore që dyfishohen
   + rafinim lokal me rrjeta gjithnjë më të imta.
3. **Rendi udhëzues statik** (`static_order.py`) – kosto statike (ndryshimi i planit
   orbital + Hohmann) → TSP-path me Nearest-Neighbour + 2-opt + Or-opt.
   Ky rend ndan problemin në **segmente** të lidhura me përjashtime.
4. **Konstruksion greedy i varur nga koha** (`solver.py: construct`) – nga domatja aktuale,
   shqyrtohen (screening i vektorizuar) deri në W domate të segmentit aktual dhe zgjidhet ajo
   që arrihet **më herët**. Një invariant garanton që asnjë domate nuk mbetet “e bllokuar”
   (pjesa e mbetur e segmentit mund të vizitohet gjithmonë me transfere të lira).
5. **DP për përjashtimet** (`dp_exceptions`) – për renditje fikse, vetia **FIFO**
   (të mbërrish më herët nuk është kurrë më keq, sepse lejohet pritja) jep
   programim dinamik optimal mbi gjendjet (pozicioni, përjashtime të përdorura).
6. **Local search** (`local_search`) – lëvizje Or-opt në dritare; një lëvizje pranohet
   vetëm nëse koha e mbërritjes në pozicionin e parë të pandryshuar zvogëlohet
   (sipas FIFO kjo garanton përmirësim global) dhe ndryshimi propagohet përpara
   vetëm sa kohë që kohët ndryshojnë.
7. **Multi-start (GRASP)** – rende udhëzuese me zhurmë, dritare W të ndryshme dhe zgjedhje
   gjysmë-të rastësishme; ruhet më e mira.

### Kompleksiteti (përafërsisht)
* Lambert: O(1) për transfer (≤ 15 iteracione Householder për çdo degë × ≤ 41 degë).
* Earliest-arrival: O((H/h)²) vlerësime Lambert (H = dritarja kohore, h = hapi).
* Konstruksioni: O(N · W · G) ku G = madhësia e rrjetës së screening-ut.
* DP: O(N · (E_spare+1)) thirrje earliest-arrival.
* Local search: O(N · w² · w) thirrje për një kalim (w = dritarja e lëvizjeve).

## Si ta ekzekutosh

### VS Code (lokalisht) – e rekomanduar për easy/medium
```bash
pip install -r requirements.txt
python solve.py --level easy --time 900
python validate.py easy
python plot_solution.py easy
```
`solve.py` ruan zgjidhjen vetëm nëse është më e mirë se ajo e ruajtur në `solutions/`
dhe krijon automatikisht `submissions/<level>_submission.json`.

### Kaggle – e rekomanduar për hard (ekzekutim i gjatë)
Hap `kaggle_run.ipynb` në Kaggle (File → Import Notebook), ndrysho `REPO` me linkun e GitHub-it
tënd, pastaj **Save Version → Save & Run All** (ekzekutohet në sfond deri në 12 orë).
Rezultatet shkarkohen nga tab-i *Output*.

### Kontrolli zyrtar me pykep (opsional)
```bash
pip install "pykep<3"        # ose: conda install -c conda-forge "pykep<3"
python validate.py hard --pykep
```

## Dorëzimi në optimise.esa.int
1. Hyr në llogari → sfida *Keplerian Tomato Traveling Salesperson* → **Submit**.
2. Ngarko `submissions/easy_submission.json` (pastaj medium, hard).
3. Nëse platforma ankohet për `problem`, kontrollo në tab-in *Problems* ID-në e saktë dhe
   ndryshoje te `PROBLEM_ID` në `solve.py` (aktualisht `small` / `medium` / `large`), pastaj
   `python make_submissions.py`.

## Struktura
```
kttsp/lambert.py        zgjidhësi Lambert (Izzo) i vektorizuar
kttsp/problem.py        leximi i instancës, efemeridet, vlerësuesi
kttsp/transfer.py       kërkimi earliest-arrival
kttsp/static_order.py   rendi udhëzues (TSP statik)
kttsp/solver.py         konstruksioni, DP, local search, multi-start
solve.py                pika hyrëse (e njëjta për 3 nivelet)
validate.py             kontrolli i zgjidhjeve (+ pykep opsional)
make_submissions.py     rikrijon JSON-et e dorëzimit nga solutions/
plot_solution.py        grafiku i misionit
analyze.py              kohët sipas rajoneve (për raportin)
problems/               instancat zyrtare (ESA, GPL-3.0)
solutions/              zgjidhjet më të mira (vektori x)
submissions/            skedarët JSON për dorëzim
```

## Ku shkon koha (hard, f = 319.77 ditë) – ide për përmirësim
`python analyze.py hard` tregon që ≈186 ditë shkojnë në “shkallën” (ladder) e orbitës së lartë
(129 transfere × 1.44 ditë), kurse grupet (clusters) kushtojnë vetëm 0.05–0.26 ditë për transfer.
Përmirësimi më i madh i mundshëm: renditje më e zgjuar e shkallës (p.sh. kalim zig-zag sipas fazës orbitale).
