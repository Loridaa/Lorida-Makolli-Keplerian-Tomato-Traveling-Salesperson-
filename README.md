<image src="../images/KTTSP_banner.png" align="center" alt="Challange 1: Torso Decompositions"/>

# Challenge 2: Keplerian Tomato Traveling Salesperson Problem

This challenge is part of the [SpOC (Space Optimisation Challenge)](https://www.esa.int/gsp/ACT/news/spoc-2026/)
organised by ESA's Advanced Concepts Team and hosted in [GECCO 2026](https://gecco-2026.sigevo.org/HomePage).

## Keplerian Tomato Traveling Salesperson

### Introduction

Oh no! One of the spacecraft that was carrying tomatoes to the Moon
has crashed! Now there is a large amount of tomatoes orbiting the Moon,
forming dangerous debris that could severely compromise the
integrity of future space missions. You have been called as a
logistics expert to solve the misery: You need to collect all
of the orbiting tomatoes in the lowest possible time. Unfortunately,
we were only able to provide you with limited power; therefore any
maneuver that you will perform from tomato to tomato must require a
$\Delta V$ lower than our maximum capability $\Delta V_{max}$. Luckily, your spacecraft is able to handle a limited number of high $\Delta V$ maneuvers to get even the most tricky tomatoes!
You are allowed to wait any arbitrary amount of time at any tomato, but
keep in mind that the future of the Luna Tomato Industry depends on
how quickly you can collect all of the lost tomatoes.

### Theory

This problem combines elements of orbital mechanics and combinatorial
optimization. On one hand, it resembles a Traveling Salesperson
Problem (TSP), where a set of targets must be visited. On the other
hand, the cost of moving between targets is governed by Keplerian
dynamics.

Each tomato follows a fixed Keplerian orbit around the Moon. Transfers
between tomatoes are computed by solving Lambert’s problem, which
determines the velocity changes required to connect two positions in a
given time.

Unlike the classical TSP, the cost of traveling between two tomatoes
depends on both the departure time and the time of
flight. Furthermore, not all transfers are feasible: only those
requiring a $\Delta V$ below a specified threshold are allowed. We additionally allow a small number of special moves with a higher $\Delta V$ threshold.
This introduces a strong coupling between the visiting order, timing, and
feasibility of the solution.

### Input

The input to this challenge is a list of orbits around a central body denoting the path of a tomato orbiting the Moon. Each orbit is defined by six parameters:

$$(a,e,i,\Omega,\omega,\nu)$$

- $a$ is the semi-major axis in meters.
- $e$ is the eccentricity .
- $i$ is the inclination in radians.
- $\Omega$ is the right ascension of the ascending node in radians.
- $\omega$ is the argument of periapsis in radians.
- $\nu$ is the true anomaly at time $0$ in radians.

Each parameter is given as a floating point value.

Orbits are referenced by their index in the input file and start at $0$.

Additionally, each problem comes with parameter line (starts with `p`) that contains additional parameters:

$$(t_0,\Delta t_{min},t_{max},\Delta V_{max}, \Delta V_{max}^{\mathrm{exception}}, E)$$

- $t_0$ is the starting time in days
- $\Delta t_{min}$ is the minimum time of flight in days
- $t_{max}$ is the maximum total time in days
- $\Delta V_{max}$ is the maximum change of velocity for a single transfer.
- $\Delta V_{max}^{\mathrm{exception}}$ is the maximum change of velocity for special high $\Delta V$ moves
- $E$ the number of high $\Delta V$ moves allowed.

All parameters are given as floating point values.

The input file could contain other comment lines (starting with `c`) that can be ignored.

Therefore, an instance file could look like this:

```DIMACS
c starting time [d], minimum time of flight [d], maximum total time [d], maximum delta-v [m/s], exception threshold [m/s] and number of exceptions
p kttsp 0.0 0.001 150.0 100.0 600.0 5
c
c The orbital parameters of the tomatoes:
1.496861100e+07 7.282163486e-03 1.053864218e+00 3.677831327e-03 3.972894704e+00 3.980584570e+00
1.497827726e+07 1.158690595e-03 1.593610788e+00 6.232981268e-03 2.079093608e+00 3.993488927e-01
1.497495376e+07 1.694927467e-03 3.037206232e+00 5.568012625e-03 5.882033922e+00 4.373284192e+00
1.498069418e+07 2.713490318e-03 2.065518687e-02 3.567533267e-03 1.765163584e+00 3.409860056e+00
3.746038900e+06 2.912291402e-03 1.577824251e+00 1.394938607e-03 1.835598963e+00 2.301919351e+00
1.500157063e+07 8.492234105e-03 2.306501279e+00 6.576128923e-03 3.570788266e+00 5.885759249e-01
1.497660921e+07 9.429097039e-03 5.319321089e-01 3.232029320e-03 3.259657612e+00 4.417198393e+00
1.501424963e+07 6.334037565e-03 1.143864218e-01 8.714605902e-03 5.049620585e+00 1.172254253e+00
1.495371830e+07 2.786464642e-03 8.450913743e-01 9.082658860e-03 1.505211752e+00 9.104013314e-01
1.500024812e+07 3.390297910e-03 1.784569170e+00 3.492095746e-03 4.561314055e+00 5.636710004e+00
1.499520489e+07 9.717649377e-04 3.141592654e+00 6.150072267e-03 6.220691804e+00 8.801738263e-01
3.749550897e+06 1.705241237e-03 1.543467712e+00 9.488855373e-03 6.067245002e+00 5.079310340e+00
```

Central to this problem is the computation of a transfer between orbits which corresponds to solving a Lambert problem. To help with this, we provide a Lambert solver that calculates the $\Delta V$ for a given transfer, specified by their starting orbit, end orbit, departure time and time of flight.

```python
def compute_transfer(i_from: int, i_to:int, t_start: float, tof:float)
```

More on this in [the Utilities section](#utilities).

### Output

The decision vector $x$, also called chromosome, consists of the departure times, the times of flight, and the sequence of tomatoes to visit. All times are given in days.

For $N$ tomatoes, the solution $x$ needs to contain the following elements:

$$x = [t_1, \ldots, t_{N-1}, \mathrm{tof}_1, \ldots, \mathrm{tof}_{N-1}, \pi_1, \ldots, \pi_N]$$

- $t_i$ is the departure time of the $i$-th transfer (floating point)
- $\mathrm{tof}_i$ is the time of flight of the $i$-th transfer (floating point)
- $\pi_i$ is the index of the tomato that we are departing from at step $k$ (integer)

Note that $\pi_1, \ldots, \pi_N$ should be a permutation on $0,\ldots,N-1$ as we should visit every tomato but no tomato twice. Additionally, note that we only need $N-1$ transfers to visit $N$ tomatoes.

Transfer $i$ therefore goes from $\pi_i$ to $\pi_{i+1}$ and starts at $t_i$ and takes $\mathrm{tof}_i$ days. It should hold that $t_i + \mathrm{tof}_i \leq t_{i+1}$, i.e. we do not depart from tomato $i+1$ before we even arrived there.

Every transfer has to stay under the imposed $\Delta V_{max}$ limit, i.e.

```python
compute_transfer(pi_i,pi_{i+1},t_i,tof_i) <= Delta-v_max
```

except for up to $E$ maneuvers that have to instead stay under a special, higher $\Delta V$ limit:

```python
compute_transfer(pi_i,pi_{i+1},t_i,tof_i) <= Delta-v_max^exception
```

All in all, the chromosome $x$ should contain $3N-2$ values:

- $N-1$ departure times
- $N-1$ times of flight
- $N$ visited tomatoes

### Objective

The objective function of this problem is the total mission completion time:

$$f = t_{N-1} + \mathrm{tof}_{N-1}$$

Minimizing this value corresponds to completing the mission as quickly as possible.

### Utilities

We provide a python utilities file, containing useful functions such as a Lambert solver and plotting functions. This file can be found on the [SpOC4 Github](https://github.com/ESA/spoc4).

The file contains a central class `TomatoProblem` which is to be instantiated via the relevant problem file

```python
util = TomatoProblem("instance.kttsp")
```

Note that we need `numpy`,`matplotlib` and `pykep` (Version `2.x`, not `3.x`) as dependencies.

One can then find the $\Delta V$ cost of a transfer via

```python
util.compute_transfer(i_from,i_to,t_start,tof)
```

One can then plot (parts of) their trajectory via

```python
util.plot_transfer(i_from,i_to,t_start,tof)
util.plot_full_trajectory(x) # x is the trajectory encoded like the chromosome
```

### Hints

- Transfers may be infeasible depending on timing and orbital geometry.
- Two tomatoes might never have a single valid transfer between them.
- The visiting order strongly interacts with orbital dynamics.
- Waiting can reduce $\Delta V$, but increase total mission time.
- Clustered orbits may allow low-cost local transfers, while long-range transfers require careful timing.

### Submitting

To submit a solution, you can prepare a submission file with the [submission helper](https://api.optimize.esa.int/data/tools/submission_helper.py) via

```python
from submisson_helper import create_submission
create_submission("spoc-4-keplerian-tomato-traveling-salesperson","{problem}",x,"submission_file.json","submission_name","submission_description")
```

where `{problem}` corresponds to the instance name (`small`,`medium`,`large`), and $x$ is the chromosome.


## Project Overview

Keplerian Tomato Traveling Salesperson is an algorithm
optimization project developed for the Design and
Analysis of Algorithms course.

The objective is to find an efficient route for
collecting tomatoes orbiting the Moon while
respecting spacecraft maneuvering constraints.

## Current Progress

- Python and Conda environment configured
- Project structure initialized
- ESA datasets added
- Easy instance with 49 tomatoes loaded
- Mission parameters validated
- Initial Python tests completed

## Credits

This project is based on the ESA SpOC4 challenge:
https://github.com/esa/SpOC4

The orbital datasets and utils_users.py originate
from ESA's SpOC4 repository and are used with
modifications for this educational project.


## ALNS V1.1 - Hapi 169G
- Metoda: Adaptive Large Neighborhood Search me Simulated Annealing.
- Instanca: easy.kttsp (49 domate).
- Baseline i verifikuar: 38/49.
- Niveli me i mire i raportuar: 38/49.
- Prova: 18.
- Kandidate te vlefshem: 2.
- Vleresime orbitale: 2124.
- Verifikimi: True.
- Rekordi origjinal 38/49 nuk eshte mbishkruar.


## Hapi 170A - ALNS V2 Append Probe
- Baza: 38/49 domate, 5/5 SPECIAL.
- Testi: shtim i domates ne fund te rruges.
- Rezultati: nuk u gjet transferim NORMAL.
- Vleresime: 2354.
- Hapi pasues: futje brenda rruges.
- Rekordi origjinal mbetet i pandryshuar.


## Hapi 170C - Internal Insertion Search
- Metoda: futje e nje domateje brenda rruges.
- Baseline: 38/49 domate.
- Rezultati: 38/49.
- Kombinime: 396.
- Vleresime: 14354.
- Kandidat i verifikuar: False.
- Rekordi 38/49 nuk u mbishkrua.


## Hapi 170E - ALNS V2 Segment Repair
- Metoda: ALNS targeted destroy-repair.
- Riorganizim i segmenteve me 2-5 domate.
- Instanca: easy.kttsp.
- Baseline: 38/49 domate.
- Rezultati: 38/49.
- Prova: 90.
- Vleresime orbitale: 24536.
- Kandidat i verifikuar: False.
- Rekordi origjinal mbetet i paprekur.


## Hapi 171 - ALNS V3 Suffix Rebuild
- Metoda: ALNS V3 me riorganizim segmentesh dhe rillogaritje te transferimeve.
- Instanca: easy.kttsp.
- Baseline: 38/49 domate.
- Rezultati: 38/49.
- Prova: 75.
- Vleresime orbitale: 67648.
- Riparime te plota: 0.
- Maksimumi i transferimeve te riparuara: 19.
- Nuk u gjet kandidat i ri i verifikuar.
- Rekordi 38/49 mbeti i paprekur.
- Hapi tjeter: analiza e pikes ku deshton riparimi i rruges.


## Hapi 172 - ALNS V4 DP Repair
- Algoritmi: ALNS V4 me dynamic programming.
- Instanca: easy.kttsp (49 domate).
- Baseline: 38/49 domate.
- Rezultati: 39/49.
- Prova: 32.
- Vleresime: 11953.
- Riparime te plota: 1.
- Kandidat i verifikuar: True.
- Rekordi baze mbetet i pandryshuar.


## Hapi 173 - Rekordi 39/49
- Algoritmi: ALNS V4 + Dynamic Programming.
- Instanca: easy.kttsp.
- Rekordi i ri: 39/49 domate.
- Domatja e shtuar: 26.
- Dita perfundimtare: 181.016876.
- Transferime SPECIAL: 5/5.
- Verifikuar me PyKEP.
- Rekordi 38/49 mbetet i ruajtur.
- Objektivi i ardhshem: 40/49.


## Hapi 174 - ALNS V5 nga 39/49
- Algoritmi: targeted neighborhood repair me Dynamic Programming.
- Baza: rekordi i verifikuar 39/49.
- Rezultati: 39/49.
- Prova: 210.
- Vleresime orbitale: 51885.
- Riparime te plota: 0.
- Kandidat i verifikuar: False.
- Rekordet e vjetra nuk u mbishkruan.


## Hapi 175 - ALNS V6 Special Release
- Algoritmi: ALNS V6 + Dynamic Programming.
- Strategjia: lirimi i transferimeve SPECIAL para futjes se domates 40.
- Baseline: 39/49 domate.
- Rezultati: 39/49.
- Prova release: 140.
- Variante 39: 0.
- Prova insertion: 0.
- Vleresime: 44951.
- Kandidat 40 i verifikuar: False.
- Rekordet 38 dhe 39 nuk u mbishkruan.


## Hapi 176 - ALNS V7 Neutral Walk
- Algoritmi: ALNS V7 + Dynamic Programming.
- Strategjia: variante te vlefshme me 39 domate, pastaj kerkim per 40.
- Baseline: 39/49 domate.
- Rezultati: 39/49.
- Prova 39: 19.
- Variante 39: 12.
- Prova 40: 260.
- Vleresime: 24404.
- Kandidat 40 i verifikuar: False.
- Rekordet 38 dhe 39 mbeten te pandryshuara.


## Hapi 177 - ALNS V8 Edge Escape
- Metoda: ALNS Edge Escape + Dynamic Programming.
- Baza: 39/49 domate.
- Burimet: rekordi 39 dhe alternativa e V7.
- Strategjia: pozicione dhe renditje te reja per te shmangur lidhjet problematike.
- Rezultati: 39/49.
- Prova: 320.
- Vleresime: 45256.
- Riparime te plota: 0.
- Kandidat 40 i verifikuar: False.
- Rekordet 38/39 mbeten te pandryshuara.


## Hapi 178 - ALNS V9 Multi-Start
- Instanca: easy.kttsp (49 domate).
- Metoda: neutral 39/49 multi-start + DP targeted insertion.
- Rekordi baze: 39/49; rezultati: 39/49.
- Alternativa te reja 39: 3.
- Prova per 40: 420.
- Vleresime orbitale: 43143.
- Kandidati 40 i verifikuar: False.
- Rekordet 38/49 dhe 39/49 nuk u mbishkruan.


## Hapi 179 - ALNS V10 Temporal Labels
- Algoritmi: ALNS V10 me multi-label Dynamic Programming.
- Baza: 39/49 domate.
- Burimet: rekordi 39 dhe arkivi V9.
- Rezultati: 39/49.
- Prova: 180.
- Vleresime: 20286.
- Kandidat 40: False.
- Rekordet 38/39 mbeten te pandryshuara.


## Hapi 180 - ALNS V11 Exchange + Insert
- Metoda: zevendesim i nje domateje, pastaj futje per 40/49.
- Baseline: 39/49 domate.
- Variante 39: 0.
- Prova exchange: 160.
- Prova insertion: 0.
- Vleresime: 18955.
- Rezultati: 39/49.
- Kandidat 40: False.
- Rekordet origjinale te paprekura.


## Hapi 181 - ALNS V12 Direct 1-for-2
- Metoda: zevendesim direkt 1-for-2 me Dynamic Programming.
- Baza: 39/49 domate.
- Nuk kerkohet nje zgjidhje e ndermjetme 39/49.
- Rezultati: 39/49.
- Prova: 320.
- Vleresime orbitale: 21032.
- Kandidat 40: False.
- Rekordet 38 dhe 39 nuk mbishkruhen.


## Hapi 182 - ALNS V13 Edge Rescue
- Metoda: edge rescue, zgjerim dritaresh dhe dy metoda Dynamic Programming.
- Baza: 39/49 domate.
- Fokus: lidhjet problematike te V12.
- Rezultati: 39/49.
- Prova: 260.
- Vleresime: 50595.
- Riparime te plota: 0.
- Kandidat 40: False.
- Rekordet 38 dhe 39 te pandryshuara.


## Hapi 183 - Diagnostika e V13
- Metoda: riprodhim i proves me te afert dhe skanim orbital me PyKEP.
- Baseline: 39/49.
- Lidhja e analizuar: {'from': 23, 'to': 4}.
- DV minimale e testuar: 1626.610952.
- Vleresime shtese: 445.
- Skanimi nuk provon pamundesi globale.
- Rekordet 38/49 dhe 39/49 jane te pandryshuara.


## Hapi 184 - ALNS V14 Edge Bypass
- Algoritmi: ALNS V14 + Dynamic Programming.
- Strategjia: shmangia e lidhjes 23 -> 4 ne dritaren problematike.
- Baza: 39/49 domate.
- Rezultati: 39/49.
- Prova: 220.
- Vleresime orbitale: 65552.
- Riparime te plota: 0.
- Kandidat 40: False.
- Rekordet 38 dhe 39 mbeten te pandryshuara.


## Hapi 185 - ALNS V15 Adaptive Edge Repair
- Metoda: Adaptive Edge Repair + Dynamic Programming.
- Baza: rekordi 39/49.
- Fokus: prova 25/27 nga ALNS V14.
- Ndryshime adaptive rreth transferimeve qe deshtojne.
- Rezultati: 39/49.
- Prova: 210.
- Vleresime: 66431.
- Riparime te plota: 0.
- Kandidat 40: False.
- Rekordet 38 dhe 39 nuk u mbishkruan.
