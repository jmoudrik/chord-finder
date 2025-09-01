# Chord Finder CLI

A command-line Python tool that generates playable chord fingerings for fretted string instruments (e.g. guitar, ukulele) based on a given tuning.
Instead of exhaustively listing every possible shape, it uses heuristics to suggest fingerings that are ergonomic and practical to play.
This makes it very flexible, allowing different tunings, instruments, and even custom finger counts (e.g. for players with only 3 fingers).

## Features
- 🎸 Supports multiple instruments (Guitar, Ukulele, Bass, Mandolin, Charango, Lapsteel, Dobro) and custom tunings.
- 🧠 Heuristic algorithm prioritizes comfortable and natural fingerings.
- 🎶 Outputs multiple viable chord shapes to choose from.
- 🖖 Allows specifying the number of available fingers (e.g., `--fingers=3`).
- 🌍 Supports English and Czech tone names.
- 💻 Lightweight CLI, no GUI dependencies.

## Usage
```
usage: chords.py [-h] [-n NUMBER] [--fingers FINGERS] [-g | -u | -b | -m | -c | -l | -d | -i INSTRUMENT] [--en | --cz] CHORD [CHORD ...]

Find guitar chords. The program uses many heuristics to suggest fingerings that are ergonomic and most practical to play.

positional arguments:
  CHORD                 a chord to find, e.g. C, C#m, Cmi, Cmi7, etc.

optional arguments:
  -h, --help            show this help message and exit
  -n NUMBER, --number NUMBER
                        number of chords to show
  --fingers FINGERS     number of available fingers
  -g, --guitar          use guitar tuning (default)
  -u, --ukulele         use ukulele tuning
  -b, --bass            use 4-string bass guitar tuning
  -m, --mandolin        use mandolin tuning (GDAE, lowest to highest)
  -c, --charango        use charango tuning
  -l, --lapsteel        use 6-string lap steel tuning (C6)
  -d, --dobro           use dobro / resonator guitar tuning (Open G)
  -i INSTRUMENT, --instrument INSTRUMENT
                        use custom instrument tuning, bottom-most string first - e.g. EADGBE is guitar, DADGBE is guitar Drop-D tuning
  --en                  Use English tone names (default)
  --cz                  Use Czech tone names
```

## Examples
```
 $ # guitar, default
 $ ./chords.py Am
Am

E |---|---|
B |-1-|---|
G |---|-3-|
D |---|-2-|
A |---|---|
E |---|---|

E |---|---|---|
B |-1-|---|---|
G |---|-3-|---|
D |---|-2-|---|
A |---|---|-4-|
E |---|---|---|

    5
E |-1-|---|---|
B |-1-|---|---|
G |-1-|---|---|
D |---|---|-4-|
A |---|---|-3-|

 $ # ukulele
 $ ./chords.py -u Am
Am

A |---|---|
E |---|---|
C |---|---|
G |---|-1-|

A |---|---|-3-|
E |---|---|---|
C |---|---|---|
G |---|-1-|---|
E |-1-|---|---|

 $ # guitar, custom tuning, Drop-D tuning
 $ ./chords.py -i DADGBE D
D

E |---|-1-|---|
B |---|---|-2-|
G |---|-1-|---|
D |---|---|---|
A |---|---|---|
D |---|---|---|

E |---|-1-|---|---|
B |---|---|-2-|---|
G |---|-1-|---|---|
D |---|---|---|-3-|
A |---|---|---|---|
D |---|---|---|---|

 $ # guitar, normal tuning, D for comparison (note the x, not playing the top E)
 $ ./chords.py  D
D

E |---|-1-|---|
B |---|---|-2-|
G |---|-1-|---|
D |---|---|---|
A |---|---|---|
E |---|---|---| x

E |---|-1-|---|---|
B |---|---|-2-|---|
G |---|-1-|---|---|
D |---|---|---|-3-|
A |---|---|---|---|
E |---|---|---|---| x

```
