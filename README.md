# Chord Finder CLI

A command-line Python tool that generates playable chord fingerings for fretted string instruments (e.g. guitar, ukulele) based on a given tuning.  
Instead of exhaustively listing every possible shape, it uses heuristics to suggest fingerings that are ergonomic and practical to play.

## Features
- see `chords.py --help`  for usage & details
- 🎸 Supports custom tunings (guitar, ukulele, or any fretted instrument).  
- 🧠 Heuristic algorithm prioritizes comfortable and natural fingerings.  
- 🎶 Outputs multiple viable chord shapes to choose from.  
- 🖖 Allows specifying the number of available fingers (e.g., `--fingers=3`).
- 💻 Lightweight CLI, no GUI dependencies.  

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
