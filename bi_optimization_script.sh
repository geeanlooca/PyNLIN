#!/bin/bash
cd /home/lorenzi/Scrivania/progetti/NLIN/PyNLIN/

echo "y" | python scripts/amplification_function.py -L 80
echo "y" | python scripts/plotter_profiles.py -L 80
echo "y" | python scripts/amplification_function.py -L 70
echo "y" | python scripts/plotter_profiles.py -L 70
