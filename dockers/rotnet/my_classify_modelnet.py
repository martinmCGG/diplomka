import numpy as np
import os
import sys
import argparse
import glob
import time


ang = np.zeros([60,20],dtype=int)
ang[ 0 ][ 0 ] = 0; ang[ 0 ][ 1 ] = 1; ang[ 0 ][ 2 ] = 2; ang[ 0 ][ 3 ] = 3; ang[ 0 ][ 4 ] = 4; ang[ 0 ][ 5 ] = 5; ang[ 0 ][ 6 ] = 6; ang[ 0 ][ 7 ] = 7; ang[ 0 ][ 8 ] = 8; ang[ 0 ][ 9 ] = 9; ang[ 0 ][ 10 ] = 10; ang[ 0 ][ 11 ] = 11; ang[ 0 ][ 12 ] = 12; ang[ 0 ][ 13 ] = 13; ang[ 0 ][ 14 ] = 14; ang[ 0 ][ 15 ] = 15; ang[ 0 ][ 16 ] = 16; ang[ 0 ][ 17 ] = 17; ang[ 0 ][ 18 ] = 18; ang[ 0 ][ 19 ] = 19; 
ang[ 1 ][ 0 ] = 0; ang[ 1 ][ 1 ] = 4; ang[ 1 ][ 2 ] = 1; ang[ 1 ][ 3 ] = 5; ang[ 1 ][ 4 ] = 2; ang[ 1 ][ 5 ] = 6; ang[ 1 ][ 6 ] = 3; ang[ 1 ][ 7 ] = 7; ang[ 1 ][ 8 ] = 12; ang[ 1 ][ 9 ] = 14; ang[ 1 ][ 10 ] = 13; ang[ 1 ][ 11 ] = 15; ang[ 1 ][ 12 ] = 16; ang[ 1 ][ 13 ] = 17; ang[ 1 ][ 14 ] = 18; ang[ 1 ][ 15 ] = 19; ang[ 1 ][ 16 ] = 8; ang[ 1 ][ 17 ] = 10; ang[ 1 ][ 18 ] = 9; ang[ 1 ][ 19 ] = 11; 
ang[ 2 ][ 0 ] = 3; ang[ 2 ][ 1 ] = 1; ang[ 2 ][ 2 ] = 7; ang[ 2 ][ 3 ] = 5; ang[ 2 ][ 4 ] = 2; ang[ 2 ][ 5 ] = 0; ang[ 2 ][ 6 ] = 6; ang[ 2 ][ 7 ] = 4; ang[ 2 ][ 8 ] = 18; ang[ 2 ][ 9 ] = 16; ang[ 2 ][ 10 ] = 19; ang[ 2 ][ 11 ] = 17; ang[ 2 ][ 12 ] = 11; ang[ 2 ][ 13 ] = 9; ang[ 2 ][ 14 ] = 10; ang[ 2 ][ 15 ] = 8; ang[ 2 ][ 16 ] = 13; ang[ 2 ][ 17 ] = 12; ang[ 2 ][ 18 ] = 15; ang[ 2 ][ 19 ] = 14; 
ang[ 3 ][ 0 ] = 6; ang[ 3 ][ 1 ] = 4; ang[ 3 ][ 2 ] = 2; ang[ 3 ][ 3 ] = 0; ang[ 3 ][ 4 ] = 7; ang[ 3 ][ 5 ] = 5; ang[ 3 ][ 6 ] = 3; ang[ 3 ][ 7 ] = 1; ang[ 3 ][ 8 ] = 19; ang[ 3 ][ 9 ] = 17; ang[ 3 ][ 10 ] = 18; ang[ 3 ][ 11 ] = 16; ang[ 3 ][ 12 ] = 10; ang[ 3 ][ 13 ] = 8; ang[ 3 ][ 14 ] = 11; ang[ 3 ][ 15 ] = 9; ang[ 3 ][ 16 ] = 14; ang[ 3 ][ 17 ] = 15; ang[ 3 ][ 18 ] = 12; ang[ 3 ][ 19 ] = 13; 
ang[ 4 ][ 0 ] = 6; ang[ 4 ][ 1 ] = 2; ang[ 4 ][ 2 ] = 7; ang[ 4 ][ 3 ] = 3; ang[ 4 ][ 4 ] = 4; ang[ 4 ][ 5 ] = 0; ang[ 4 ][ 6 ] = 5; ang[ 4 ][ 7 ] = 1; ang[ 4 ][ 8 ] = 14; ang[ 4 ][ 9 ] = 12; ang[ 4 ][ 10 ] = 15; ang[ 4 ][ 11 ] = 13; ang[ 4 ][ 12 ] = 19; ang[ 4 ][ 13 ] = 18; ang[ 4 ][ 14 ] = 17; ang[ 4 ][ 15 ] = 16; ang[ 4 ][ 16 ] = 10; ang[ 4 ][ 17 ] = 8; ang[ 4 ][ 18 ] = 11; ang[ 4 ][ 19 ] = 9; 
ang[ 5 ][ 0 ] = 5; ang[ 5 ][ 1 ] = 7; ang[ 5 ][ 2 ] = 1; ang[ 5 ][ 3 ] = 3; ang[ 5 ][ 4 ] = 4; ang[ 5 ][ 5 ] = 6; ang[ 5 ][ 6 ] = 0; ang[ 5 ][ 7 ] = 2; ang[ 5 ][ 8 ] = 17; ang[ 5 ][ 9 ] = 19; ang[ 5 ][ 10 ] = 16; ang[ 5 ][ 11 ] = 18; ang[ 5 ][ 12 ] = 9; ang[ 5 ][ 13 ] = 11; ang[ 5 ][ 14 ] = 8; ang[ 5 ][ 15 ] = 10; ang[ 5 ][ 16 ] = 15; ang[ 5 ][ 17 ] = 14; ang[ 5 ][ 18 ] = 13; ang[ 5 ][ 19 ] = 12; 
ang[ 6 ][ 0 ] = 3; ang[ 6 ][ 1 ] = 7; ang[ 6 ][ 2 ] = 2; ang[ 6 ][ 3 ] = 6; ang[ 6 ][ 4 ] = 1; ang[ 6 ][ 5 ] = 5; ang[ 6 ][ 6 ] = 0; ang[ 6 ][ 7 ] = 4; ang[ 6 ][ 8 ] = 13; ang[ 6 ][ 9 ] = 15; ang[ 6 ][ 10 ] = 12; ang[ 6 ][ 11 ] = 14; ang[ 6 ][ 12 ] = 18; ang[ 6 ][ 13 ] = 19; ang[ 6 ][ 14 ] = 16; ang[ 6 ][ 15 ] = 17; ang[ 6 ][ 16 ] = 11; ang[ 6 ][ 17 ] = 9; ang[ 6 ][ 18 ] = 10; ang[ 6 ][ 19 ] = 8; 
ang[ 7 ][ 0 ] = 5; ang[ 7 ][ 1 ] = 1; ang[ 7 ][ 2 ] = 4; ang[ 7 ][ 3 ] = 0; ang[ 7 ][ 4 ] = 7; ang[ 7 ][ 5 ] = 3; ang[ 7 ][ 6 ] = 6; ang[ 7 ][ 7 ] = 2; ang[ 7 ][ 8 ] = 15; ang[ 7 ][ 9 ] = 13; ang[ 7 ][ 10 ] = 14; ang[ 7 ][ 11 ] = 12; ang[ 7 ][ 12 ] = 17; ang[ 7 ][ 13 ] = 16; ang[ 7 ][ 14 ] = 19; ang[ 7 ][ 15 ] = 18; ang[ 7 ][ 16 ] = 9; ang[ 7 ][ 17 ] = 11; ang[ 7 ][ 18 ] = 8; ang[ 7 ][ 19 ] = 10; 
ang[ 8 ][ 0 ] = 0; ang[ 8 ][ 1 ] = 2; ang[ 8 ][ 2 ] = 4; ang[ 8 ][ 3 ] = 6; ang[ 8 ][ 4 ] = 1; ang[ 8 ][ 5 ] = 3; ang[ 8 ][ 6 ] = 5; ang[ 8 ][ 7 ] = 7; ang[ 8 ][ 8 ] = 16; ang[ 8 ][ 9 ] = 18; ang[ 8 ][ 10 ] = 17; ang[ 8 ][ 11 ] = 19; ang[ 8 ][ 12 ] = 8; ang[ 8 ][ 13 ] = 10; ang[ 8 ][ 14 ] = 9; ang[ 8 ][ 15 ] = 11; ang[ 8 ][ 16 ] = 12; ang[ 8 ][ 17 ] = 13; ang[ 8 ][ 18 ] = 14; ang[ 8 ][ 19 ] = 15; 
ang[ 9 ][ 0 ] = 4; ang[ 9 ][ 1 ] = 15; ang[ 9 ][ 2 ] = 16; ang[ 9 ][ 3 ] = 9; ang[ 9 ][ 4 ] = 10; ang[ 9 ][ 5 ] = 19; ang[ 9 ][ 6 ] = 12; ang[ 9 ][ 7 ] = 3; ang[ 9 ][ 8 ] = 8; ang[ 9 ][ 9 ] = 7; ang[ 9 ][ 10 ] = 0; ang[ 9 ][ 11 ] = 11; ang[ 9 ][ 12 ] = 17; ang[ 9 ][ 13 ] = 5; ang[ 9 ][ 14 ] = 2; ang[ 9 ][ 15 ] = 18; ang[ 9 ][ 16 ] = 14; ang[ 9 ][ 17 ] = 6; ang[ 9 ][ 18 ] = 1; ang[ 9 ][ 19 ] = 13; 
ang[ 10 ][ 0 ] = 18; ang[ 10 ][ 1 ] = 11; ang[ 10 ][ 2 ] = 6; ang[ 10 ][ 3 ] = 15; ang[ 10 ][ 4 ] = 12; ang[ 10 ][ 5 ] = 1; ang[ 10 ][ 6 ] = 8; ang[ 10 ][ 7 ] = 17; ang[ 10 ][ 8 ] = 2; ang[ 10 ][ 9 ] = 9; ang[ 10 ][ 10 ] = 10; ang[ 10 ][ 11 ] = 5; ang[ 10 ][ 12 ] = 19; ang[ 10 ][ 13 ] = 7; ang[ 10 ][ 14 ] = 0; ang[ 10 ][ 15 ] = 16; ang[ 10 ][ 16 ] = 3; ang[ 10 ][ 17 ] = 13; ang[ 10 ][ 18 ] = 14; ang[ 10 ][ 19 ] = 4; 
ang[ 11 ][ 0 ] = 14; ang[ 11 ][ 1 ] = 5; ang[ 11 ][ 2 ] = 8; ang[ 11 ][ 3 ] = 16; ang[ 11 ][ 4 ] = 19; ang[ 11 ][ 5 ] = 11; ang[ 11 ][ 6 ] = 2; ang[ 11 ][ 7 ] = 13; ang[ 11 ][ 8 ] = 6; ang[ 11 ][ 9 ] = 9; ang[ 11 ][ 10 ] = 10; ang[ 11 ][ 11 ] = 1; ang[ 11 ][ 12 ] = 4; ang[ 11 ][ 13 ] = 17; ang[ 11 ][ 14 ] = 18; ang[ 11 ][ 15 ] = 3; ang[ 11 ][ 16 ] = 15; ang[ 11 ][ 17 ] = 7; ang[ 11 ][ 18 ] = 0; ang[ 11 ][ 19 ] = 12; 
ang[ 12 ][ 0 ] = 10; ang[ 12 ][ 1 ] = 18; ang[ 12 ][ 2 ] = 14; ang[ 12 ][ 3 ] = 7; ang[ 12 ][ 4 ] = 0; ang[ 12 ][ 5 ] = 13; ang[ 12 ][ 6 ] = 17; ang[ 12 ][ 7 ] = 9; ang[ 12 ][ 8 ] = 8; ang[ 12 ][ 9 ] = 3; ang[ 12 ][ 10 ] = 4; ang[ 12 ][ 11 ] = 11; ang[ 12 ][ 12 ] = 6; ang[ 12 ][ 13 ] = 19; ang[ 12 ][ 14 ] = 16; ang[ 12 ][ 15 ] = 1; ang[ 12 ][ 16 ] = 2; ang[ 12 ][ 17 ] = 12; ang[ 12 ][ 18 ] = 15; ang[ 12 ][ 19 ] = 5; 
ang[ 13 ][ 0 ] = 2; ang[ 13 ][ 1 ] = 8; ang[ 13 ][ 2 ] = 13; ang[ 13 ][ 3 ] = 16; ang[ 13 ][ 4 ] = 19; ang[ 13 ][ 5 ] = 14; ang[ 13 ][ 6 ] = 11; ang[ 13 ][ 7 ] = 5; ang[ 13 ][ 8 ] = 18; ang[ 13 ][ 9 ] = 4; ang[ 13 ][ 10 ] = 3; ang[ 13 ][ 11 ] = 17; ang[ 13 ][ 12 ] = 12; ang[ 13 ][ 13 ] = 0; ang[ 13 ][ 14 ] = 7; ang[ 13 ][ 15 ] = 15; ang[ 13 ][ 16 ] = 10; ang[ 13 ][ 17 ] = 6; ang[ 13 ][ 18 ] = 1; ang[ 13 ][ 19 ] = 9; 
ang[ 14 ][ 0 ] = 18; ang[ 14 ][ 1 ] = 12; ang[ 14 ][ 2 ] = 11; ang[ 14 ][ 3 ] = 1; ang[ 14 ][ 4 ] = 6; ang[ 14 ][ 5 ] = 8; ang[ 14 ][ 6 ] = 15; ang[ 14 ][ 7 ] = 17; ang[ 14 ][ 8 ] = 19; ang[ 14 ][ 9 ] = 0; ang[ 14 ][ 10 ] = 7; ang[ 14 ][ 11 ] = 16; ang[ 14 ][ 12 ] = 3; ang[ 14 ][ 13 ] = 13; ang[ 14 ][ 14 ] = 14; ang[ 14 ][ 15 ] = 4; ang[ 14 ][ 16 ] = 2; ang[ 14 ][ 17 ] = 10; ang[ 14 ][ 18 ] = 9; ang[ 14 ][ 19 ] = 5; 
ang[ 15 ][ 0 ] = 9; ang[ 15 ][ 1 ] = 3; ang[ 15 ][ 2 ] = 16; ang[ 15 ][ 3 ] = 12; ang[ 15 ][ 4 ] = 15; ang[ 15 ][ 5 ] = 19; ang[ 15 ][ 6 ] = 4; ang[ 15 ][ 7 ] = 10; ang[ 15 ][ 8 ] = 5; ang[ 15 ][ 9 ] = 18; ang[ 15 ][ 10 ] = 17; ang[ 15 ][ 11 ] = 2; ang[ 15 ][ 12 ] = 1; ang[ 15 ][ 13 ] = 13; ang[ 15 ][ 14 ] = 14; ang[ 15 ][ 15 ] = 6; ang[ 15 ][ 16 ] = 11; ang[ 15 ][ 17 ] = 7; ang[ 15 ][ 18 ] = 0; ang[ 15 ][ 19 ] = 8; 
ang[ 16 ][ 0 ] = 13; ang[ 16 ][ 1 ] = 18; ang[ 16 ][ 2 ] = 0; ang[ 16 ][ 3 ] = 10; ang[ 16 ][ 4 ] = 9; ang[ 16 ][ 5 ] = 7; ang[ 16 ][ 6 ] = 17; ang[ 16 ][ 7 ] = 14; ang[ 16 ][ 8 ] = 1; ang[ 16 ][ 9 ] = 19; ang[ 16 ][ 10 ] = 16; ang[ 16 ][ 11 ] = 6; ang[ 16 ][ 12 ] = 12; ang[ 16 ][ 13 ] = 2; ang[ 16 ][ 14 ] = 5; ang[ 16 ][ 15 ] = 15; ang[ 16 ][ 16 ] = 3; ang[ 16 ][ 17 ] = 11; ang[ 16 ][ 18 ] = 8; ang[ 16 ][ 19 ] = 4; 
ang[ 17 ][ 0 ] = 1; ang[ 17 ][ 1 ] = 17; ang[ 17 ][ 2 ] = 11; ang[ 17 ][ 3 ] = 15; ang[ 17 ][ 4 ] = 12; ang[ 17 ][ 5 ] = 8; ang[ 17 ][ 6 ] = 18; ang[ 17 ][ 7 ] = 6; ang[ 17 ][ 8 ] = 13; ang[ 17 ][ 9 ] = 4; ang[ 17 ][ 10 ] = 3; ang[ 17 ][ 11 ] = 14; ang[ 17 ][ 12 ] = 9; ang[ 17 ][ 13 ] = 5; ang[ 17 ][ 14 ] = 2; ang[ 17 ][ 15 ] = 10; ang[ 17 ][ 16 ] = 16; ang[ 17 ][ 17 ] = 0; ang[ 17 ][ 18 ] = 7; ang[ 17 ][ 19 ] = 19; 
ang[ 18 ][ 0 ] = 9; ang[ 18 ][ 1 ] = 15; ang[ 18 ][ 2 ] = 3; ang[ 18 ][ 3 ] = 19; ang[ 18 ][ 4 ] = 16; ang[ 18 ][ 5 ] = 4; ang[ 18 ][ 6 ] = 12; ang[ 18 ][ 7 ] = 10; ang[ 18 ][ 8 ] = 1; ang[ 18 ][ 9 ] = 14; ang[ 18 ][ 10 ] = 13; ang[ 18 ][ 11 ] = 6; ang[ 18 ][ 12 ] = 11; ang[ 18 ][ 13 ] = 7; ang[ 18 ][ 14 ] = 0; ang[ 18 ][ 15 ] = 8; ang[ 18 ][ 16 ] = 5; ang[ 18 ][ 17 ] = 17; ang[ 18 ][ 18 ] = 18; ang[ 18 ][ 19 ] = 2; 
ang[ 19 ][ 0 ] = 14; ang[ 19 ][ 1 ] = 8; ang[ 19 ][ 2 ] = 19; ang[ 19 ][ 3 ] = 2; ang[ 19 ][ 4 ] = 5; ang[ 19 ][ 5 ] = 16; ang[ 19 ][ 6 ] = 11; ang[ 19 ][ 7 ] = 13; ang[ 19 ][ 8 ] = 15; ang[ 19 ][ 9 ] = 0; ang[ 19 ][ 10 ] = 7; ang[ 19 ][ 11 ] = 12; ang[ 19 ][ 12 ] = 6; ang[ 19 ][ 13 ] = 10; ang[ 19 ][ 14 ] = 9; ang[ 19 ][ 15 ] = 1; ang[ 19 ][ 16 ] = 4; ang[ 19 ][ 17 ] = 17; ang[ 19 ][ 18 ] = 18; ang[ 19 ][ 19 ] = 3; 
ang[ 20 ][ 0 ] = 17; ang[ 20 ][ 1 ] = 0; ang[ 20 ][ 2 ] = 14; ang[ 20 ][ 3 ] = 10; ang[ 20 ][ 4 ] = 9; ang[ 20 ][ 5 ] = 13; ang[ 20 ][ 6 ] = 7; ang[ 20 ][ 7 ] = 18; ang[ 20 ][ 8 ] = 5; ang[ 20 ][ 9 ] = 12; ang[ 20 ][ 10 ] = 15; ang[ 20 ][ 11 ] = 2; ang[ 20 ][ 12 ] = 4; ang[ 20 ][ 13 ] = 8; ang[ 20 ][ 14 ] = 11; ang[ 20 ][ 15 ] = 3; ang[ 20 ][ 16 ] = 16; ang[ 20 ][ 17 ] = 1; ang[ 20 ][ 18 ] = 6; ang[ 20 ][ 19 ] = 19; 
ang[ 21 ][ 0 ] = 8; ang[ 21 ][ 1 ] = 6; ang[ 21 ][ 2 ] = 17; ang[ 21 ][ 3 ] = 15; ang[ 21 ][ 4 ] = 12; ang[ 21 ][ 5 ] = 18; ang[ 21 ][ 6 ] = 1; ang[ 21 ][ 7 ] = 11; ang[ 21 ][ 8 ] = 0; ang[ 21 ][ 9 ] = 19; ang[ 21 ][ 10 ] = 16; ang[ 21 ][ 11 ] = 7; ang[ 21 ][ 12 ] = 4; ang[ 21 ][ 13 ] = 14; ang[ 21 ][ 14 ] = 13; ang[ 21 ][ 15 ] = 3; ang[ 21 ][ 16 ] = 10; ang[ 21 ][ 17 ] = 2; ang[ 21 ][ 18 ] = 5; ang[ 21 ][ 19 ] = 9; 
ang[ 22 ][ 0 ] = 12; ang[ 22 ][ 1 ] = 10; ang[ 22 ][ 2 ] = 16; ang[ 22 ][ 3 ] = 4; ang[ 22 ][ 4 ] = 3; ang[ 22 ][ 5 ] = 19; ang[ 22 ][ 6 ] = 9; ang[ 22 ][ 7 ] = 15; ang[ 22 ][ 8 ] = 13; ang[ 22 ][ 9 ] = 6; ang[ 22 ][ 10 ] = 1; ang[ 22 ][ 11 ] = 14; ang[ 22 ][ 12 ] = 0; ang[ 22 ][ 13 ] = 8; ang[ 22 ][ 14 ] = 11; ang[ 22 ][ 15 ] = 7; ang[ 22 ][ 16 ] = 2; ang[ 22 ][ 17 ] = 18; ang[ 22 ][ 18 ] = 17; ang[ 22 ][ 19 ] = 5; 
ang[ 23 ][ 0 ] = 16; ang[ 23 ][ 1 ] = 8; ang[ 23 ][ 2 ] = 5; ang[ 23 ][ 3 ] = 14; ang[ 23 ][ 4 ] = 13; ang[ 23 ][ 5 ] = 2; ang[ 23 ][ 6 ] = 11; ang[ 23 ][ 7 ] = 19; ang[ 23 ][ 8 ] = 1; ang[ 23 ][ 9 ] = 10; ang[ 23 ][ 10 ] = 9; ang[ 23 ][ 11 ] = 6; ang[ 23 ][ 12 ] = 17; ang[ 23 ][ 13 ] = 4; ang[ 23 ][ 14 ] = 3; ang[ 23 ][ 15 ] = 18; ang[ 23 ][ 16 ] = 0; ang[ 23 ][ 17 ] = 12; ang[ 23 ][ 18 ] = 15; ang[ 23 ][ 19 ] = 7; 
ang[ 24 ][ 0 ] = 7; ang[ 24 ][ 1 ] = 9; ang[ 24 ][ 2 ] = 14; ang[ 24 ][ 3 ] = 17; ang[ 24 ][ 4 ] = 18; ang[ 24 ][ 5 ] = 13; ang[ 24 ][ 6 ] = 10; ang[ 24 ][ 7 ] = 0; ang[ 24 ][ 8 ] = 19; ang[ 24 ][ 9 ] = 1; ang[ 24 ][ 10 ] = 6; ang[ 24 ][ 11 ] = 16; ang[ 24 ][ 12 ] = 15; ang[ 24 ][ 13 ] = 5; ang[ 24 ][ 14 ] = 2; ang[ 24 ][ 15 ] = 12; ang[ 24 ][ 16 ] = 11; ang[ 24 ][ 17 ] = 3; ang[ 24 ][ 18 ] = 4; ang[ 24 ][ 19 ] = 8; 
ang[ 25 ][ 0 ] = 11; ang[ 25 ][ 1 ] = 13; ang[ 25 ][ 2 ] = 5; ang[ 25 ][ 3 ] = 16; ang[ 25 ][ 4 ] = 19; ang[ 25 ][ 5 ] = 2; ang[ 25 ][ 6 ] = 14; ang[ 25 ][ 7 ] = 8; ang[ 25 ][ 8 ] = 7; ang[ 25 ][ 9 ] = 12; ang[ 25 ][ 10 ] = 15; ang[ 25 ][ 11 ] = 0; ang[ 25 ][ 12 ] = 9; ang[ 25 ][ 13 ] = 1; ang[ 25 ][ 14 ] = 6; ang[ 25 ][ 15 ] = 10; ang[ 25 ][ 16 ] = 3; ang[ 25 ][ 17 ] = 18; ang[ 25 ][ 18 ] = 17; ang[ 25 ][ 19 ] = 4; 
ang[ 26 ][ 0 ] = 9; ang[ 26 ][ 1 ] = 16; ang[ 26 ][ 2 ] = 15; ang[ 26 ][ 3 ] = 4; ang[ 26 ][ 4 ] = 3; ang[ 26 ][ 5 ] = 12; ang[ 26 ][ 6 ] = 19; ang[ 26 ][ 7 ] = 10; ang[ 26 ][ 8 ] = 11; ang[ 26 ][ 9 ] = 0; ang[ 26 ][ 10 ] = 7; ang[ 26 ][ 11 ] = 8; ang[ 26 ][ 12 ] = 5; ang[ 26 ][ 13 ] = 17; ang[ 26 ][ 14 ] = 18; ang[ 26 ][ 15 ] = 2; ang[ 26 ][ 16 ] = 1; ang[ 26 ][ 17 ] = 13; ang[ 26 ][ 18 ] = 14; ang[ 26 ][ 19 ] = 6; 
ang[ 27 ][ 0 ] = 19; ang[ 27 ][ 1 ] = 15; ang[ 27 ][ 2 ] = 10; ang[ 27 ][ 3 ] = 4; ang[ 27 ][ 4 ] = 3; ang[ 27 ][ 5 ] = 9; ang[ 27 ][ 6 ] = 12; ang[ 27 ][ 7 ] = 16; ang[ 27 ][ 8 ] = 18; ang[ 27 ][ 9 ] = 5; ang[ 27 ][ 10 ] = 2; ang[ 27 ][ 11 ] = 17; ang[ 27 ][ 12 ] = 6; ang[ 27 ][ 13 ] = 14; ang[ 27 ][ 14 ] = 13; ang[ 27 ][ 15 ] = 1; ang[ 27 ][ 16 ] = 7; ang[ 27 ][ 17 ] = 11; ang[ 27 ][ 18 ] = 8; ang[ 27 ][ 19 ] = 0; 
ang[ 28 ][ 0 ] = 18; ang[ 28 ][ 1 ] = 6; ang[ 28 ][ 2 ] = 12; ang[ 28 ][ 3 ] = 8; ang[ 28 ][ 4 ] = 11; ang[ 28 ][ 5 ] = 15; ang[ 28 ][ 6 ] = 1; ang[ 28 ][ 7 ] = 17; ang[ 28 ][ 8 ] = 3; ang[ 28 ][ 9 ] = 14; ang[ 28 ][ 10 ] = 13; ang[ 28 ][ 11 ] = 4; ang[ 28 ][ 12 ] = 2; ang[ 28 ][ 13 ] = 10; ang[ 28 ][ 14 ] = 9; ang[ 28 ][ 15 ] = 5; ang[ 28 ][ 16 ] = 19; ang[ 28 ][ 17 ] = 7; ang[ 28 ][ 18 ] = 0; ang[ 28 ][ 19 ] = 16; 
ang[ 29 ][ 0 ] = 7; ang[ 29 ][ 1 ] = 14; ang[ 29 ][ 2 ] = 18; ang[ 29 ][ 3 ] = 10; ang[ 29 ][ 4 ] = 9; ang[ 29 ][ 5 ] = 17; ang[ 29 ][ 6 ] = 13; ang[ 29 ][ 7 ] = 0; ang[ 29 ][ 8 ] = 11; ang[ 29 ][ 9 ] = 4; ang[ 29 ][ 10 ] = 3; ang[ 29 ][ 11 ] = 8; ang[ 29 ][ 12 ] = 19; ang[ 29 ][ 13 ] = 6; ang[ 29 ][ 14 ] = 1; ang[ 29 ][ 15 ] = 16; ang[ 29 ][ 16 ] = 15; ang[ 29 ][ 17 ] = 5; ang[ 29 ][ 18 ] = 2; ang[ 29 ][ 19 ] = 12; 
ang[ 30 ][ 0 ] = 14; ang[ 30 ][ 1 ] = 19; ang[ 30 ][ 2 ] = 5; ang[ 30 ][ 3 ] = 11; ang[ 30 ][ 4 ] = 8; ang[ 30 ][ 5 ] = 2; ang[ 30 ][ 6 ] = 16; ang[ 30 ][ 7 ] = 13; ang[ 30 ][ 8 ] = 4; ang[ 30 ][ 9 ] = 18; ang[ 30 ][ 10 ] = 17; ang[ 30 ][ 11 ] = 3; ang[ 30 ][ 12 ] = 15; ang[ 30 ][ 13 ] = 7; ang[ 30 ][ 14 ] = 0; ang[ 30 ][ 15 ] = 12; ang[ 30 ][ 16 ] = 6; ang[ 30 ][ 17 ] = 10; ang[ 30 ][ 18 ] = 9; ang[ 30 ][ 19 ] = 1; 
ang[ 31 ][ 0 ] = 7; ang[ 31 ][ 1 ] = 18; ang[ 31 ][ 2 ] = 9; ang[ 31 ][ 3 ] = 13; ang[ 31 ][ 4 ] = 14; ang[ 31 ][ 5 ] = 10; ang[ 31 ][ 6 ] = 17; ang[ 31 ][ 7 ] = 0; ang[ 31 ][ 8 ] = 15; ang[ 31 ][ 9 ] = 2; ang[ 31 ][ 10 ] = 5; ang[ 31 ][ 11 ] = 12; ang[ 31 ][ 12 ] = 11; ang[ 31 ][ 13 ] = 3; ang[ 31 ][ 14 ] = 4; ang[ 31 ][ 15 ] = 8; ang[ 31 ][ 16 ] = 19; ang[ 31 ][ 17 ] = 6; ang[ 31 ][ 18 ] = 1; ang[ 31 ][ 19 ] = 16; 
ang[ 32 ][ 0 ] = 15; ang[ 32 ][ 1 ] = 6; ang[ 32 ][ 2 ] = 11; ang[ 32 ][ 3 ] = 18; ang[ 32 ][ 4 ] = 17; ang[ 32 ][ 5 ] = 8; ang[ 32 ][ 6 ] = 1; ang[ 32 ][ 7 ] = 12; ang[ 32 ][ 8 ] = 5; ang[ 32 ][ 9 ] = 10; ang[ 32 ][ 10 ] = 9; ang[ 32 ][ 11 ] = 2; ang[ 32 ][ 12 ] = 7; ang[ 32 ][ 13 ] = 19; ang[ 32 ][ 14 ] = 16; ang[ 32 ][ 15 ] = 0; ang[ 32 ][ 16 ] = 14; ang[ 32 ][ 17 ] = 4; ang[ 32 ][ 18 ] = 3; ang[ 32 ][ 19 ] = 13; 
ang[ 33 ][ 0 ] = 6; ang[ 33 ][ 1 ] = 7; ang[ 33 ][ 2 ] = 4; ang[ 33 ][ 3 ] = 5; ang[ 33 ][ 4 ] = 2; ang[ 33 ][ 5 ] = 3; ang[ 33 ][ 6 ] = 0; ang[ 33 ][ 7 ] = 1; ang[ 33 ][ 8 ] = 10; ang[ 33 ][ 9 ] = 11; ang[ 33 ][ 10 ] = 8; ang[ 33 ][ 11 ] = 9; ang[ 33 ][ 12 ] = 14; ang[ 33 ][ 13 ] = 15; ang[ 33 ][ 14 ] = 12; ang[ 33 ][ 15 ] = 13; ang[ 33 ][ 16 ] = 19; ang[ 33 ][ 17 ] = 18; ang[ 33 ][ 18 ] = 17; ang[ 33 ][ 19 ] = 16; 
ang[ 34 ][ 0 ] = 3; ang[ 34 ][ 1 ] = 2; ang[ 34 ][ 2 ] = 1; ang[ 34 ][ 3 ] = 0; ang[ 34 ][ 4 ] = 7; ang[ 34 ][ 5 ] = 6; ang[ 34 ][ 6 ] = 5; ang[ 34 ][ 7 ] = 4; ang[ 34 ][ 8 ] = 11; ang[ 34 ][ 9 ] = 10; ang[ 34 ][ 10 ] = 9; ang[ 34 ][ 11 ] = 8; ang[ 34 ][ 12 ] = 13; ang[ 34 ][ 13 ] = 12; ang[ 34 ][ 14 ] = 15; ang[ 34 ][ 15 ] = 14; ang[ 34 ][ 16 ] = 18; ang[ 34 ][ 17 ] = 19; ang[ 34 ][ 18 ] = 16; ang[ 34 ][ 19 ] = 17; 
ang[ 35 ][ 0 ] = 5; ang[ 35 ][ 1 ] = 4; ang[ 35 ][ 2 ] = 7; ang[ 35 ][ 3 ] = 6; ang[ 35 ][ 4 ] = 1; ang[ 35 ][ 5 ] = 0; ang[ 35 ][ 6 ] = 3; ang[ 35 ][ 7 ] = 2; ang[ 35 ][ 8 ] = 9; ang[ 35 ][ 9 ] = 8; ang[ 35 ][ 10 ] = 11; ang[ 35 ][ 11 ] = 10; ang[ 35 ][ 12 ] = 15; ang[ 35 ][ 13 ] = 14; ang[ 35 ][ 14 ] = 13; ang[ 35 ][ 15 ] = 12; ang[ 35 ][ 16 ] = 17; ang[ 35 ][ 17 ] = 16; ang[ 35 ][ 18 ] = 19; ang[ 35 ][ 19 ] = 18; 
ang[ 36 ][ 0 ] = 8; ang[ 36 ][ 1 ] = 17; ang[ 36 ][ 2 ] = 12; ang[ 36 ][ 3 ] = 1; ang[ 36 ][ 4 ] = 6; ang[ 36 ][ 5 ] = 15; ang[ 36 ][ 6 ] = 18; ang[ 36 ][ 7 ] = 11; ang[ 36 ][ 8 ] = 10; ang[ 36 ][ 9 ] = 5; ang[ 36 ][ 10 ] = 2; ang[ 36 ][ 11 ] = 9; ang[ 36 ][ 12 ] = 0; ang[ 36 ][ 13 ] = 16; ang[ 36 ][ 14 ] = 19; ang[ 36 ][ 15 ] = 7; ang[ 36 ][ 16 ] = 4; ang[ 36 ][ 17 ] = 14; ang[ 36 ][ 18 ] = 13; ang[ 36 ][ 19 ] = 3; 
ang[ 37 ][ 0 ] = 10; ang[ 37 ][ 1 ] = 14; ang[ 37 ][ 2 ] = 0; ang[ 37 ][ 3 ] = 17; ang[ 37 ][ 4 ] = 18; ang[ 37 ][ 5 ] = 7; ang[ 37 ][ 6 ] = 13; ang[ 37 ][ 7 ] = 9; ang[ 37 ][ 8 ] = 2; ang[ 37 ][ 9 ] = 15; ang[ 37 ][ 10 ] = 12; ang[ 37 ][ 11 ] = 5; ang[ 37 ][ 12 ] = 8; ang[ 37 ][ 13 ] = 4; ang[ 37 ][ 14 ] = 3; ang[ 37 ][ 15 ] = 11; ang[ 37 ][ 16 ] = 6; ang[ 37 ][ 17 ] = 19; ang[ 37 ][ 18 ] = 16; ang[ 37 ][ 19 ] = 1; 
ang[ 38 ][ 0 ] = 2; ang[ 38 ][ 1 ] = 19; ang[ 38 ][ 2 ] = 8; ang[ 38 ][ 3 ] = 14; ang[ 38 ][ 4 ] = 13; ang[ 38 ][ 5 ] = 11; ang[ 38 ][ 6 ] = 16; ang[ 38 ][ 7 ] = 5; ang[ 38 ][ 8 ] = 12; ang[ 38 ][ 9 ] = 7; ang[ 38 ][ 10 ] = 0; ang[ 38 ][ 11 ] = 15; ang[ 38 ][ 12 ] = 10; ang[ 38 ][ 13 ] = 6; ang[ 38 ][ 14 ] = 1; ang[ 38 ][ 15 ] = 9; ang[ 38 ][ 16 ] = 18; ang[ 38 ][ 17 ] = 3; ang[ 38 ][ 18 ] = 4; ang[ 38 ][ 19 ] = 17; 
ang[ 39 ][ 0 ] = 12; ang[ 39 ][ 1 ] = 3; ang[ 39 ][ 2 ] = 10; ang[ 39 ][ 3 ] = 19; ang[ 39 ][ 4 ] = 16; ang[ 39 ][ 5 ] = 9; ang[ 39 ][ 6 ] = 4; ang[ 39 ][ 7 ] = 15; ang[ 39 ][ 8 ] = 0; ang[ 39 ][ 9 ] = 11; ang[ 39 ][ 10 ] = 8; ang[ 39 ][ 11 ] = 7; ang[ 39 ][ 12 ] = 2; ang[ 39 ][ 13 ] = 18; ang[ 39 ][ 14 ] = 17; ang[ 39 ][ 15 ] = 5; ang[ 39 ][ 16 ] = 13; ang[ 39 ][ 17 ] = 1; ang[ 39 ][ 18 ] = 6; ang[ 39 ][ 19 ] = 14; 
ang[ 40 ][ 0 ] = 12; ang[ 40 ][ 1 ] = 16; ang[ 40 ][ 2 ] = 3; ang[ 40 ][ 3 ] = 9; ang[ 40 ][ 4 ] = 10; ang[ 40 ][ 5 ] = 4; ang[ 40 ][ 6 ] = 19; ang[ 40 ][ 7 ] = 15; ang[ 40 ][ 8 ] = 2; ang[ 40 ][ 9 ] = 17; ang[ 40 ][ 10 ] = 18; ang[ 40 ][ 11 ] = 5; ang[ 40 ][ 12 ] = 13; ang[ 40 ][ 13 ] = 1; ang[ 40 ][ 14 ] = 6; ang[ 40 ][ 15 ] = 14; ang[ 40 ][ 16 ] = 0; ang[ 40 ][ 17 ] = 8; ang[ 40 ][ 18 ] = 11; ang[ 40 ][ 19 ] = 7; 
ang[ 41 ][ 0 ] = 13; ang[ 41 ][ 1 ] = 0; ang[ 41 ][ 2 ] = 9; ang[ 41 ][ 3 ] = 17; ang[ 41 ][ 4 ] = 18; ang[ 41 ][ 5 ] = 10; ang[ 41 ][ 6 ] = 7; ang[ 41 ][ 7 ] = 14; ang[ 41 ][ 8 ] = 3; ang[ 41 ][ 9 ] = 8; ang[ 41 ][ 10 ] = 11; ang[ 41 ][ 11 ] = 4; ang[ 41 ][ 12 ] = 1; ang[ 41 ][ 13 ] = 16; ang[ 41 ][ 14 ] = 19; ang[ 41 ][ 15 ] = 6; ang[ 41 ][ 16 ] = 12; ang[ 41 ][ 17 ] = 2; ang[ 41 ][ 18 ] = 5; ang[ 41 ][ 19 ] = 15; 
ang[ 42 ][ 0 ] = 1; ang[ 42 ][ 1 ] = 12; ang[ 42 ][ 2 ] = 17; ang[ 42 ][ 3 ] = 8; ang[ 42 ][ 4 ] = 11; ang[ 42 ][ 5 ] = 18; ang[ 42 ][ 6 ] = 15; ang[ 42 ][ 7 ] = 6; ang[ 42 ][ 8 ] = 9; ang[ 42 ][ 9 ] = 2; ang[ 42 ][ 10 ] = 5; ang[ 42 ][ 11 ] = 10; ang[ 42 ][ 12 ] = 16; ang[ 42 ][ 13 ] = 0; ang[ 42 ][ 14 ] = 7; ang[ 42 ][ 15 ] = 19; ang[ 42 ][ 16 ] = 13; ang[ 42 ][ 17 ] = 3; ang[ 42 ][ 18 ] = 4; ang[ 42 ][ 19 ] = 14; 
ang[ 43 ][ 0 ] = 16; ang[ 43 ][ 1 ] = 13; ang[ 43 ][ 2 ] = 8; ang[ 43 ][ 3 ] = 2; ang[ 43 ][ 4 ] = 5; ang[ 43 ][ 5 ] = 11; ang[ 43 ][ 6 ] = 14; ang[ 43 ][ 7 ] = 19; ang[ 43 ][ 8 ] = 17; ang[ 43 ][ 9 ] = 3; ang[ 43 ][ 10 ] = 4; ang[ 43 ][ 11 ] = 18; ang[ 43 ][ 12 ] = 0; ang[ 43 ][ 13 ] = 12; ang[ 43 ][ 14 ] = 15; ang[ 43 ][ 15 ] = 7; ang[ 43 ][ 16 ] = 1; ang[ 43 ][ 17 ] = 9; ang[ 43 ][ 18 ] = 10; ang[ 43 ][ 19 ] = 6; 
ang[ 44 ][ 0 ] = 16; ang[ 44 ][ 1 ] = 5; ang[ 44 ][ 2 ] = 13; ang[ 44 ][ 3 ] = 11; ang[ 44 ][ 4 ] = 8; ang[ 44 ][ 5 ] = 14; ang[ 44 ][ 6 ] = 2; ang[ 44 ][ 7 ] = 19; ang[ 44 ][ 8 ] = 0; ang[ 44 ][ 9 ] = 15; ang[ 44 ][ 10 ] = 12; ang[ 44 ][ 11 ] = 7; ang[ 44 ][ 12 ] = 1; ang[ 44 ][ 13 ] = 9; ang[ 44 ][ 14 ] = 10; ang[ 44 ][ 15 ] = 6; ang[ 44 ][ 16 ] = 17; ang[ 44 ][ 17 ] = 4; ang[ 44 ][ 18 ] = 3; ang[ 44 ][ 19 ] = 18; 
ang[ 45 ][ 0 ] = 17; ang[ 45 ][ 1 ] = 14; ang[ 45 ][ 2 ] = 9; ang[ 45 ][ 3 ] = 7; ang[ 45 ][ 4 ] = 0; ang[ 45 ][ 5 ] = 10; ang[ 45 ][ 6 ] = 13; ang[ 45 ][ 7 ] = 18; ang[ 45 ][ 8 ] = 16; ang[ 45 ][ 9 ] = 6; ang[ 45 ][ 10 ] = 1; ang[ 45 ][ 11 ] = 19; ang[ 45 ][ 12 ] = 5; ang[ 45 ][ 13 ] = 15; ang[ 45 ][ 14 ] = 12; ang[ 45 ][ 15 ] = 2; ang[ 45 ][ 16 ] = 4; ang[ 45 ][ 17 ] = 8; ang[ 45 ][ 18 ] = 11; ang[ 45 ][ 19 ] = 3; 
ang[ 46 ][ 0 ] = 4; ang[ 46 ][ 1 ] = 10; ang[ 46 ][ 2 ] = 15; ang[ 46 ][ 3 ] = 19; ang[ 46 ][ 4 ] = 16; ang[ 46 ][ 5 ] = 12; ang[ 46 ][ 6 ] = 9; ang[ 46 ][ 7 ] = 3; ang[ 46 ][ 8 ] = 17; ang[ 46 ][ 9 ] = 2; ang[ 46 ][ 10 ] = 5; ang[ 46 ][ 11 ] = 18; ang[ 46 ][ 12 ] = 14; ang[ 46 ][ 13 ] = 6; ang[ 46 ][ 14 ] = 1; ang[ 46 ][ 15 ] = 13; ang[ 46 ][ 16 ] = 8; ang[ 46 ][ 17 ] = 0; ang[ 46 ][ 18 ] = 7; ang[ 46 ][ 19 ] = 11; 
ang[ 47 ][ 0 ] = 8; ang[ 47 ][ 1 ] = 12; ang[ 47 ][ 2 ] = 6; ang[ 47 ][ 3 ] = 18; ang[ 47 ][ 4 ] = 17; ang[ 47 ][ 5 ] = 1; ang[ 47 ][ 6 ] = 15; ang[ 47 ][ 7 ] = 11; ang[ 47 ][ 8 ] = 4; ang[ 47 ][ 9 ] = 13; ang[ 47 ][ 10 ] = 14; ang[ 47 ][ 11 ] = 3; ang[ 47 ][ 12 ] = 10; ang[ 47 ][ 13 ] = 2; ang[ 47 ][ 14 ] = 5; ang[ 47 ][ 15 ] = 9; ang[ 47 ][ 16 ] = 0; ang[ 47 ][ 17 ] = 16; ang[ 47 ][ 18 ] = 19; ang[ 47 ][ 19 ] = 7; 
ang[ 48 ][ 0 ] = 13; ang[ 48 ][ 1 ] = 9; ang[ 48 ][ 2 ] = 18; ang[ 48 ][ 3 ] = 7; ang[ 48 ][ 4 ] = 0; ang[ 48 ][ 5 ] = 17; ang[ 48 ][ 6 ] = 10; ang[ 48 ][ 7 ] = 14; ang[ 48 ][ 8 ] = 12; ang[ 48 ][ 9 ] = 5; ang[ 48 ][ 10 ] = 2; ang[ 48 ][ 11 ] = 15; ang[ 48 ][ 12 ] = 3; ang[ 48 ][ 13 ] = 11; ang[ 48 ][ 14 ] = 8; ang[ 48 ][ 15 ] = 4; ang[ 48 ][ 16 ] = 1; ang[ 48 ][ 17 ] = 16; ang[ 48 ][ 18 ] = 19; ang[ 48 ][ 19 ] = 6; 
ang[ 49 ][ 0 ] = 11; ang[ 49 ][ 1 ] = 5; ang[ 49 ][ 2 ] = 19; ang[ 49 ][ 3 ] = 14; ang[ 49 ][ 4 ] = 13; ang[ 49 ][ 5 ] = 16; ang[ 49 ][ 6 ] = 2; ang[ 49 ][ 7 ] = 8; ang[ 49 ][ 8 ] = 3; ang[ 49 ][ 9 ] = 17; ang[ 49 ][ 10 ] = 18; ang[ 49 ][ 11 ] = 4; ang[ 49 ][ 12 ] = 7; ang[ 49 ][ 13 ] = 15; ang[ 49 ][ 14 ] = 12; ang[ 49 ][ 15 ] = 0; ang[ 49 ][ 16 ] = 9; ang[ 49 ][ 17 ] = 1; ang[ 49 ][ 18 ] = 6; ang[ 49 ][ 19 ] = 10; 
ang[ 50 ][ 0 ] = 15; ang[ 50 ][ 1 ] = 17; ang[ 50 ][ 2 ] = 6; ang[ 50 ][ 3 ] = 8; ang[ 50 ][ 4 ] = 11; ang[ 50 ][ 5 ] = 1; ang[ 50 ][ 6 ] = 18; ang[ 50 ][ 7 ] = 12; ang[ 50 ][ 8 ] = 7; ang[ 50 ][ 9 ] = 16; ang[ 50 ][ 10 ] = 19; ang[ 50 ][ 11 ] = 0; ang[ 50 ][ 12 ] = 14; ang[ 50 ][ 13 ] = 4; ang[ 50 ][ 14 ] = 3; ang[ 50 ][ 15 ] = 13; ang[ 50 ][ 16 ] = 5; ang[ 50 ][ 17 ] = 9; ang[ 50 ][ 18 ] = 10; ang[ 50 ][ 19 ] = 2; 
ang[ 51 ][ 0 ] = 4; ang[ 51 ][ 1 ] = 16; ang[ 51 ][ 2 ] = 10; ang[ 51 ][ 3 ] = 12; ang[ 51 ][ 4 ] = 15; ang[ 51 ][ 5 ] = 9; ang[ 51 ][ 6 ] = 19; ang[ 51 ][ 7 ] = 3; ang[ 51 ][ 8 ] = 14; ang[ 51 ][ 9 ] = 1; ang[ 51 ][ 10 ] = 6; ang[ 51 ][ 11 ] = 13; ang[ 51 ][ 12 ] = 8; ang[ 51 ][ 13 ] = 0; ang[ 51 ][ 14 ] = 7; ang[ 51 ][ 15 ] = 11; ang[ 51 ][ 16 ] = 17; ang[ 51 ][ 17 ] = 5; ang[ 51 ][ 18 ] = 2; ang[ 51 ][ 19 ] = 18; 
ang[ 52 ][ 0 ] = 10; ang[ 52 ][ 1 ] = 0; ang[ 52 ][ 2 ] = 18; ang[ 52 ][ 3 ] = 13; ang[ 52 ][ 4 ] = 14; ang[ 52 ][ 5 ] = 17; ang[ 52 ][ 6 ] = 7; ang[ 52 ][ 7 ] = 9; ang[ 52 ][ 8 ] = 6; ang[ 52 ][ 9 ] = 16; ang[ 52 ][ 10 ] = 19; ang[ 52 ][ 11 ] = 1; ang[ 52 ][ 12 ] = 2; ang[ 52 ][ 13 ] = 12; ang[ 52 ][ 14 ] = 15; ang[ 52 ][ 15 ] = 5; ang[ 52 ][ 16 ] = 8; ang[ 52 ][ 17 ] = 4; ang[ 52 ][ 18 ] = 3; ang[ 52 ][ 19 ] = 11; 
ang[ 53 ][ 0 ] = 19; ang[ 53 ][ 1 ] = 10; ang[ 53 ][ 2 ] = 3; ang[ 53 ][ 3 ] = 12; ang[ 53 ][ 4 ] = 15; ang[ 53 ][ 5 ] = 4; ang[ 53 ][ 6 ] = 9; ang[ 53 ][ 7 ] = 16; ang[ 53 ][ 8 ] = 7; ang[ 53 ][ 9 ] = 8; ang[ 53 ][ 10 ] = 11; ang[ 53 ][ 11 ] = 0; ang[ 53 ][ 12 ] = 18; ang[ 53 ][ 13 ] = 2; ang[ 53 ][ 14 ] = 5; ang[ 53 ][ 15 ] = 17; ang[ 53 ][ 16 ] = 6; ang[ 53 ][ 17 ] = 14; ang[ 53 ][ 18 ] = 13; ang[ 53 ][ 19 ] = 1; 
ang[ 54 ][ 0 ] = 11; ang[ 54 ][ 1 ] = 19; ang[ 54 ][ 2 ] = 13; ang[ 54 ][ 3 ] = 2; ang[ 54 ][ 4 ] = 5; ang[ 54 ][ 5 ] = 14; ang[ 54 ][ 6 ] = 16; ang[ 54 ][ 7 ] = 8; ang[ 54 ][ 8 ] = 9; ang[ 54 ][ 9 ] = 6; ang[ 54 ][ 10 ] = 1; ang[ 54 ][ 11 ] = 10; ang[ 54 ][ 12 ] = 3; ang[ 54 ][ 13 ] = 18; ang[ 54 ][ 14 ] = 17; ang[ 54 ][ 15 ] = 4; ang[ 54 ][ 16 ] = 7; ang[ 54 ][ 17 ] = 15; ang[ 54 ][ 18 ] = 12; ang[ 54 ][ 19 ] = 0; 
ang[ 55 ][ 0 ] = 1; ang[ 55 ][ 1 ] = 11; ang[ 55 ][ 2 ] = 12; ang[ 55 ][ 3 ] = 18; ang[ 55 ][ 4 ] = 17; ang[ 55 ][ 5 ] = 15; ang[ 55 ][ 6 ] = 8; ang[ 55 ][ 7 ] = 6; ang[ 55 ][ 8 ] = 16; ang[ 55 ][ 9 ] = 7; ang[ 55 ][ 10 ] = 0; ang[ 55 ][ 11 ] = 19; ang[ 55 ][ 12 ] = 13; ang[ 55 ][ 13 ] = 3; ang[ 55 ][ 14 ] = 4; ang[ 55 ][ 15 ] = 14; ang[ 55 ][ 16 ] = 9; ang[ 55 ][ 17 ] = 5; ang[ 55 ][ 18 ] = 2; ang[ 55 ][ 19 ] = 10; 
ang[ 56 ][ 0 ] = 17; ang[ 56 ][ 1 ] = 9; ang[ 56 ][ 2 ] = 0; ang[ 56 ][ 3 ] = 13; ang[ 56 ][ 4 ] = 14; ang[ 56 ][ 5 ] = 7; ang[ 56 ][ 6 ] = 10; ang[ 56 ][ 7 ] = 18; ang[ 56 ][ 8 ] = 4; ang[ 56 ][ 9 ] = 11; ang[ 56 ][ 10 ] = 8; ang[ 56 ][ 11 ] = 3; ang[ 56 ][ 12 ] = 16; ang[ 56 ][ 13 ] = 1; ang[ 56 ][ 14 ] = 6; ang[ 56 ][ 15 ] = 19; ang[ 56 ][ 16 ] = 5; ang[ 56 ][ 17 ] = 15; ang[ 56 ][ 18 ] = 12; ang[ 56 ][ 19 ] = 2; 
ang[ 57 ][ 0 ] = 15; ang[ 57 ][ 1 ] = 11; ang[ 57 ][ 2 ] = 17; ang[ 57 ][ 3 ] = 1; ang[ 57 ][ 4 ] = 6; ang[ 57 ][ 5 ] = 18; ang[ 57 ][ 6 ] = 8; ang[ 57 ][ 7 ] = 12; ang[ 57 ][ 8 ] = 14; ang[ 57 ][ 9 ] = 3; ang[ 57 ][ 10 ] = 4; ang[ 57 ][ 11 ] = 13; ang[ 57 ][ 12 ] = 5; ang[ 57 ][ 13 ] = 9; ang[ 57 ][ 14 ] = 10; ang[ 57 ][ 15 ] = 2; ang[ 57 ][ 16 ] = 7; ang[ 57 ][ 17 ] = 19; ang[ 57 ][ 18 ] = 16; ang[ 57 ][ 19 ] = 0; 
ang[ 58 ][ 0 ] = 19; ang[ 58 ][ 1 ] = 3; ang[ 58 ][ 2 ] = 15; ang[ 58 ][ 3 ] = 9; ang[ 58 ][ 4 ] = 10; ang[ 58 ][ 5 ] = 12; ang[ 58 ][ 6 ] = 4; ang[ 58 ][ 7 ] = 16; ang[ 58 ][ 8 ] = 6; ang[ 58 ][ 9 ] = 13; ang[ 58 ][ 10 ] = 14; ang[ 58 ][ 11 ] = 1; ang[ 58 ][ 12 ] = 7; ang[ 58 ][ 13 ] = 11; ang[ 58 ][ 14 ] = 8; ang[ 58 ][ 15 ] = 0; ang[ 58 ][ 16 ] = 18; ang[ 58 ][ 17 ] = 2; ang[ 58 ][ 18 ] = 5; ang[ 58 ][ 19 ] = 17; 
ang[ 59 ][ 0 ] = 2; ang[ 59 ][ 1 ] = 13; ang[ 59 ][ 2 ] = 19; ang[ 59 ][ 3 ] = 11; ang[ 59 ][ 4 ] = 8; ang[ 59 ][ 5 ] = 16; ang[ 59 ][ 6 ] = 14; ang[ 59 ][ 7 ] = 5; ang[ 59 ][ 8 ] = 10; ang[ 59 ][ 9 ] = 1; ang[ 59 ][ 10 ] = 6; ang[ 59 ][ 11 ] = 9; ang[ 59 ][ 12 ] = 18; ang[ 59 ][ 13 ] = 3; ang[ 59 ][ 14 ] = 4; ang[ 59 ][ 15 ] = 17; ang[ 59 ][ 16 ] = 12; ang[ 59 ][ 17 ] = 0; ang[ 59 ][ 18 ] = 7; ang[ 59 ][ 19 ] = 15; 


caffe_root = '/opt/caffe/caffe-rotationnet2/'  # Change this to your path.
sys.path.insert(0, caffe_root + 'python')

import caffe

fname =  'classes.txt'
f = open(fname)
classes = f.readlines()
f.close()
classes = [f[:-1] for f in classes]


def main(argv):
    pycaffe_dir = caffe_root + 'python/'

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "--input_file",
        default ="/data/converted/testrotnet.txt",
        help="text file containg the image paths"
    )
    
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default="./Training/rotationnet_modelnet40_case1_solver.prototxt",
        help="Model definition file."
    )
    parser.add_argument('--weights', type=int, default=-1)
    parser.add_argument('--views', type=int, default=12)
    parser.add_argument('--log_dir', default='logs', type=str)    
    
    parser.add_argument(
        "--center_only",
        action='store_true',
        default = False,
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--images_dim",
        default='227,227',
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(caffe_root,
                             'data/ilsvrc12/imagenet_mean.binaryproto'),
        help="Data set image mean of H x W x K dimensions (np array). " +
             "Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        default=255,
        help="Multiply input features by this scale before input to net"
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."

    )

    args = parser.parse_args()
    
    args.pretrained_model = os.path.join(args.log_dir, 'case1_iter_'+str(args.weights) + '.caffemodel')
    
    image_dims = [int(s) for s in args.images_dim.split(',')]
    channel_swap = [int(s) for s in args.channel_swap.split(',')]
    
    
    if args.mean_file:
        mean = get_mean(args.mean_file)

    caffe.set_mode_gpu()
        
    # Make classifier.
    classifier = caffe.Classifier(args.model_def, args.pretrained_model,
            image_dims=image_dims, mean=mean,
            input_scale=1.0, raw_scale=255.0, channel_swap=channel_swap)


    listfiles, labels = read_lists(args.input_file)

    #dataset = Dataset(listfiles, labels, subtract_mean=False, V=20)
    # Load image file.
    args.input_file = os.path.expanduser(args.input_file)
    
    preds = []
    labels = [int(label) for label in labels]
    
    total = len(listfiles)
    
    views = args.views
    batch = 8 * views
    for i in range( len(listfiles) / (batch*views)):

        #im_files = [line.rstrip('\n') for line in open(listfiles[views*i+j])]
        im_files = listfiles[i*batch*views : (i+1)*batch*views]
        
        #labels.append(int(im_files[0]))
        #im_files = im_files[2:]
        inputs= [caffe.io.load_image(im_f) for im_f in im_files]

        predictions = classifier.predict(inputs, not args.center_only)
        classified = classify(predictions)
        preds.append(classified)
        print(classified)
    
    import Evaluation_tools as et
    data = '/data'
    logs = '/logs'
    eval_file = os.path.join(logs, 'rotnet.txt')
    et.write_eval_file(data, eval_file, preds, labels, 'ROTNET')
    et.make_matrix(data, eval_file, logs)

def read_lists(list_of_lists_file):
    listfile_labels = np.loadtxt(list_of_lists_file, dtype=str).tolist()
    listfiles, labels  = zip(*[(l[0], int(l[1])) for l in listfile_labels])
    return listfiles, labels

def get_mean(mean_file):
    image_dims = [227,227]
    channel_swap = [2,1,0]
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(mean_file , 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob))
    mean = arr[0]
    # Resize mean (which requires H x W x K input in range [0,1]).
    in_shape = image_dims
    m_min, m_max = mean.min(), mean.max()
    normal_mean = (mean - m_min) / (m_max - m_min)
    mean = caffe.io.resize_image(normal_mean.transpose((1,2,0)),
                                 in_shape).transpose((2,0,1)) * (m_max - m_min) + m_min  
    return mean
    

def classify(prediction, alligned = True):
    
    clsnum = len(classes)
    clsid = -1
    classified  = []
    
    scores = prediction
    numR = scores.shape[1]/ (clsnum+1)
    nsamp = len(scores) / numR
    
    for i in range(0,len(scores)):
        for j in range(0,numR):
            for k in range(0,clsnum):
                scores[i][ j * (clsnum+1) + k ] = scores[i][ j * (clsnum+1) + k ] / scores[i][ j * (clsnum+1) + clsnum ]
            scores[i][ j * (clsnum+1) + clsnum ] = 0
    clsnum = (clsnum+1)
    
    if alligned:
        for n in range(nsamp):
            s = np.ones(clsnum*numR)
            for i in range(numR):
                for j in range(clsnum):
                    for k in range(numR):
                        idx = i + k
                        if idx > (numR-1):
                            idx = idx - numR
                        s[ i * clsnum + j ] = s[ i * clsnum + j ] * scores[ n * numR + k ][ idx * clsnum + j ]
            classified.append(np.argmax( s ) % clsnum)
        
    else:
        for n in range(nsamp):
            s = np.ones(clsnum*ang.shape[0])
            for i in range(ang.shape[0]):
                for j in range(clsnum):
                    for k in range(numR):
                        s[ i * clsnum + j ] = s[ i * clsnum + j ] * scores[ n * numR + ang[ i ][ k ] ][ k * clsnum + j ]
            classified.append(np.argmax( s ) % clsnum)
        
    return classified



if __name__ == '__main__':
    main(sys.argv)


