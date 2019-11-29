# Gray-Scott-PDE
### David Augustin, Jakke Neiro, Thijs van der Plas
### 29 November 2019

## Gray-Scott equations:

The Gray-Scott equations originate from the class of reaction-diffusion equations, and depending or their two-dimensional parameter settings <img src="https://tex.s2cms.ru/svg/(F%2C%20k)" alt="(F, k)" /> they can model a variety of patterns, such as spots, stripes, maze formations, ripples et cetera [1]. The Gray-Scott equations model the following chemical reaction: 

<img src="https://tex.s2cms.ru/svg/U%20%20%2B%202V%20%5Cto%203V" alt="U  + 2V \to 3V" />

<img src="https://tex.s2cms.ru/svg/%20V%20%5Cto%20P%20" alt=" V \to P " />

This is modelled by the following coupled Partial Differential Equations (PDEs), consisting of a diffusion term, interaction term, negative <img src="https://tex.s2cms.ru/svg/U" alt="U" /> and <img src="https://tex.s2cms.ru/svg/V" alt="V" /> feeds and positive U feed [2]:

<img src="https://tex.s2cms.ru/svg/%20%5Cfrac%7B%5Cdelta%20U%7D%7B%5Cdelta%20t%7D%20%3D%20%5Cepsilon_1%20%5CDelta%20U%20-%20UV%5E2%20%2B%20F(1%20-%20U)%20" alt=" \frac{\delta U}{\delta t} = \epsilon_1 \Delta U - UV^2 + F(1 - U) " />

<img src="https://tex.s2cms.ru/svg/%20%5Cfrac%7B%5Cdelta%20V%7D%7B%5Cdelta%20t%7D%20%3D%20%5Cepsilon_2%20%5CDelta%20V%20%2B%20UV%5E2%20-%20(F%20%2B%20k)V%20%20" alt=" \frac{\delta V}{\delta t} = \epsilon_2 \Delta V + UV^2 - (F + k)V  " />

Here, <img src="https://tex.s2cms.ru/svg/U" alt="U" /> and <img src="https://tex.s2cms.ru/svg/V" alt="V" /> are two-dimensional in space (<img src="https://tex.s2cms.ru/svg/x%2C%20y" alt="x, y" />), and <img src="https://tex.s2cms.ru/svg/%5CDelta" alt="\Delta" /> denotes their laplacian operator. Here, we numerically solve these equations by approximating the time derivative by finite difference methods. The time derivative and laplacian are approximated by a forward difference and central difference scheme respectively [3]. 

Without the interaction and feed terms, the PDEs collapse to regular (uncoupled) heat equations. This case is also considered in our code as a test case.

[1] Trefethen, Nick "The (Unfinished) PDE Coffee Table Book".   http://people.maths.ox.ac.uk/trefethen/pdectb/reaction2.pdf

[2] Pearson, John E. "Complex patterns in a simple system." Science 261.5118 (1993): 189-192.

[3] Recktenwald, Gerald W. "Finite-difference approximations to the heat equation." Mechanical Engineering 10 (2004): 1-27. Updated in 2011.

## Installation

Clone repository by opening the terminal and typing

```
git clone https://github.com/SABS-R3-projects/Gray-Scott-PDE/
```

Then install the grayscott module and all dependencies with
```
pip install .
```

Congratulations! You successfully installed grayscott.


## Executing 
All the code can be run by executing the jupyter notebook `Main.ipynb`. The Jupyter interface is initiated by running `jupyter notebook` in your terminal. 



This project is release under a BSD 3-Clause License.

Enjoy!

![picture](/figures/u_matrix_F=0.035_k=0.06.gif)
