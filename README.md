# Gravity
Repository dedicated to the study of the motion of bodies under the Newtonian gravitational interaction.

## Usage 
In order to use this project, clone the repository by copying the associated URL. Make sure you have installed the required packages with the proper versions: `pip install -r requirements.txt`.
You will find a few scripts, most of which generate an animation. It is as simple as hitting `Run`, although in some of them imput prompts will appear in the terminal, that you must answer. Try and play with multiple parameters and initial conditions. Have fun! 

## Contents
* `solar_system.py`: computation of a particle orbit with a root finding method in the two body approximation. Application to the orbits of planets in the SS.
* `N_body.py`: numerical integration and simulation of the motion of N gravitationally-interacting particles.
* `runge_vs_symp.py`: showing the greatly improved performance of numerical integration from a symplectic algorithm against Runge-Kutta schemes, as well as their differences regarding Conservation of Energy.
* `mysimpint`: Verlet (symplectic) scheme for integration.

More content and some great updates are yet to come... Keep track!

