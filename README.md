# 15112 CMU Term Project

This is my term project for the course [15112](http://www.kosbie.net/cmu/spring-17/15-112/index.html) at
Carnegie Mellon University during the spring semester of my freshman year - it was my first programming
class.
Here is a [video](https://www.youtube.com/watch?v=zJlUzMCfKuQ) of me demonstrating and explaining my project.


## Description and Dependencies

My project recognizes handwritten mathematical expressions - polynomials, exponents, and basic arithmetic, and helps the user to analyze them further. The user is able to edit their inputs in case it is misread, or if they simply want to analyze something that they have not scanned.
I then wrote code to parse, simplify, differentiate/integrate, and find zeros of polynomials (using bisection
methods).

To run my program, the modules that you will need are:

- pygame (pip install pygame)
- opencv (pip install opencv-python)
- numpy (pip install numpy)
- matplotlib (pip install matplotlib)
- pylab (pip install pylab)
- operator (pip install operator)
- string (included)
- os (included)
- copy (included)
- fractions (included)
- pygame_textinput (http://pygame.org/project-Pygame+Text+Input-3013-5031.html)

The main program can be found in the file `polynomialTime.py`.
