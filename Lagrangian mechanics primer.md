# Lagrangian mechanics primer

Comments and corrections to J. M. F. Tsang (j.m.f.tsang@cantab.net).

---

## Lagrangian, momentum and energy
The **Lagrangian** $L$ is
$$
L = \text{kinetic energy} - \text{potential energy} = T - V
$$
For a particle in one dimension with position $x$, the Lagrangian is in general $L = L(x, \dot{x}, t)$. 

Expressions for the **kinetic energy** in 3D coordinate systems:
| Coordinates | KE |
| --- | --- |
| Cartesians |  $\frac{1}{2} m \left(\dot{x}^2 + \dot{y}^2 + \dot{z}^2 \right)$ 
| Cylindricals | $\frac{1}{2} m \left( \dot{r}^2 + r^2 \dot{\theta}^2 + \dot{z}^2 \right)$ |
| Sphericals | $\frac{1}{2} m \left( \dot{r}^2 + r^2 (\dot{\theta}^2+ \dot{\phi}^2 \sin^2 \theta ) \right)$ |

The **conjugate momentum** of the coordinate $x$ is defined as
$$
p_x = \frac{\partial L}{\partial \dot{x}}
$$
Examples assuming the potential $V$ does not depend on velocities:
| Coordinates | Coordinate | Conjugate momentum |
| --- | --- | --- |
| Cartesians | $x$ | $p_x = m\dot{x}$ |
| Cylindricals | $r$ | $p_r = m\dot{r}$ |
| Cylindricals | $\theta$ | $p_\theta = mr^2\dot{\theta}$ |
Note that angular momentum $p_\theta$ is *not* a constant multiple of $\dot\theta$. 

In general if the coordinates are $(q_1, q_2, \dots)$ and the conjugate momenta are $(p_1, p_2, \dots)$ then the **energy** of the system is defined as
$$
E = \sum_i q_i p_i  - L
$$
Usually, $T$ is a quadratic function of the velocities and $V$ does not depend on the velocities. In this case we recover
$$
E = T + V
$$

## Principle of Least Action
For a system (_e.g._ particle moving in a potential) with the Lagrangian $L$, define the **action** $S$ as
$$
S = \int_{t_1}^{t_2} L\,\mathrm{d} t
$$
for a given time interval $[t_1, t_2]$. The action has units $[S] = \mathrm{J}\mathrm{s} = \mathrm{kg}\mathrm{m}^2\mathrm{s}^{-1}$. It does not necessarily have a physical meaning (see [here](https://www.overleaf.com/read/hfkwprntwzhh) for a discussion of the metaphysics).

The **principle of least action** states that amongst all possible paths $(q_i(t))$, the actual path taken by the system is one that gives extremises $S$, giving $\delta S = 0$. Equivalently, $(q_i)$ must be a solution to the Euler-Lagrange equation
$$
\frac{\mathrm{d}p_i}{\mathrm{d}t} = \frac{\partial L}{\partial q_i}
$$
This is a generalised form of Newton's second law, in curvilinear coordinates. In Cartesian coordinates, $p_i$ are linear momenta and so the RHS has the interpretation of force. In cylindrical polars, $p_\theta$ is the angular momentum about the origin and the RHS is a torque.

## First integrals and Noether's theorem
**Noether's theorem** states that
> Any continuous symmetry is associated with a corresponding conserved quantity.

A *symmetry* is a transformation that leaves $L$ unchanged; a *continuous symmetry* is one that can be parameterised smoothly (contrast discrete symmetries such as reflections, which are either on or off). If a system has a continous symmetry then it is possible to change to a coordinate system where one of the coordinates parameterises that symmetry. Then $\partial L /\partial q = 0$ for that coordinate and so $p = \partial L / \partial\dot{q}$ is conserved.

Examples:
| Coordinates | Symmetry | Conserved quantity |
| --- | --- | --- |
| Cartesians | translations in $x$ | linear momentum $p_x = m\dot{x}$ |
| Cylindricals | rotations | angular momentum $p_\theta = r^2\dot\theta$ |
| Cylindricals | translations in $z$ | $p_z = m\dot{z}$ |
| Sphericals | rotations in $\phi$ | $p_\phi$ (exercise) |
| Any | time translations | energy $E = T + V$ |

For example, in cylindricals the following conditions are equivalent:
* The potential $V$ has no explicit $\theta$ dependence.
* The force $F = -\nabla V$ has no component in the $\theta$ direction.
* The force exerts no torque on the particle about the origin.
* The angular momentum is conserved.

## Hamiltonian
The Hamiltonian $H = H(q, p, t)$ is the Legendre transform of the Lagrangian $L = L(q, \dot{q}, t)$, swapping out each of the velocities $q_i$ for the corresponding conjugate momenta $p_i$. So it is the same quantity as the energy.
$$
H = \sum_i q_i p_i - L = E
$$
However, the Hamiltonian is written in terms of momenta. When $p = m\dot{x}$ we have this form for the kinetic energy:
$$ \frac{1}{2} m \dot{x}^2 = \frac{p^2}{2m} $$
In Cartesians the distinction may seem trivial since $p \propto \dot{x}$ by a constant factor $m$. The difference is more important in cylindricals, where $p_\theta = r^2\dot{\theta}$ depends on $r$ as well as $\dot{\theta}$. This is because the partial derivative
$$
\frac{\partial H}{\partial r}
$$
means taking a derivative keeping $p_\theta$ constant, *not* keeping $\dot{\theta}$ constant. 

Hamilton's equations state that for each of the coordinates $q_i$ and its corresponding conjugate momentum $p_i$,
$$
\frac{\mathrm{d}q_i}{\mathrm{d}t} = \frac{\partial H}{\partial p_i},
\qquad
\frac{\mathrm{d}p_i}{\mathrm{d}t} = -\frac{\partial H}{\partial q_i}.
$$
(\*) The Hamiltonian itself is conserved if $\partial{H}/\partial{t} = 0$. 
