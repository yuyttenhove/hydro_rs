# hydro_rs
Moving mesh hydro dynamics code for testing purposes.
Specifically we want to gain insight in the impact of the order of operations on the accuracy.

## Main takeaways so far:
1. saving the time extrapolations (which are calculated during the drift) in a seperate field and only using them in the flux exchange together with the space extrapolations,
allows the space extrapolations to be calculated independently and alows the total extrapolation to be pair wise limited.
This proved crucial to avoid overshoots and guarantee the stability (otherwise the time extrapolations are not limited at all).
2. The _old_ order of operations (mesh, primitives, gradients, fluxes) **really** is worse/less stable
3. The current way of doing things is noticeable more stable (mesh, fluxes, primitives, gradients)
4. The optimal way seems to be to do the mesh and flux calculation at the half way point during a particles timestep and the gradient calculation at the end of the timestep.
5. No big differences were observed when all particles have the same timestep. The above applies to simulations with an indiviual variable (neighbour limited) timestep.
