# Stabilized Stokes Flow FEniCSx

This is a repo for Stabilized Stokes Flow 2024 by Caleb Munger.

This repo contains code for FEniCSx 0.8.0.0 that takes a black and white .png image as the inlet shape, and than computes the outlet shape (example below)
post prcoessing the image to get the flow profile at the inlet and outlet is done using paraview 12.1. This code can be used to predict the shape of 
of a fluid at an outlet (assuming no mixing) for low Reynolds number exetrusion flow. 

The solver uses stabilized stokes flow to solve for the outlet profile. To run the "StokesChannelFlow.py" file, you must also download the 



<details>
<summary>Inlet Example</summary>
<br>
  This is an example of the inlet profile used
  
  ![Plus](Pics/Plus.png)
</details>

<details>
<summary>Inlet Profile Example</summary>
<br>
  This is an example of the inlet profile streamtrace
  
  ![InletShapePlus](Pics/InletShapePlus.png)
</details>

<details>
<summary>Outlet Profile Example</summary>
<br>
  This is an example of the outlet profile streamtrace 
  
  ![InletShapePlus](Pics/OutletShapePlus.png)
</details>
