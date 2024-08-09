
#####################################################
# Ethan Poynter
#
# Quasi 1-D Flow Solution for a Subsonic - Supersonic 
# Isentropic nozzle using MacCormack's Explicit     
# Predictor-Corrector Technique     
#
# Streamlite Terminal Input:
# streamlit run Streamlit_quasi1D_CFD_Project.py --server.port 8888    
#
#####################################################

import math as m
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
icon = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/De_laval_nozzle.svg/640px-De_laval_nozzle.svg.png"

# Streamlit Set Page Config
st.set_page_config(page_title="Quasi 1D Nozzel CFD", page_layout="wide" , page_icon= icon, initial_sidebar_state="auto")

# Set background color to black and adjust text color
st.markdown(
    """
    <style>
    .reportview-container {
        background: black;
        color: white;
    }
    .sidebar .sidebar-content {
        background: black;
        color: white;
    }
    .stPlotlyChart {
        color: white;
    }
    .stButton>button {
        color: white;
        background-color: #3a3b3c;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# cfd paramiters
nmax = 10000
tol = 0.000001

def parabolic_nozzle_ex(x, x_throat=5, inlet_radius=2, throat_radius=1, exit_radius=2):
    # Define the piecewise parabolic equation
    y = np.piecewise(
        x,
        [x < x_throat, x >= x_throat],
        [lambda x: -((inlet_radius - throat_radius) / x_throat**2) * (x - x_throat)**2 + inlet_radius,  # Converging part
         lambda x: ((exit_radius - throat_radius) / (x_throat - 10)**2) * (x - x_throat)**2 + throat_radius]  # Diverging part
    )
    return y

def finddt(n1, nmxm1, dx, u, t, c):
    # Find dt from interior gridpoints using CFL criteria
    # Apply Courant number after determination of dtmin
    dtmin = 1.0
    for i in range(n1, nmxm1):
        # Ensure u[i] and t[i] are not complex
        real_u = u[i].real if isinstance(u[i], complex) else u[i]
        real_t = t[i].real if isinstance(t[i], complex) else t[i]
        
        # Ensure we are not dealing with any negative or zero values that might arise from complex numbers
        if real_u + (real_t**0.5) != 0:
            dtmax = dx / (real_u + (real_t**0.5))
            if dtmax < dtmin:
                dtmin = dtmax
    dtmin = c * dtmin
    return dtmin

def FuncAoAStar(mach, g):
    # Function for Area Mach number relation
    # mach - Mach number, area - given area, g - ratio of specific heats
    
    y = (1/(mach**2)*(2/(g+1)*(1+(g-1)/2*mach**2))**((g+1)/(g-1)))**(1/2)
    return y

def MachAoAStar(subsup, area, g):
    # Find Mach number given A/A* by a combined
    # bisection and secant methods
    # if subsup = 0 subsonic solution
    # if subsup = 1 supersonic solution
    # area - given area ratio, g - ratio of specific heats
    # ** This function was defined in Project 1 and you
    #  should insert it here **
    
    #Here we set up the bounds
    if subsup == "sub":
        a = 0.00000001
        b = 1
    elif subsup == "sup":
        a = 1
        b = 10 

    #Here we set the tolerances and itterations for the methods
    maxn = 20
    tolb = 0.01

    #print("Bisection Replicating Table 7 Results:")
    #print(f'{"i":^1s} {"a":^9s} {"b":^9s} {"p":^10s} {"(b-a/2)":^10s} {"f(p)":^9s}')
    #print("------------------------------------------------------")
    #set itteration number
    i=0
    while i <= maxn:
        #bisect equation
        p = a + (b-a)/2
        if (FuncAoAStar(p,g)-area) == 0:
            print("Sol is ", p)
            break
        #does it hit 0?
        elif (b-a)/2 < tolb:
            break
        #print('{0:1d} {1:9.6f} {2:9.6f} {3:9.6f} {4:9.6f} {5:9.6f}'\
            #.format(i+1, a, b, p, (b-a)/2, FuncAoAStar(p,g)))
        # did it go past tolerance?
        if (FuncAoAStar(a,g)-area)*(FuncAoAStar(p,g)-area) > 0:
            a=p
        else:
            b=p
        i=i+1
    if i == maxn:
        print("method failed after ", i )

    maxn = 100
    i = 2
    p0 = a
    p1 = b
    q0 = (FuncAoAStar(p0,g)-area)
    q1 = (FuncAoAStar(p1,g)-area)

    while (i<=maxn):
        p = p1 - q1*(p1-p0)/(q1-q0)
        if abs(p-p1)<tol:
            return p
        i += 1
        p0 = p1
        q0 = q1
        p1 = p
        q1 = (FuncAoAStar(p,g)-area)
        if i == maxn-1:
            print('Method failed after ', i)

def findexact(x,area,g,r,v,t,p,m,md):
    # Define some constants and useful variables
    gogm1 = g/(g - 1)
    gm1o2 = (g - 1)/2
    npts = len(md)
    npts1 = npts-1
    areaend = area[npts1]
    minarea = min(area)
    iminarea = area.index(minarea)
    
    # Initialize exact variables
    xm,xt,xp,xr,xv,xmd=[[],[],[],[],[],[]]

    # Find exact solution throughout the nozzle for all points
    for i in range(npts):
        # Find Mach number based on area location
        # and whether flow is subsonic or supersonic
        if i <= iminarea:
            mache = MachAoAStar('sub', area[i],g)
        else: 
            mache = MachAoAStar('sup', area[i],g)
        # Determine the thermodynamic variables and mdot
        temp = 1/(1 + gm1o2*mache**2)
        pres = temp**(gogm1)
        rho = temp**(1/(g-1))
        vel = mache*(pres/rho)**(1/2)
        mdot = rho*area[i]*vel
        # Append values to the array
        xm.append(mache)
        xt.append(temp)
        xp.append(pres)
        xr.append(rho)
        xv.append(vel)
        xmd.append(mdot)
    return xm,xt,xp,xr,xv,xmd
# End Helper Functions #


def CFD_Solver(gam, npts, cour, xmax, a, b, c):

    # Nozzle Profile information
    xmin = 0
    dx = (xmax-xmin)/(npts-1)
    n1, nn = [0, npts-1]



    # Convergence tolerance, residual, and initial step
    residual = 1
    resid = []
    nstp = 0
    # Establish arrays of size npts:
    # xval, area, rho, temp, pres, vel, mach, mdot, prho, pvel, ptemp, drdt, dudt, dtdt
    xval  = [0]*npts 
    area  = [0]*npts
    rho = [0]*npts
    temp = [0]*npts
    pres = [0]*npts
    vel = [0]*npts
    mach = [0]*npts
    mdot = [0]*npts
    prho = [0]*npts
    pvel = [0]*npts
    ptemp = [0]*npts
    drdt = [0]*npts
    dudt = [0]*npts
    dtdt = [0]*npts

    # Initialize variables for plotting convergence at throat
    # Use append to add to array
    prest = []
    macht = []

    #Streamlit iteration data
    sxval  = []
    sarea  = []
    srho = []
    stemp = []
    spres = []
    svel = []
    smach = []
    smdot = []
    sprho = []
    spvel = []
    sptemp = []
    sdrdt = []
    sdudt = []
    sdtdt = []


    

    # Define initial conditions for the following variables: 
    # xval, area, rho, temp, pres, vel, mach, mdot

    # linear variation based on our knowlege of a convergant divergat nozzle
    # rho and temp decrease as vel increases
    x = xmin
    for i in range(npts):
        xval[i] = round(x,4)
        area[i] = a+b*(x-c)**2
        rho[i] = 1 - 0.3146*xval[i]
        temp[i] = 1 - 0.2314*xval[i]
        pres[i] = rho[i]*temp[i]*287
        vel[i] = (0.1+1.09*xval[i])*(temp[i]**(1/2)) #
        mach[i] = vel[i]/(temp[i]**(1/2))
        mdot[i] = rho[i]*vel[i]*area[i]
        x += dx



    # Calculate timestep based on internal points
    dt = finddt(1,npts-1,dx,vel,temp,cour)


    # Find the index of the minimum area for convergence
    throat =  area.index(min(area))


    # Set the initial predicted inflow boundary conditions (prho, ptemp, pres)
    # and define mach and mdot at inflow
    prho[n1]  = rho[n1]
    ptemp[n1] = temp[n1]
    pres[n1] = 1


    ############### Integrate in time using MacCormack's technique  ########################
    while nstp < nmax and residual > tol:
    ### Predictor step ###
        for i in range(n1+1,nn):
            # Compute all the x derivatives with forward differences
            dadx = (m.log(area[i+1]) - m.log(area[i]))/dx
            dudx = (vel[i+1] - vel[i])/dx
            drdx = (rho[i+1] - rho[i])/dx
            dtdx = (temp[i+1] - temp[i])/dx

            # Compute the t derivatives
            dudt[i] = -vel[i]*dudx - 1/gam * (dtdx + temp[i]/rho[i] * drdx)
            dtdt[i] = -vel[i]*dtdx - (gam - 1) * (temp[i]*dudx + temp[i]*vel[i]*dadx)
            drdt[i] = -rho[i]*dudx - rho[i]*vel[i]*dadx - vel[i]*drdx

            # Compute predicted variables
            prho[i]  = rho[i]  + drdt[i]*dt
            pvel[i] = vel[i]  + dudt[i]*dt
            ptemp[i] = temp[i]  + dtdt[i]*dt

        # Linearly extrapolate velocity to inflow
            pvel[n1] = 2*pvel[n1+1]-pvel[n1+2]

    ### Correcting step ###
        for i in range(n1+1,nn):
            # Compute x derivatives with backward differences (use the predicted values)
            dadx = (m.log(area[i]) - m.log(area[i-1]))/dx
            dudx = (pvel[i]-pvel[i-1])/dx
            drdx = (prho[i]-prho[i-1])/dx
            dtdx = (ptemp[i] - ptemp[i-1])/dx

            # Compute the t derivatives
            cdrdt = -prho[i]*dudx - prho[i]*pvel[i]*dadx - pvel[i]*drdx
            cdudt = -pvel[i]*dudx - 1/gam * (dtdx + ptemp[i]/prho[i] * drdx)
            cdtdt = (-pvel[i])*(dtdx)+(-1*(gam-1))*ptemp[i]*(dudx+pvel[i]*dadx)

            # Find the average time derivative
            adrdt = 0.5*(drdt[i]+cdrdt)
            adudt = 0.5*(dudt[i]+cdudt)
            adtdt = 0.5*(dtdt[i]+cdtdt)

            # Update rho, vel, temp variables
            # and compute pressure and Mach number
            rho[i]  = rho[i]  + adrdt*dt
            vel[i]  = vel[i]  + adudt*dt
            temp[i] = temp[i]  + adtdt*dt
            pres[i] = rho[i]*temp[i]
            mach[i] = vel[i]/(((temp[i])**(1/2)))
            mdot[i] = rho[i]*area[i]*vel[i]
            

    # Define residual at the throat to check for convergence
            if i == throat:
                residual = abs(adrdt)
                resid.append(residual)
                ## Add the convergence data to array for plotting
                prest.append(pres[i])
                macht.append(mach[i])

    # Apply vel and Mach boundary conditions at grid point 1
        vel[n1]  = 2*vel[n1+1]  - vel[n1+2]
        mach[n1] = 2*mach[n1+1] - mach[n1+2]

    # Apply rho, vel, temp boundary conditions at
    # grid point N (float all variables)
        rho[nn]  = 2*rho[nn-1]  - rho[nn-2]
        vel[nn]  = 2*vel[nn-1]  - vel[nn-2]
        temp[nn]  = 2*temp[nn-1]  - temp[nn-2]
        mdot[nn] = rho[nn]*area[nn]*vel[nn]

    # Evaluate pressure and Mach number at exit
        mach[nn] = vel[nn]/temp[nn]**(1/2)
        pres[nn] = rho[nn]*temp[nn]

        

    # Save iteration data
        sxval.append(list(xval))
        sarea.append(list(area))
        srho.append(list(rho))
        stemp.append(list(temp))
        spres.append(list(pres))
        svel.append(list(vel))
        smach.append(list(mach))
        smdot.append(list(mdot))




    # update the timestep
        dt = finddt(1,npts-1,dx,vel,temp,cour)

    # Update Itteration Value
        nstp += 1

    ######################################################################################

    # Print final information
    converged = True 
    print ("\n  Converged after {:4d} iterations".format(nstp))
    print ("  Residual (drdt @ throat): {:1.4E})".format(residual))


    # Determine the exact solution for given nozzle
    xm,xt,xp,xr,xv,xmd = findexact(xval,area,gam,rho,vel,temp,pres,mach,mdot)
    return macht, prest, sxval, sarea, srho, stemp, spres, svel, smach, smdot, sprho, spvel, sptemp, sdrdt, sdudt, sdtdt, xm, xt, xp, xr, xv, xmd



### Streamlit PAGE ###
# Adding a simple plot to the sidebar
st.sidebar.title("Project 5")
st.sidebar.subheader("Quasi 1D CFD Simulation")
st.sidebar.caption("Ethan Poynter, Dr. Lind")

# Multi-page option in Streamlit
page = st.sidebar.selectbox("Choose a page", ["MacCormack Technique", "Simulation and Data"])   

st.sidebar.markdown("""
## Navigation
Use the dropdown menu above to navigate through different sections of the app.

- **MacCormack:** Dive deep into the MacCormack method, a powerful numerical technique used in Computational Fluid Dynamics (CFD). Understand its principles, implementation steps, and its significance in solving complex fluid flow problems.
- **Simulation and Data:** Customize your simulation parameters such as inlet conditions, nozzle geometry, and grid resolution. Visualize how these adjustments influence the flow characteristics and observe the results in real-time through various plots and data tables.

""")
st.sidebar.image("https://lrhgit.github.io/tkt4140/allfiles/digital_compendium/chapter6/fig01.png", use_column_width=True)





## Initialize session states if not already done
if "new_sim" not in st.session_state:
    st.session_state.new_sim = False
if "xval" not in st.session_state:
    st.session_state.xval = [0]*100
if "area" not in st.session_state:
    st.session_state.area = [0]*100
if "fig2" not in st.session_state:
    # Initial plot setup
    st.session_state.fig2 = go.Figure()
    st.session_state.fig2.add_trace(go.Scatter(x=st.session_state.xval, y=st.session_state.area, mode='lines', fill='tozeroy', name='parea', line=dict(color='#00FF00')))
    st.session_state.fig2.update_layout(
        title="Nozzle Default Profile",
        xaxis_title="x'/L",
        yaxis_title="A/A*",
        template="plotly_dark",
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        legend_title="Legend",
        xaxis=dict(showgrid=True, gridcolor='gray'),
        yaxis=dict(showgrid=True, gridcolor='gray')
    )



if page == "MacCormack Technique":
    #page 1 talks about what is the macormach technique and how it is used.
    st.title("MacCormach Predictor Corrector")
    st.divider()
    col1, col2 = st.columns([1, 3])
    col1.image("https://i0.wp.com/blog.gridpro.com/wp-content/uploads/2020/08/Screenshot-2020-08-07-at-11.54.12-PM.png?resize=700%2C510&ssl=1")
    
    
    col2.subheader("Project Overview")
    col2.markdown("""
        Welcome to the MacCormack Predictor-Corrector CFD Calculator For a 1D Axisymetric Nozzle.
        
        ### Objectives
        - Provide an interactive platform for CFD simulations.
        - Illustrate the MacCormack Predictor-Corrector method.
        - Allow user inputs to customize the simulations.
        
        
                  """)

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        col1.caption("Predictor")
        st.latex(r'''
                (\frac{\partial \rho}{\partial x})_{i,j}= \frac{\rho_{i+1,j} - \rho_{i,j}}{\Delta x} + \sigma \Delta x
                ''')

    with col2:
        col2.caption("Central")
        st.latex(r'''
                (\frac{\partial \rho}{\partial x})_{i,j}= \frac{\rho_{i,j} - \rho_{i-1,j}}{\Delta x} + \sigma \Delta x
                ''')


    with col3:
        col3.caption("Corrector")
        st.latex(r'''
                (\frac{\partial \rho}{\partial x})_{i,j}= \frac{\rho_{i+1,j} - \rho_{i-1,j}}{\Delta x} + \sigma \Delta x
                ''')




    st.markdown("""

        ## What is the MacCormack Method?
        ---

        The MacCormack method is a two-step numerical approach for solving hyperbolic partial differential equations. It's built around a Taylor series expansion in time and involves two main steps:

        1. **Predictor Step**: This estimates the solution for the next time step using forward differencing.
        2. **Corrector Step**: This refines the estimated solution by applying backward differencing.

        Imagine a 2D horizontal plane where time progresses along the Z-axis. As numerical iterations proceed, time moves upward along this axis in increments of ('\u0394't). In this setup, our 2D nodes, 
        each with associated values, evolve over time. For hyperbolic equations applied to a mesh of nodes, we can find steady-state values by marching the data through time until convergence is achieved.
        The accuracy and stability of this method are influenced by the Courant-Friedrichs-Lewy (CFL) number, and will typically lie between 0 and 1.
        Convergence occurs when the change in properties between times (t) and (t-'\u0394't) falls within a small, predetermined tolerance.

        For our specific case, we are dealing with an axisymmetric isentropic subsonic-supersonic nozzle. This nozzle is created by revolving a parabolic plot around the x-axis over a length (L). 
        We distribute N nodes uniformly along the nozzle's length. We start by assigning properties to the node at (i = 1), and for the remaining nodes, we assign initial values intelligently to 
        ensure stability in our calculations. We then use the MacCormack equations to advance the flow properties at each grid point until we approach steady-state values.

        The MacCormack technique is crucial for applying to unsteady equations in quasi-1D flow scenarios, ensuring that we achieve a steady-state condition efficiently. 



                  """)
    
    # 3D Plot Representing 9 nodes across time
    col1, col2 = st.columns([0.75,1])
    with col1:
        # Define the grid points
        x = np.arange(-1, 2)  # Values from -1 to 1
        y = np.arange(-1, 2)  # Values from -1 to 1
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()

        # Define two sets of z values for demonstration purposes
        z1 = np.zeros_like(x)  # Set z1 to 0 for the first set of points
        z2 = np.ones_like(x)   # Set z2 to 1 for the second set of points
        z3 = (np.ones_like(x))+1   # Set z3 to 2 for the third set of points


        # Create the 3D scatter plot
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=x, 
            y=y, 
            z=z1, 
            mode='markers',
            marker=dict(size=6, color='blue'),
            name='Time Step 1'
        ))

        fig.add_trace(go.Scatter3d(
            x=x, 
            y=y, 
            z=z2, 
            mode='markers',
            marker=dict(size=6, color='red'),
            name='Time Step 2'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=x, 
            y=y, 
            z=z3, 
            mode='markers',
            marker=dict(size=6, color='green'),
            name='Time Step 3'
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title='i axis',
                yaxis_title='j axis',
                zaxis_title='t+ \u0394t axis',
                zaxis=dict(range=[0, 3.25]),
                xaxis=dict(range=[-1.5, 1.5]),
                yaxis=dict(range=[-1.5, 1.5])
            ),
            title='Time Stepped Mesh'
        )
        st.plotly_chart(fig)

    with col2:
        # Define the x range (e.g., from 0 to 3)
        x = np.linspace(0, 3, 500)

        # Calculate the nozzle radius for each x
        radii = np.sqrt(1 + 2.2 * (x - 1.5)**2)

        # Generate data for the 3D plot
        theta = np.linspace(0, 2 * np.pi, 100)  # Angle for the circular cross-section
        X, Theta = np.meshgrid(x, theta)
        R = np.sqrt(1 + 2.2 * (x - 1.5)**2)
        Z = R * np.cos(Theta)
        Y = R * np.sin(Theta)

        # Define 30 equally spaced points along the x-axis
        x_points = np.linspace(0, 3, 15)
        y_points = [0]*15  # y-coordinates (0 in this case for simplicity)
        z_points = [0]*15  # z-coordinates (0 in this case for simplicity)

        # Create the 3D surface plot using Plotly
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])

        # Add the 30 equally spaced points along the axis of revolution
        fig.add_trace(go.Scatter3d(
            x=x_points,
            y=y_points,
            z=z_points,
            mode='markers',
            marker=dict(size=4, color='blue'),
            name='Quasi 1D Nodes'
        ))

        # Update layout for better visualization
        fig.update_layout(
            title="3D Axisymmetric Nozzle Profile with Nodes",
            scene=dict(
                xaxis_title="Length (x)",
                yaxis_title="Width",
                zaxis_title="Height"
            )
        )

        # Display the plot inline on the same page
        st.plotly_chart(fig)




    st.markdown("""

        ## Unsteady Equations. What Can We Model?
        ---

        To analyze our subsonic-supersonic isentropic quasi-1D nozzle, we derive the necessary equations from the principles of conservation: continuity, momentum, and energy. These equations are 
        typically presented in algebraic form, but we need to work with their unsteady counterparts in integral form for continuity, momentum, and energy, assuming no body forces or heat addition. 
        We apply these equations to an infinitesimal control volume, where (dV = A dx), and rewrite the energy equations for unsteady flow in terms of specific heat at constant volume (e = c_v * t).
        This process yields three equations with four unknowns. To resolve this, we simplify by eliminating pressure terms in the momentum and energy equations, expressing them 
        in terms of (R/c_v = gamma - 1), which reduces our system to three equations with three unknowns.

        For nozzle flow analysis, it's advantageous to express these equations in non-dimensional variables. This approach reduces the number of free parameters, provides insight into their 
        relative magnitudes, and confines the range of variables to between 0 and 1. For instance, non-dimensional variables such as (p/p_0), \(T/T_0), and (\rho/\rho_0) can be used,
        where (p_0), (T_0), and (\rho_0) represent stagnation conditions.

        By applying these non-dimensionalized conditions, we can determine the values of our output parameters and gain a clearer understanding of the nozzle flow dynamics.
                
        At the end we can apply our total conditons to find values for our outputs. 

                  """)
    st.write("")
    st.write("")
    st.caption("Finding our Equations:")
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.subheader("Conservation Equations:")
        st.caption("Continuity Equation")
        st.latex(r'''
                { \rho_1} {u_1} {A_1}=  { \rho_2} {u_2} {A_2}
                ''')
        st.caption("Momentum Equation")
        st.latex(r'''
                p_1 A_1 + \rho_1 u_1^2 A_1 + \int_{A_1}^{A_2}  \, pdA = p_2 A_2 + \rho_2 u_2^2 A_2
                ''')
        st.caption("Energy Equation")
        st.latex(r'''
                h_1 + u_1^2/2 = h_2 + u_2^2/2
                ''')
        
        with col2:
            st.subheader("Unsteady Conservation Equations:")
            st.caption("Unsteady Continuity Equation")
            st.latex(r'''
                    \frac{\partial}{\partial t} \int\int\int\, \rho dV + \int\int\, \rho \mathbf{V} \cdot d \mathbf{S} = 0
                    ''')
            st.caption("Unsteady Momentum Equation")
            st.latex(r'''
                    \frac{\partial}{\partial t} \int\int\int\, d\rho \mathbf{V} dV + \int\int\,(\rho \mathbf{V} \cdot d\mathbf{S})\mathbf{V}= -\int\int\, pd\mathbf{S} = \mathbf{0}
                    ''')
            st.caption("Unsteady Energy Equation")
            st.latex(r'''
                    \frac{\partial}{\partial t} \int\int\int\, \rho(e+\frac{V^2}{2})dV + \int\int\,\rho(e+\frac{V^2}{2})\mathbf{V} \cdot d\mathbf{S} = \int\int\int\, p\mathbf{V} \cdot d\mathbf{S}
                    ''')
        with col3:
            st.subheader("Derrived Conservation Equations:")
            st.caption("Derrived Continuity Equation")
            st.latex(r'''
                    \frac{\partial \rho}{\partial t} = -\rho \frac{\partial u}{\partial x} - pu \frac{\partial(ln A)}{\partial x}-u\frac{\partial \rho}{\partial x}
                    ''')
            st.caption("Derrived Momentum Equation")
            st.latex(r'''
                    \frac{\partial u}{\partial t} = -u \frac{\partial u}{\partial x} - R(\frac{\partial T}{\partial x} + \frac{T}{\rho}\frac{\partial \rho}{\partial x} )
                    ''')
            st.caption("Derrived Energy Equation")
            st.latex(r'''
                    \frac{\partial T}{\partial t} = -u \frac{\partial T}{\partial x}-(\gamma - 1)T (\frac{\partial u}{\partial x} +u \frac{\partial(ln A)}{\partial x} )
                    ''')

    st.divider()
    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("Nondimentionalizing Equations:")
    with col2:
        st.subheader("Nondimentionalized Conservation Equations:")

    col1, col2, col3, col4 = st.columns([0.33, 0.33, 0.33, 1])
    with col1: 
        st.subheader("")
        st.caption("Length")
        st.latex(r'''
                x' = \frac{x}{L} \to x=x'L
                ''')
        st.caption("Velocity")
        st.latex(r'''
                u' = \frac{u}{a_o} \to u=u'a_o
                ''')
    with col2: 
        st.subheader("")
        st.caption("Temprtaure")
        st.latex(r'''
                T' = \frac{T}{T_o} \to T=T'T_o
                ''')
        st.caption("time")
        st.latex(r'''
                u' = \frac{t}{L/a_o} \to t=Lt'/a_o
                ''')
    with col3: 
        st.subheader("")
        st.caption("Density")
        st.latex(r'''
                \rho' = \frac{\rho}{\rho_o} \to \rho=\rho'\rho_o
                ''')
        st.caption("Area")
        st.latex(r'''
                A' = \frac{A}{A^*} \to A=A'A^*
                ''')
    with col4:
        st.caption("Nondimentionalized Continuity Equation")
        st.latex(r'''
                \frac{\partial \rho'}{\partial t'} = -\rho' \frac{\partial u'}{\partial x'} - p'u' \frac{\partial(ln A')}{\partial x'}-u'\frac{\partial \rho'}{\partial x'}
                ''')
        st.caption(" Nondimentionalized Equation")
        st.latex(r'''
                \frac{\partial u'}{\partial t'} = -u' \frac{\partial u'}{\partial x'} - \frac{1}{\gamma}(\frac{\partial T'}{\partial x'} + \frac{T'}{\rho'}\frac{\partial \rho'}{\partial x'} )
                ''')
        st.caption("Nondimentionalized Energy Equation")
        st.latex(r'''
                \frac{\partial T'}{\partial t'} = -u' \frac{\partial T'}{\partial x'}-(\gamma - 1)T' (\frac{\partial u'}{\partial x'} +u' \frac{\partial(ln A')}{\partial x'} )
                ''')








# Parameters and simulation page
if page == "Simulation and Data":
    col1, col2, col3 = st.columns([2.9, 1.5, 0.75])

    with col1:
        st.title("Input Parameters")
        st.markdown("### Enter the parameters for your simulation:")
        st.markdown("""
        ## Input Parameters
        Fill in the fields below to customize your CFD simulation.

        - **Nozzle Geometry:** Define the shape and dimensions of the nozzle.
        - **Flow Conditions:** Specify initial conditions like pressure and temperature ratios.
        - **Simulation Parameters:** Set the tolerance and Courant number for the simulation.
        """)
    with col2:

        # Create a placeholder for the plot
        plot_placeholder = st.empty()

        # Display the initial plot using the placeholder
        plot_placeholder.plotly_chart(st.session_state.fig2, use_container_width=True)

    st.header("Nozzle Geometry")
    st.markdown("Nozzle General Form: A(x) = a+b*(x-c)**2")
    col3, col4, col5 = st.columns(3)
    with col3:
        A = st.number_input("A value", value=1.0, step=0.1)
    with col4:
        B = st.number_input("B value", value=2.2, step=0.1)
    with col5:
        C = st.number_input("C value", value=1.5, step=0.1)
    Length = st.number_input("Length Of Nozzle", min_value=0.1, value=3.0, step=0.1)

    st.markdown("### Evaluate Nozzle")
    if st.button("Plot"):
        st.write("Running the simulation with the provided parameters...")
        step = 0
        xval = [0] * 100
        area = [0] * 100
        for i in range(100):
            xval[i] = step
            area[i] = round(A + B * (xval[i] - C) ** 2, 4)
            step += Length / 100

        # Update session state with new values
        st.session_state.xval = xval
        st.session_state.area = area

        # Update the plot data
        st.session_state.fig2.data = []  # Clear existing data
        st.session_state.fig2.add_trace(go.Scatter(x=st.session_state.xval, y=st.session_state.area, mode='lines', fill='tozeroy', name='parea', line=dict(color='#00FF00')))
        st.session_state.fig2.update_layout(
            title="Updated Nozzle Profile",
            xaxis_title="x'/L",
            yaxis_title="A/A*",
            template="plotly_dark",
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            legend_title="Legend",
            xaxis=dict(showgrid=True, gridcolor='gray'),
            yaxis=dict(showgrid=True, gridcolor='gray')
        )

        # Update the plot using the placeholder
        plot_placeholder.plotly_chart(st.session_state.fig2, use_container_width=True)



    # Simulation Conditions 
    st.header("Flow Conditions")
    col1, col2, col3 = st.columns(3)
    with col1:
        initial_Pressure = st.number_input("Initial Pressure (kpa)", min_value= 100, value = 101, step = 50)
    with col2:
        initial_temperature = st.number_input("Initial Temperature (K)", min_value= 0, value = 275, step=10)
    with col3:
        gamma = st.number_input("Gamma", value=1.4, step=0.01)
    
    st.header("Simulation Parameters")
    col1, col2 = st.columns(2)
    with col1:
        nstp = st.number_input("Number of Points", min_value=1, value=31)
    with col2:
        courant_number = st.number_input("Courant Number", value=0.5, step=0.1)


    # Button to trigger the simulation
    st.markdown("### Run Simulation")
    if st.button("Run Simulation"):
        st.write("Running the simulation with the provided parameters...")
        macht, prest, sxval, sarea, srho, stemp, spres, svel, smach, smdot, sprho, spvel, sptemp, sdrdt, sdudt, sdtdt, xm, xt, xp, xr, xv, xmd = CFD_Solver(gamma, nstp, courant_number, Length, A, B, C)
        st.session_state.new_sim = True

    st.divider()



    if st.session_state.new_sim == True:

        initial_density = initial_Pressure / (287*initial_temperature)
        # multply array by values for dementioned data
        dtemp = [x * (initial_temperature) for x in stemp[-1]]
        dpres = [x * (initial_Pressure) for x in spres[-1]]
        drho = [x * (initial_density) for x in srho[-1]]


        # Set Plotting Array
        a = 0
        b = int(len(sxval)/10)
        c = int(len(sxval)/6)
        leng = []
        for i in range(nstp):
            leng.append(round(Length/nstp, 4)*i)



        st.title("Steady State Solution Plots")
        col1, col2 = st.columns([1, 1])
        with col1:
            # Mach Number Plot
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=list(leng), y=smach[-1], mode='lines', name='Mach (' + str(len(smach[-1])) + ')', line=dict(color='#00FF00')))
            fig1.update_layout(
                title="Mach",
                xaxis_title="L",
                yaxis_title="Mach Number",
                template="plotly_dark",
                plot_bgcolor='#303030',
                paper_bgcolor='#303030',
                legend_title="Legend",
                xaxis=dict(showgrid=True, gridcolor='gray'),
                yaxis=dict(showgrid=True, gridcolor='gray')
            )
            st.plotly_chart(fig1, use_container_width=True)

            # Temptature Plot
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=list(leng), y=dtemp, mode='lines', name='T/To*To', line=dict(color='#00FF00')))
            fig2.update_layout(
                title="Temprature (K)",
                xaxis_title="L",
                yaxis_title="T (K)",
                template="plotly_dark",
                plot_bgcolor='#303030',
                paper_bgcolor='#303030',
                legend_title="Legend",
                xaxis=dict(showgrid=True, gridcolor='gray'),
                yaxis=dict(showgrid=True, gridcolor='gray')
            )
            st.plotly_chart(fig2, use_container_width=True)


        # Pressure Plot
        with col2:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=list(leng), y=dpres, mode='lines', name='P/Po*Po', line=dict(color='#00FF00')))
            fig3.update_layout(
                title="Pressure (kpa)",
                xaxis_title="x",
                yaxis_title="P (Kpa)",
                template="plotly_dark",
                plot_bgcolor='#303030',
                paper_bgcolor='#303030',
                legend_title="Legend",
                xaxis=dict(showgrid=True, gridcolor='gray'),
                yaxis=dict(showgrid=True, gridcolor='gray')
            )
            st.plotly_chart(fig3, use_container_width=True)

            # Density Plot
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=list(leng), y=drho, mode='lines', name='ρ/ρo*ρo', line=dict(color='#00FF00')))
            fig4.update_layout(
                title="Density (kg/m-3)",
                xaxis_title="x",
                yaxis_title="ρ (kg/m-3)",
                template="plotly_dark",
                plot_bgcolor='#303030',
                paper_bgcolor='#303030',
                legend_title="Legend",
                xaxis=dict(showgrid=True, gridcolor='gray'),
                yaxis=dict(showgrid=True, gridcolor='gray')
            )
            st.plotly_chart(fig4, use_container_width=True)




        # Time Evolution 
        st.title("Time Evolution")
        col1, col2, col3 = st.columns([0.75, 0.75, 0.75])
        

        # Mach Number Convergence Plot
        with col1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=list(leng), y=smach[a], mode='lines', name='Mach (' + str(a) + ')', line=dict(color='#FF00FF')))
            fig1.add_trace(go.Scatter(x=list(leng), y=smach[b], mode='lines', name='Mach (' + str(b) + ')', line=dict(color='#FF0000')))
            fig1.add_trace(go.Scatter(x=list(leng), y=smach[c], mode='lines', name='Mach (' + str(c) + ')', line=dict(color='#00FFFF')))
            fig1.add_trace(go.Scatter(x=list(leng), y=smach[-1], mode='lines', name='Mach (' + str(len(smach)) + ')', line=dict(color='#00FF00')))
            fig1.add_trace(go.Scatter(x=list(leng), y=xm, mode='markers', name='Explicit Value', line=dict(color='#000000')))
            fig1.update_layout(
                title="Mach Time Convergance",
                xaxis_title="x/L",
                yaxis_title="Mach Number",
                template="plotly_dark",
                plot_bgcolor='#303030',
                paper_bgcolor='#303030',
                legend_title="Legend",
                xaxis=dict(showgrid=True, gridcolor='gray'),
                yaxis=dict(showgrid=True, gridcolor='gray')
            )
            st.plotly_chart(fig1, use_container_width=True)

            # Pressure Evolution
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=list(leng), y=stemp[a], mode='lines', name='T/To (' + str(a) + ')', line=dict(color='#FF00FF')))
            fig2.add_trace(go.Scatter(x=list(leng), y=stemp[b], mode='lines', name='T/To (' + str(b) + ')', line=dict(color='#FF0000')))
            fig2.add_trace(go.Scatter(x=list(leng), y=stemp[c], mode='lines', name='T/To (' + str(c) + ')', line=dict(color='#00FFFF')))
            fig2.add_trace(go.Scatter(x=list(leng), y=stemp[-1], mode='lines', name='T/To (' + str(len(stemp)) + ')', line=dict(color='#00FF00')))
            fig2.add_trace(go.Scatter(x=list(leng), y=xt, mode='markers', name='Explicit Value', line=dict(color='#000000')))
            fig2.update_layout(
                title="Temprature Time Convergance",
                xaxis_title="x/L",
                yaxis_title="T/To",
                template="plotly_dark",
                plot_bgcolor='#303030',
                paper_bgcolor='#303030',
                legend_title="Legend",
                xaxis=dict(showgrid=True, gridcolor='gray'),
                yaxis=dict(showgrid=True, gridcolor='gray')
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            # Velocity Evolution
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=list(leng), y=svel[a], mode='lines', name='Velocity (' + str(a) + ')', line=dict(color='#FF00FF')))
            fig3.add_trace(go.Scatter(x=list(leng), y=svel[b], mode='lines', name='Velocity (' + str(b) + ')', line=dict(color='#FF0000')))
            fig3.add_trace(go.Scatter(x=list(leng), y=svel[c], mode='lines', name='Velocity (' + str(c) + ')', line=dict(color='#00FFFF')))
            fig3.add_trace(go.Scatter(x=list(leng), y=svel[-1], mode='lines', name='Velocity (' + str(len(svel)) + ')', line=dict(color='#00FF00')))
            fig3.add_trace(go.Scatter(x=list(leng), y=xv, mode='markers', name='Explicit Value', line=dict(color='#000000')))
            fig3.update_layout(
                title="Velocity Time Convergance",
                xaxis_title="x/L",
                yaxis_title="Vel m/s",
                template="plotly_dark",
                plot_bgcolor='#303030',
                paper_bgcolor='#303030',
                legend_title="Legend",
                xaxis=dict(showgrid=True, gridcolor='gray'),
                yaxis=dict(showgrid=True, gridcolor='gray')
            )
            st.plotly_chart(fig3, use_container_width=True)

            # Pressure Evolution
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=list(leng), y=spres[a], mode='lines', name='P/Po (' + str(a) + ')', line=dict(color='#FF00FF')))
            fig4.add_trace(go.Scatter(x=list(leng), y=spres[b], mode='lines', name='P/Po (' + str(b) + ')', line=dict(color='#FF0000')))
            fig4.add_trace(go.Scatter(x=list(leng), y=spres[c], mode='lines', name='P/Po (' + str(c) + ')', line=dict(color='#00FFFF')))
            fig4.add_trace(go.Scatter(x=list(leng), y=spres[-1], mode='lines', name='P/Po (' + str(len(spres)) + ')', line=dict(color='#00FF00')))
            fig4.add_trace(go.Scatter(x=list(leng), y=xp, mode='markers', name='Explicit Value', line=dict(color='#000000')))
            fig4.update_layout(
                title="Pressure Time Convergance",
                xaxis_title="x/L",
                yaxis_title="P/Po",
                template="plotly_dark",
                plot_bgcolor='#303030',
                paper_bgcolor='#303030',
                legend_title="Legend",
                xaxis=dict(showgrid=True, gridcolor='gray'),
                yaxis=dict(showgrid=True, gridcolor='gray')
            )
            st.plotly_chart(fig4, use_container_width=True)


        with col3:
            # M_dot Evolution
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=list(leng), y=smdot[a], mode='lines', name='Mdot (' + str(a) + ')', line=dict(color='#FF00FF')))
            fig5.add_trace(go.Scatter(x=list(leng), y=smdot[b], mode='lines', name='Mdot (' + str(b) + ')', line=dict(color='#FF0000')))
            fig5.add_trace(go.Scatter(x=list(leng), y=smdot[c], mode='lines', name='Mdot (' + str(c) + ')', line=dict(color='#00FFFF')))
            fig5.add_trace(go.Scatter(x=list(leng), y=smdot[-1], mode='lines', name='Mdot (' + str(len(spres)) + ')', line=dict(color='#00FF00')))
            fig5.add_trace(go.Scatter(x=list(leng), y=xmd, mode='markers', name='Explicit Value', line=dict(color='#000000')))
            fig5.update_layout(
                title="Mass Flow Rate Time Convergance",
                xaxis_title="x/L",
                yaxis_title="M_dot",
                template="plotly_dark",
                plot_bgcolor='#303030',
                paper_bgcolor='#303030',
                legend_title="Legend",
                xaxis=dict(showgrid=True, gridcolor='gray'),
                yaxis=dict(showgrid=True, gridcolor='gray')
            )
            st.plotly_chart(fig5, use_container_width=True)

            # Density Evolution
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(x=list(leng), y=srho[a], mode='lines', name='ρ/ρo (' + str(a) + ')', line=dict(color='#FF00FF')))
            fig6.add_trace(go.Scatter(x=list(leng), y=srho[b], mode='lines', name='ρ/ρo (' + str(b) + ')', line=dict(color='#FF0000')))
            fig6.add_trace(go.Scatter(x=list(leng), y=srho[c], mode='lines', name='ρ/ρo (' + str(c) + ')', line=dict(color='#00FFFF')))
            fig6.add_trace(go.Scatter(x=list(leng), y=srho[-1], mode='lines', name='ρ/ρo (' + str(len(spres)) + ')', line=dict(color='#00FF00')))
            fig6.add_trace(go.Scatter(x=list(leng), y=xr, mode='markers', name='Explicit Value', line=dict(color='#000000')))
            fig6.update_layout(
                title="Density Time Convergance",
                xaxis_title="x/L",
                yaxis_title="ρ/ρo",
                template="plotly_dark",
                plot_bgcolor='#303030',
                paper_bgcolor='#303030',
                legend_title="Legend",
                xaxis=dict(showgrid=True, gridcolor='gray'),
                yaxis=dict(showgrid=True, gridcolor='gray')
            )
            st.plotly_chart(fig6, use_container_width=True)



        # Validation Information
        st.title("Convergance")
        col1, col2, col3 = st.columns([0.75, 0.75, 0.75])

        # Pressure at throat convergance plot
        with col1:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=list(range(len(prest))), y=prest, mode='lines', name='P/Po (' + str(len(prest)) + ')', line=dict(color='#0000FF')))
            fig3.update_layout(
                title="Pressure at Throat Convergance",
                xaxis_title="Iteration",
                yaxis_title="P/Po",
                template="plotly_dark",
                plot_bgcolor='#303030',
                paper_bgcolor='#303030',
                legend_title="Legend",
                xaxis=dict(showgrid=True, gridcolor='gray'),
                yaxis=dict(showgrid=True, gridcolor='gray')
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Mach at Throat Plot
        with col2:
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=list(range(len(macht))), y=macht, mode='lines', name='macht (' + str(len(macht)) + ')', line=dict(color='#0000FF')))
            fig4.update_layout(
                title="Mach Number at Throat Convergance",
                xaxis_title="Iteration",
                yaxis_title="Thoat Mach",
                template="plotly_dark",
                plot_bgcolor='#303030',
                paper_bgcolor='#303030',
                legend_title="Legend",
                xaxis=dict(showgrid=True, gridcolor='gray'),
                yaxis=dict(showgrid=True, gridcolor='gray')
            )
            st.plotly_chart(fig4, use_container_width=True)

            # Mass Flow Rate Plot
            with col3:
                fig5 = go.Figure()
                fig5.add_trace(go.Scatter(x=list(range(len(smdot[-1]))), y=smdot[-1], mode='lines', name='M_dot (' + str(len(macht)) + ')', line=dict(color='#0000FF')))
                fig5.update_layout(
                    title="Mass Flow Rate",
                    xaxis_title="Iteration",
                    yaxis_title="M_dot",
                    template="plotly_dark",
                    plot_bgcolor='#303030',
                    paper_bgcolor='#303030',
                    legend_title="Legend",
                    xaxis=dict(showgrid=True, gridcolor='gray'),
                    yaxis=dict(showgrid=True, gridcolor='gray')
                )
                st.plotly_chart(fig5, use_container_width=True)


        st.divider()
        BulkRatios = {
            "x": sxval,
            "Area": sarea,
            "R/Ro": srho,
            "Velocity": svel,
            "T/To": stemp,
            "P/Po": spres,
            "Mach": smach,
            "Mdot": smdot
        }

        Ratio_Results = {
            "x": sxval[-1],
            "Area": sarea[-1],
            "R/Ro": srho[-1],
            "Velocity": svel[-1],
            "T/To": stemp[-1],
            "P/Po": spres[-1],
            "Mach": smach[-1],
            "Mdot": smdot[-1]
        }
       
        Results = {
            "x": sxval[-1],
            "Area": sarea[-1],
            "R": drho,
            "Velocity": svel[-1],
            "T": dtemp,
            "P": dpres,
            "Mach": smach[-1],
            "Mdot": smdot[-1]
        }

        BulkRatios_ex = pd.DataFrame(BulkRatios)
        Ratio_Results_ex = pd.DataFrame(Ratio_Results)
        Results_ex = pd.DataFrame(Results)

        st.title("Simulation Ratio Data")
        st.dataframe(BulkRatios_ex)

        col1, col2 = st.columns([1,1])
        with col1:
            st.title("Flow Ratio Results")
            st.dataframe(Ratio_Results_ex)
        with col2:
            st.title("Flow Property Results")
            st.dataframe(Results_ex)



























































