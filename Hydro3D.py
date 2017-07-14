import numpy as np
import matplotlib.pyplot as plt
import pylab
import math
import copy
import os

# ===================================================================
#                         FARGO3D to RADMC3D
#                           by Seba Perez
# ===================================================================
#
# Setup FARGO3D outputs for input into RADMC3D (v0.39, Dullemond et
# al) using python.
#
# Considerations:
#    - Gas density and velocities come from simulation. Dust is
#      assumed standard dust size distribution.
#
#    - Spherical coordinate system based on input simulation grid.
#      Fargo convention:
#        X = theta == azimuthal angle
#        Y = r     == radius
#        Z = phi   == colatitude
#
#      RADMC convention:
#        x, y, z <---> rad (r), col (theta), azi (phi)
#
# ===================================================================

# -------------------------------------------------------------------
# global constants
# -------------------------------------------------------------------

M_Sun = 1.9891e33               # [M_sun] = g
G = 6.67259e-8                  # [G] = dyn cm2 g-2
au = 14959787066000             # [Astronomical unit] = cm
m_H = 1.673534e-24              # [Hydrogen mass] = g
pi = 3.141592653589793116       # [pi]
R_Sun = 6.961e10                # [R_sun] = cm
kB = 1.380658E-16
pc = 3.085678E18                # [pc] = cm
c  = 2.99792458E10              # [c] = cm s-1

# -------------------------------------------------------------------
# building mesh arrays for theta, r, phi (x, y, z)
# -------------------------------------------------------------------

class Mesh():
    # based on P. Benitex routine
    """
    Mesh class, keeps all mesh data.
    Input: directory [string] -> place where domain files are
    """
    def __init__(self, directory=""):
        if len(directory) > 1:
            if directory[-1] != '/':
                directory += '/'
        try:
            domain_x = np.loadtxt(directory+"domain_x.dat") # AZIMUTH
        except IOError:
            print "IOError with domain_x.dat"
        try:
            # avoid ghost cells!
            domain_y = np.loadtxt(directory+"domain_y.dat")[3:-3] # RADIUS
        except IOError:
            print "IOError with domain_y.dat"
        try:
            # avoid ghost cells!
            domain_z = np.loadtxt(directory+"domain_z.dat")[3:-3] # COLATITUDE
        except IOError:
            print "IOError with domain_z.dat"

        # --------------------------------------------------------------
        # SWAPPING CONVENTION
        # --------------------------------------------------------------
        # rad will be stored in X
        # col will be stored in Y
        # azi will be stored in Z
        # X, Y, Z --> Y, Z, X
        # --------------------------------------------------------------

        self.xm = domain_y # rad-edge
        self.ym = domain_z # col-Edge
        self.zm = domain_x # azi-Edge

        self.xmed = 0.5*(domain_y[:-1] + domain_y[1:]) # rad-Center
        self.ymed = 0.5*(domain_z[:-1] + domain_z[1:]) # col-Center
        self.zmed = 0.5*(domain_x[:-1] + domain_x[1:]) # azi-Center

        # surfaces taken from the edges
        # make 2D arrays for x, y, that are (theta, r)
        T,R = np.meshgrid(self.zm, self.xm)
        R2  = R*R

        # --------------------------------------------------------------
        # plotting the mesh
        # --------------------------------------------------------------

        # surfaces taken from the edges
        # make 2D arrays for rad, azi (r, phi)

        # radius vs azimuth (cartesian)
        Plot = False
        if Plot:
            import matplotlib.pyplot as plt
            ax = plt.gca()
            P,R = np.meshgrid(self.zm, self.xm)
            X = R*np.cos(P)
            Y = R*np.sin(P)
            plt.pcolor(X,Y,np.random.rand(len(self.xm),len(self.zm)),
                       cmap='plasma', edgecolors='black')
            plt.axis('equal')
            plt.show()

        self.surf = 0.5*(T[:-1,1:]-T[:-1,:-1])*(R2[1:,:-1]-R2[:-1,:-1])



# -------------------------------------------------------------------
# reading parameter file
# -------------------------------------------------------------------

class Parameters():
    # based on P. Benitex routine
    """
    Reading simulation parameters.
    input: string -> name of the parfile, normally variables.par
    """
    def __init__(self, directory=''):
        if len(directory) > 1:
            if directory[-1] != '/':
                directory += '/'
        try:
            params = open(directory+"variables.par",'r') # opening parfile
        except IOError:         # error checker.
            print  paramfile + " not found."
            return
        lines = params.readlines()     # reading parfile
        params.close()                 # closing parfile
        par = {}                       # allocating a dictionary
        for line in lines:             # iterating over parfile
            name, value = line.split() # spliting name and value (first blank)
            try:
                float(value)           # first trying with float
            except ValueError:         # if it is not float
                try:
                    int(value)         # we try with integer
                except ValueError:     # if it is not integer, we know it is string
                    value = '"' + value + '"'
            par[name] = value          # filling variable
        self._params = par             # control atribute, good for debbuging
        for name in par:               # iterating over the dictionary
            exec("self."+name.lower()+"="+par[name]) # making atributes at runtime


# -------------------------------------------------------------------
# reading fields
# can be density, energy, velocities, etc
# -------------------------------------------------------------------

class Field(Mesh, Parameters):
    # based on P. Benitex routine
    """
    Field class, it stores all the mesh, parameters and scalar data
    for a scalar field.
    Input: field [string] -> filename of the field
           staggered='c' [string] -> staggered direction of the field.
                                      Possible values: 'x', 'y', 'xy', 'yx'
           directory='' [string] -> where filename is
           dtype='float64' (numpy dtype) -> 'float64', 'float32',
           depends if FARGO_OPT+=-DFLOAT is activated
    """
    def __init__(self, field, staggered='c', directory='', dtype='float64'):
        if len(directory) > 1:
            if directory[-1] != '/':
                directory += '/'

        Mesh.__init__(self, directory) # all Mesh attributes inside Field
        Parameters.__init__(self, directory) # all Parameters inside Field

        # fixing grid sizes due to reordering
        self.nx = (self.xm.size-1)
        self.ny = (self.ym.size-1)
        self.nz = (self.zm.size-1)

        # --------------------------------------------------------------
        # Unit conversion for radii
        # assumes R0 = 1 au
        # --------------------------------------------------------------
        self.xm *= au
        self.xmed *= au

        # now, staggering:
        if staggered.count('x')>0:
            self.x = self.xm[:-1] # do not dump last element
        else:
            self.x = self.xmed
        if staggered.count('y')>0:
            self.y = self.ym[:-1]
        else:
            self.y = self.ymed

        # scalar data is here:
        self.data_total = self.__open_field(directory+field,dtype)
        self.data=self.data_total[-1,:,:] # -1 = midplane in HALFDISK mode
        # self.data=np.sum(self.data_total, axis = 0) # collapse over colatitude

    def __open_field(self, f, dtype):
        """
        Reading the data
        """
        field = np.fromfile(f, dtype=dtype)
        return field.reshape(self.ny,self.nx,self.nz)


# -------------------------------------------------------------------
# write AMR Grid
# -------------------------------------------------------------------


def write_AMRgrid(F, R_Scaling, Plot=False):

    print "writing AMR GRID"
    path_grid='amr_grid.inp'
    grid=open(path_grid,'w')

    grid.write('1 \n')              # iformat/ format number = 1
    grid.write('0 \n')              # Grid style (regular = 0)
    grid.write('101 \n')            # coordsystem: 100 < spherical < 200
    grid.write('0 \n')              # gridinfo
    grid.write('1 \t 1 \t 1 \n')    # incl x, incl y, incl z

    # radius colatitude and azimuth
    grid.write(str(F.nx)+ '\t'+ str(F.ny)+'\t'+ str(F.nz)+'\n')

    # ghost cells have been avoided in the Mesh class
    # radius
    # Scaling
    F.xm *= R_Scaling

    for i in range(F.nx+1):
        grid.write(str(F.xm[i])+'\t')
    grid.write('\n')

    # colatitude
    for i in range(F.ny+1):
        grid.write(str(F.ym[i])+'\t')
    grid.write('\n')

    # azimuth
    # if azimuth grid goes between -pi and pi, add pi
    if ( np.abs(F.xmax-pi)/pi < 1e-3 ):
        print '\tforcing azimuth between 0 and 2pi'
        F.zm += pi
    for i in range(F.nz+1):
        grid.write(str(F.zm[i])+'\t') # added pi
    grid.write('\n')

    grid.close()

    Plot = False
    if Plot:
        import matplotlib.pyplot as plt
        ax = plt.gca()
        P,R = np.meshgrid(F.zm, F.xm)
        X = R*np.cos(P)
        Y = R*np.sin(P)
        plt.pcolor(X,Y,np.random.rand(len(F.xm),len(F.zm)),
                   cmap='plasma', edgecolors='black')
        plt.axis('equal')
        plt.show()



# -------------------------------------------------------------------
# writing out wavelength
# -------------------------------------------------------------------

def write_wavelength():

    wmin = 0.1
    wmax = 3000.0
    Nw = 150
    Pw = (wmax/wmin)**(1.0/(Nw-1))

    waves = np.zeros(Nw)
    waves[0] = wmin
    for i in xrange(1, Nw):
        waves[i]=wmin*Pw**i

    print 'writing wavelength_micron.inp'

    path = 'wavelength_micron.inp'
    wave = open(path,'w')
    wave.write(str(Nw)+'\n')
    for i in xrange(Nw):
        wave.write(str(waves[i])+'\n')
    wave.close()


def write_stars(Rstar = 1, Tstar = 6000):

    wmin = 0.1
    wmax = 3000.0
    Nw = 150
    Pw = (wmax/wmin)**(1.0/(Nw-1))

    waves = np.zeros(Nw)
    waves[0] = wmin
    for i in xrange(1, Nw):
        waves[i]=wmin*Pw**i

    print 'writing stars.inp'

    path = 'stars.inp'
    wave = open(path,'w')

    wave.write('\t 2\n')
    wave.write('1 \t'+str(Nw)+'\n')
    wave.write(str(Rstar*R_Sun)+'\t'+str(M_Sun)+'\t 0 \t 0 \t 0 \n')
    for i in xrange(Nw):
        wave.write('\t'+str(waves[i])+'\n')
    wave.write('\t -'+str(Tstar)+'\n')

    wave.close()





# -------------------------------------------------------------------
# writing dustopac
# -------------------------------------------------------------------

def write_dustopac(species=['ac_opct', 'Draine_Si']):
    nspec = len(species)
    print 'writing dust opacity out'

    hline="-----------------------------------------------------------------------------\n"

    OPACOUT=open('dustopac.inp','w')

    lines0=["2               iformat (2)\n",
            str(nspec)+"               species\n",
            hline]
    OPACOUT.writelines(lines0)

    for i in xrange(nspec):
        lines=["1               in which form the dust opacity of dust species is to be read\n",
               "0               0 = thermal grains\n",
               species[i]+"         dustkappa_***.inp file\n",
               hline
           ]

        OPACOUT.writelines(lines)

    OPACOUT.close()




# -------------------------------------------------------------------
# produce the dust kappa files
# requires the .lnk files and the bhmie.f code inside a folder
# called opac
# -------------------------------------------------------------------
def make_dustkappa(species='Draine_Si', amin=0.1, amax=1000,
                   graindens=2.0, nbins=20, abin = 0, alpha=3.5, Plot=False):

    print 'making dustkappa files'

    os.chdir("opac/")
    pathout='dustkappa_'+species+str(abin)+'.inp'
    path="./"
    lnk_file=species
    Type=species
    Pa=(amax/amin)**(1.0/(nbins-1.0))

    A=np.zeros(nbins)
    A[0]=amin


    for i in range(nbins):
        os.system('rm '+path+'param.inp')
        A[i]=amin*(Pa**(i))
        acm=A[i]*10.0**(-4.0)
        print "  a = %1.2e um"  %A[i]
        file_inp=open(path+'param.inp','w')
        file_inp.write(lnk_file+'\n')
        e=round(np.log10(acm))
        b=acm/(10.0**e)
        file_inp.write('%1.2fd%i \n' %(b,e))
        file_inp.write('%1.2f \n' %graindens)
        file_inp.write('1')

        file_inp.close()

        os.system(path+'makeopac')

        os.system('mv '+path+'dustkappa_'+lnk_file+'.inp '+path+'dustkappa_'+Type+'_'+str(i+1)+'.inp ')



    #--------- READ OPACITIES AND COMPUTE MEAN OPACITY

    # read number of wavelengths
    opct=np.loadtxt(path+lnk_file+'.lnk')
    Nw=len(opct[:,0])

    Op=np.zeros((Nw,4))         # wl, kappa_abs, kappa_scat, g
    Op[:,0]=opct[:,0]

    Ws_mass=np.zeros(nbins)     # weigths by mass and abundances
    Ws_number=np.zeros(nbins)   # weights by abundances

    for i in xrange(nbins):
        Ws_mass[i]=(A[i]**(-alpha))*(A[i]**(3.0))*A[i]  # w(a) propto n(a)*m(a)*da and da propto a
        Ws_number[i]=A[i]**(-alpha)*A[i]

    W_mass=Ws_mass/np.sum(Ws_mass)
    W_number=Ws_number/np.sum(Ws_number)

    for i in xrange(nbins):
        file_inp=open(path+'dustkappa_'+Type+'_'+str(i+1)+'.inp','r')
        file_inp.readline()
        file_inp.readline()


        for j in xrange(Nw):
            line=file_inp.readline()
            dat=line.split()
            kabs=float(dat[1])
            kscat=float(dat[2])
            g=float(dat[3])

            Op[j,1]+=kabs*W_mass[i]
            Op[j,2]+=kscat*W_mass[i]
            Op[j,3]+=g*W_mass[i]

        file_inp.close()
        os.system('rm '+path+'dustkappa_'+Type+'_'+str(i+1)+'.inp')

    #---------- WRITE MEAN OPACITY

    if Plot:
        import matplotlib.pyplot as plt
        plt.plot(Op[:,0],Op[:,1])
        plt.xscale('log')
        plt.yscale('log')
        plt.show()


    final=open(path+pathout,'w')

    final.write('3 \n')
    final.write(str(Nw)+'\n')
    for i in xrange(Nw):
        final.write('%f \t %f \t %f \t %f\n' %(Op[i,0],Op[i,1],Op[i,2],Op[i,3]))
    final.close()

    os.system('mv '+path+pathout+' ../'+path+pathout)
    os.chdir("../")


# -------------------------------------------------------------------
# writing radmc3d.inp
# -------------------------------------------------------------------

def write_radmc3dinp(incl_dust = 1,
                     incl_lines = 0,
                     lines_mode = 1,
                     nphot = 1000000,
                     nphot_scat = 1000000,
                     nphot_spec = 1000000,
                     nphot_mono = 1000000,
                     istar_sphere = 1,
                     scattering_mode_max = 0,
                     tgas_eq_tdust = 1,
                     modified_random_walk = 0,
                     setthreads=2):

    print 'writing radmc3d.inp out'

    RADMCINP = open('radmc3d.inp','w')
    inplines = ["incl_dust = "+str(int(incl_dust))+"\n",
                "incl_lines = "+str(int(incl_lines))+"\n",
                'lines_mode = '+str(int(lines_mode))+'\n',
                "nphot = "+str(int(nphot))+"\n",
                "nphot_scat = "+str(int(nphot_scat))+"\n",
                "nphot_spec = "+str(int(nphot_spec))+"\n",
                "nphot_mono = "+str(int(nphot_mono))+"\n",
                "istar_sphere = "+str(int(istar_sphere))+"\n",
                "scattering_mode_max = "+str(int(scattering_mode_max))+"\n",
                "tgas_eq_tdust = "+str(int(tgas_eq_tdust))+"\n",
                "modified_random_walk = "+str(int(modified_random_walk))+"\n",
                "setthreads="+str(int(setthreads))+"\n"  ]

    RADMCINP.writelines(inplines)
    RADMCINP.close()







# -------------------------------------------------------------------
# writing dust and gas densities
# -------------------------------------------------------------------

def write_densities(F, R_Scaling, Plot=False):

    rho_gas = F.data_total
    ncells = F.nx*F.ny*F.nz

    # -------------------------------------------------------------------
    # unit conversion
    # -------------------------------------------------------------------
    rho_gas = rho_gas * M_Sun / (au * au * au)

    # scaling
    rho_gas = rho_gas / (R_Scaling * R_Scaling * R_Scaling)

    # fractions
    ep = 0.01                   # dust-to-gas ratio
    f_ac = 0.3
    f_si = 0.7
    f_h2 = 1 / (2*m_H);
    f_12co = 1e-4 / (2*m_H);
    f_13co = 1e-6 / (2*m_H);
    f_c18o = 1.7e-7 / (2*m_H);



    print 'writing dust and gas density fields'

    DUSTOUT    = open('dust_density.inp','w')
    GASOUTH2   = open('numberdens_h2.inp','w')
    GASOUT12CO = open('numberdens_12c16o.inp','w')
    GASOUT13CO = open('numberdens_13c16o.inp','w')
    GASOUTC18O = open('numberdens_12c18o.inp','w')

    DUSTOUT.write('1 \n')            # iformat
    DUSTOUT.write(str(ncells)+' \n') # n cells
    DUSTOUT.write(str(1)+' \n')      # n species

    GASOUTH2.write('1 \n')            # iformat
    GASOUTH2.write(str(ncells)+' \n') # n cells

    GASOUT12CO.write('1 \n')            # iformat
    GASOUT12CO.write(str(ncells)+' \n') # n cells
    GASOUT13CO.write('1 \n')            # iformat
    GASOUT13CO.write(str(ncells)+' \n') # n cells
    GASOUTC18O.write('1 \n')            # iformat
    GASOUTC18O.write(str(ncells)+' \n') # n cells

    if Plot:
        field = np.zeros((F.nx, F.ny, F.nz))

    # DENSITIES
    # only one dust species
    for k in range(F.nz):
        for j in range(F.ny):
            for i in range(F.nx):
                rho_ijk = rho_gas[j,i,k]

                if Plot:
                    field[i,j,k] = rho_ijk # to check field

                DUSTOUT.write(str(rho_ijk*ep)+' \n')
                GASOUTH2.write(str(rho_ijk*f_h2)+' \n')
                GASOUT12CO.write(str(rho_ijk*f_12co)+' \n')
                GASOUT13CO.write(str(rho_ijk*f_13co)+' \n')
                GASOUTC18O.write(str(rho_ijk*f_c18o)+' \n')

    DUSTOUT.close()
    GASOUTH2.close()
    GASOUT12CO.close()
    GASOUT13CO.close()
    GASOUTC18O.close()


    # --------------------------------------------------------------
    # plotting the field to check the model
    # --------------------------------------------------------------
    if Plot:
        # import matplotlib
        # print matplotlib.rcsetup.all_backends
        # matplotlib.use('GTK3Agg')
        # import aplpy
        import matplotlib.pyplot as plt

        # X Y (cartesian)
        P,R = np.meshgrid(F.zm, F.xm)
        X = R*np.cos(P)/au
        Y = R*np.sin(P)/au

        # field = np.log10(F.data)
        field2d = np.log10(field[:,0,:])
        plt.pcolormesh(X,Y,field2d,cmap='plasma')


        # X Z (cartesian)
        # T,R = np.meshgrid(F.ym, F.xm)
        # X = R*np.sin(T)/au
        # Y = R*np.cos(T)/au

        # X += F.ymax

        # field=F.data_total[:,:,0]
        # plt.pcolormesh(X,Y,field, cmap='viridis')

        # plt.imshow(np.sqrt(data), cmap='hot')
        plt.colorbar()
        plt.axis('equal')
        plt.show()





# -------------------------------------------------------------------
# writing dust temperatures
# -------------------------------------------------------------------

def write_temperatures(F, R_Scaling):

    # receives the energy field for an isothermal run, which
    # corresponds to the sound speed (C_s)

    c_s = F.data_total          # in Isothermal ene = c_s
    ncells = F.nx*F.ny*F.nz

    # Energy to Temperature
    c_s *= np.sqrt(G*M_Sun/au)
    temperature =  2.3 * m_H * (c_s)**2 / kB # K

    # scaling
    temperature = temperature / np.sqrt(R_Scaling)

    print 'writing temperature fields'

    TEMPOUT    = open('dust_temperature.dat','w')

    TEMPOUT.write('1 \n')            # iformat
    TEMPOUT.write(str(ncells)+' \n') # n cells
    TEMPOUT.write(str(1)+' \n')      # n species


    # TEMPERATURES
    # only one dust species
    for k in range(F.nz):
        for j in range(F.ny):
            for i in range(F.nx):
                T_ijk = temperature[j,i,k]
                TEMPOUT.write(str(T_ijk)+' \n')

    TEMPOUT.close()


# -------------------------------------------------------------------
# writing gas velocities
# -------------------------------------------------------------------

def write_velocities(VFx, VFy, VFz, R_Scaling):

    VX = VFx.data_total
    VY = VFy.data_total
    VZ = VFz.data_total
    ncells = VFx.nx*VFx.ny*VFx.nz

    # unit conversion from code to CGS
    VX *= np.sqrt(G*M_Sun/au)
    VY *= np.sqrt(G*M_Sun/au)
    VZ *= np.sqrt(G*M_Sun/au)
    VZ /= np.sqrt(R_Scaling) # scaling

    print 'writing velocity fields'

    VELOUT = open('gas_velocity.inp','w')
    VELOUT.write('1 \n')            # iformat
    VELOUT.write(str(ncells)+' \n') # n cells

    for k in range(VFx.nz):
        for j in range(VFx.ny):
            for i in range(VFx.nx):
                vr = VX[j,i,k]
                vt = VY[j,i,k]
                vp = VZ[j,i,k]
                VELOUT.write(str(vr)+'\t'+str(vt)+'\t'+str(vp)+' \n')

    VELOUT.close()



# -------------------------------------------------------------------
# writing gas microturbulence
# -------------------------------------------------------------------

def write_microturbulence(F):

    ncells = F.nx*F.ny*F.nz

    print 'writing microturbulence'

    # 0.05 km/s of microturbulence
    a = 0.05*1e5
    MTURBOUT = open('microturbulence.inp','w')
    MTURBOUT.write('1 \n')
    MTURBOUT.write(str(ncells)+' \n') # n cells

    for k in range(F.nz):
        for j in range(F.ny):
            for i in range(F.nx):
                MTURBOUT.write(str(a)+' \n')

    MTURBOUT.close()



# -------------------------------------------------------------------
# write extra info about the FARGO run to a file
# for example: position of the planet at a given output
# -------------------------------------------------------------------

def write_runinfo(F,directory, outputnumber):

    print 'calculating planet position in grid'

    with open(directory+'bigplanet0.dat','r') as bigplanet:
        for line in bigplanet:
            values = line.split()
            if values[0] == outputnumber:
                xp = float(values[1])
                yp = float(values[2])

    # position of the planet as angle from the x-axis
    angle = 180. * math.atan2(yp, xp) / pi
    phi_planet = 360. - (angle - 90.)
    if phi_planet > 360.:
        phi_planet = np.mod(phi_planet,360.)

    F.phi_planet = phi_planet
    print '\tplanet at phi='+str(phi_planet)

    RUNINFOUT = open('runinfo.inp','w')
    RUNINFOUT.write(str(angle)+' \n')
    RUNINFOUT.close()

    return phi_planet


class RTmodel():
    def __init__(self, Tstar=6000, Rstar=1.0,
                 distance = 140, label='run', simoutput=0,
                 npix=256, incl=30.0, posang=0.0, phi=0.0,
                 Lambda=800, sizeau=0.0, secondorder='secondorder',
                 line = '12co',
                 imolspec=1, iline=3, linenlam=80, widthkms=4,):
        # star pars
        self.Tstar = Tstar
        self.Rstar = Rstar*R_Sun
        # disk pars
        self.distance = distance * pc
        self.label = label
        self.simoutput = simoutput
        # RT pars
        self.Lambda = Lambda
        self.line = line
        self.npix   = npix
        self.incl   = incl
        self.posang = posang
        self.phi = phi
        self.sizeau = sizeau
        # line emission pars
        self.imolspec = imolspec
        self.iline    = iline
        self.widthkms = widthkms
        self.linenlam = linenlam
        self.secondorder = secondorder


def run_mctherm():
    os.system('radmc3d mctherm')


def run_raytracing(M):
    command='radmc3d image lambda '+str(M.Lambda)+' npix '+str(M.npix)+' incl '+str(M.incl)+' posang '+str(M.posang)+' phi '+str(M.phi)+' '+str(M.secondorder)
    print 'calculating continuum image with'
    print command
    os.system(command)


def run_raytracing_lines(M):
    command='radmc3d image incl '+str(M.incl)+' phi '+str(M.phi)+' npix '+str(M.npix)+' posang '+str(M.posang)+' imolspec '+str(M.imolspec)+' iline '+str(M.iline)+' widthkms '+str(M.widthkms)+' linenlam '+str(M.linenlam)+' '+str(M.secondorder)
    print 'calculating image cube with'
    print command
    os.system(command)


def run_raytracing_tau(M):
    command = 'radmc3d tausurf 1.0 incl '+str(M.incl)+' phi '+str(M.phi)+' npix '+str(M.npix)+' posang '+str(M.posang)+' imolspec '+str(M.imolspec)+' iline '+str(M.iline)+' widthkms '+str(M.widthkms)+' linenlam '+str(M.linenlam)+' '+str(M.secondorder)
    print 'calculating image cube with'
    print command
    os.system(command)



def get_outputname(M):

    return 'image_'+str(M.line)+'_o'+str(int(M.simoutput))+'_i'+str("{:.2f}".format(M.incl))+'_phi'+str("{:.2f}".format(M.phi))+'_PA'+str("{:.2f}".format(M.posang))+'.fits'


def exportfits(M, Plot=False):

    outfile = get_outputname(M)
    print 'exporting as '+str(outfile)
    infile = 'image.out'

    # read header info:
    # iformat <=== For now this is 1 (or 2 for local observer mode)
    # im_nx im_ny
    # nlam
    # pixsize_x pixsize_y
    # lambda[1] ......... lambda[nlam+1]
    f = open(infile,'r')
    iformat = int(f.readline())
    im_nx, im_ny = tuple(np.array(f.readline().split(),dtype=int))
    nlam = int(f.readline())
    pixsize_x, pixsize_y = tuple(np.array(f.readline().split(),dtype=float))
    lbda = np.empty(nlam)
    for i in range(nlam):
        lbda[i] = float(f.readline())
    f.readline()                # empty line

    # load image data
    images = np.loadtxt(infile, skiprows=(5+nlam))

    # calculate physical scales
    distance = M.distance
    pixsize_x_deg = 180.*pixsize_x / distance / pi
    pixsize_y_deg = 180.*pixsize_y / distance / pi

    pixsurf_ster = pixsize_x_deg*pixsize_y_deg * (pi/180.)**2
    fluxfactor = 1e23 * pixsurf_ster

    if nlam>1:
        im = images.reshape(nlam,im_ny,im_nx)
        naxis = 3
    else:
        im = images.reshape(im_ny,im_nx)
        naxis = 2

    if Plot:
        import matplotlib.pyplot as plt
        plt.imshow(im, cmap = 'plasma', origin='lower',aspect='auto')
        plt.axis('equal')
        plt.show()

    from astropy.io import fits

    hdu = fits.PrimaryHDU()

    # hdu.header['SIMPLE'] = 'T       '; # makes simobserve crash
    hdu.header['BITPIX'] = -32

    hdu.header['NAXIS'] = naxis
    hdu.header['NAXIS1'] = im_nx
    hdu.header['NAXIS2'] = im_ny
    hdu.header['EPOCH']  = 2000.0
    hdu.header['EQUINOX'] = 2000.0
    hdu.header['LONPOLE'] = 180.0
    # hdu.header['SPECSYS'] = 'LSRK    '
    hdu.header['CTYPE1'] = 'RA---TAN'
    hdu.header['CTYPE2'] = 'DEC--TAN'
    hdu.header['CRVAL1'] = float(0.0)
    hdu.header['CRVAL2'] = float(0.0)
    hdu.header['CDELT1'] = float(pixsize_x_deg)
    hdu.header['CDELT2'] = float(pixsize_y_deg)
    hdu.header['CUNIT1'] = 'DEG     '
    hdu.header['CUNIT2'] = 'DEG     '
    hdu.header['CRPIX1'] = float((im_nx-1)/2)
    hdu.header['CRPIX2'] = float((im_ny-1)/2)

    hdu.header['BUNIT'] = 'JY/PIXEL'
    hdu.header['BTYPE'] = 'Intensity'
    hdu.header['BSCALE'] = 1
    hdu.header['BZERO'] = 0

    print 'PIXSIZE '+str(pixsize_x_deg*3600)

    if nlam > 1:
        restfreq = c * 1e4 / lbda[int((nlam-1)/2)] # micron to Hz
        nus = c * 1e4 / lbda                       # Hx
        # dvel = (lbda[1] - lbda[0])*c*1e-5/lbda0
        dnu = nus[1] - nus[0]
        hdu.header['NAXIS3'] = int(nlam)
        hdu.header['CTYPE3'] = 'FREQ    '
        hdu.header['CUNIT3'] = 'Hz      '
        hdu.header['CRPIX3'] = float((nlam-1)/2)
        hdu.header['CRVAL3'] = float(restfreq)
        hdu.header['CDELT3'] = float(dnu)
        hdu.header['RESTFREQ'] = float(restfreq)

    hdu.data = (im*fluxfactor).astype('float32')

    # hdu.scale('float32')
    # print hdu.data.dtype

    print 'flux '+str(np.sum(hdu.data))

    hdu.writeto('products/'+outfile, output_verify='fix', clobber=True)
    # fits.update(outfile, hdu.data, hdu.header)


















# -------------------------------------------------------------------
# various utils for Field modification
# -------------------------------------------------------------------


def shift_field(Field,direction):
    """
    Half cell shifting along the direction provided by direction
    direction can be ('x','y', 'xy', 'yx').

    After a call of this function, Field.xm/xmed has not
    sense anymore (it is not hard to improve).
    """
    F = copy.deepcopy(Field)
    if direction.count('x')>0:
        F.data = 0.5*(Field.data[:,1:]+Field.data[:,:-1])
        F.x = 0.5*(Field.x[1:]+Field.x[:-1])
    if direction.count('y')>0:
        F.data = 0.5*(F.data[1:,:]+F.data[:-1,:])
        F.y = 0.5*(F.y[1:]+F.y[:-1])

    F.nx = len(F.x)
    F.ny = len(F.y)

    return F


def cut_field(Field, direction, side):
    """
    Cutting a field:
    Input: field --> a Field class
           axis  --> 'x', 'y' or 'xy'
           side  --> 'p' (plus), 'm' (minnus), 'pm' (plus/minnus)
    """

    cutted_field = copy.deepcopy(Field)
    ny,nx = Field.ny, Field.nx
    mx = my = px = py = 0

    if direction.count('x')>0:
        if side.count('m')>0:
            mx = 1
        if side.count('p')>0:
            px = 1
    if direction.count('y')>0:
        if side.count('m')>0:
            my = 1
        if side.count('p')>0:
            py = 1

    cutted_field.data = Field.data[my:ny-py,mx:nx-px]
    cutted_field.x = cutted_field.x[mx:nx-px]
    cutted_field.y = cutted_field.y[my:ny-py]

    return cutted_field











# def ExportFits(Field, filename):

#     F = copy.deepcopy(Field)
#     cube = F.data_total

#     from astropy.io import fits

#     hdu = fits.PrimaryHDU(cube)
#     hdu.writeto(filename)
