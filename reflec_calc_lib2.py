"""
Reflectivity calculation for data fitting:
Error function based Electron density profile construction
"""
import numpy as np
from scipy.special import erf
 

class ReflecModel:
    """This is a Reflectivity Calculation for a given 
    stack of electron density, beta, thickness and 
    interfacial roughness.
    """
    _r0 = 2.82e-5 # Thomson electron radius; [r0] = Angstrom
    
    def __init__(self, energy, sub_rho, sub_beta = 0.0, d_slab=1, normalize=False):
        """ All constants
        sub_rho -> subphase density
        sub_beta -> subphase beta
        d_slab -> each slab's thickness
        normalize -> expect R/RF if true
        """
        self._energy = energy # in unit of keV
        self._lambda = 12.398/energy # wave length
        self._k0 = 2 * np.pi/self._lambda # wave-number
        self._sub_rho = sub_rho
        self._sub_delta= 2*np.pi*sub_rho*ReflecModel._r0/self._k0**2
        self._sub_beta = sub_beta
        self._alpha_c = np.sqrt(4*np.pi*self._sub_rho
            * ReflecModel._r0) / self._k0
        self._Qc=2*self._k0*np.sin(self._alpha_c)
        #self._n_slabs = n_slabs # number of slabs to compute reflectivity
        self._d_slab=d_slab # thickness of each slab
        self.normalize=normalize # return R/RF if True, otherwise just R.
        
        
    def __repr__(self):
        str1 = 'x-ray energy: {} keV\n'.format(self._energy)
        str2 = 'x-ray wavelength : {} \u212b \n'.format(
            self._lambda)
        str3 =('Subphase electron density : '
              '{} e/\u212b\u00b3 \n'.format(
               self._sub_rho))
        str4=('Critical Qz : {:6.4f} 1/\u212b\n'.format(self._Qc))
        str5 =('Core algorithm : '
              'Parratt\'s recursive method\n')
        str6 =('slabs of thickness {} \u212b to approximate ED profile : '.format
                (self._d_slab))

        return str1 + str2 +str3 + str4 + str5 + str6
            
    def RF_calc(self, qz):
        """
        Calculate the Fresnel Reflectivity for 
        a given subphase at a sequence of qz.
        _sq -> squared
        """
        Qc = 2*self._k0*np.sin(self._alpha_c)
        Qp_sq = (qz**2 - 8*(self._k0**2)*self._sub_delta
            +8j * (self._k0**2) * self._sub_beta)
        Qp = np.sqrt(Qp_sq)
        r = (qz-Qp) /(qz + Qp)
        r_sq = np.absolute(r)**2
        return r_sq
        
    @staticmethod
    def _r0_Parratts(d_array, k_array):
        """
        Return reflectivity amplitude given a sequence of 
        thickness (d_array) and wavenumber (k_array)
        d_array[0] and k_array[0] -> vaccum/vapor phase 
        """ 
        r = np.zeros_like(d_array, dtype=np.complex)
        R = np.zeros_like(d_array, dtype=np.complex)
        # There is no reflection amplitude in 
        # last layer (subphase)
        d = np.flip(d_array, axis=0)
        k = np.flip(k_array, axis=0)
        
        r[0] = 0 # here r[0] is from the subphase
        R[0] = 0
        R[1] = (k[1]-k[0])/(k[1]+k[0])
        r[1] = R[1]
        for i in range(2, len(d)):
            R[i] = (k[i]-k[i-1])/(k[i]+k[i-1])
            num = R[i] + r[i-1]*np.exp(2j*k[i-1]*d[i-1])
            den = 1 + R[i]*r[i-1]*np.exp(2j*k[i-1]*d[i-1])
            r[i]=num/den
        return r[-1]
    
         
    def _reflec_calc(self, qz, ds, rhos):    
        """ Input a sequence of parameters to 
        calculate the reflectivity
        ds-> thickness of layers
        rhos-> density of layers
        betas-> beta of layers
        """  
        N = len(ds) # N-> number of layers
        rho_e = np.zeros(N+2)
        beta = np.zeros(N+2)  # No beta contribution here
        delta = np.zeros(N+2)
        d_array = np.zeros(N+2)
        # the zero-th layer is vapor phase: all values are zero
        for i in range(1,N+1):
            rho_e[i] = rhos[i-1] # rho_e[1]=rhos[0]-> first layer
            delta[i] = (2*np.pi*rho_e[i]
                    * ReflecModel._r0/self._k0**2)
            d_array[i] = ds[i-1]
        
        # Below is subphase
        rho_e[N+1]=self._sub_rho
        beta[N+1]=self._sub_beta
        delta[N+1]=(2*np.pi*rho_e[N+1]* ReflecModel._r0/
                        self._k0**2)
        d_array[N+1] = np.inf
        alpha_arr = np.arcsin(qz/2/self._k0)
        
        Ref=np.zeros_like(qz)
        for (index, alpha) in enumerate(alpha_arr):
            k_array = self._k0*np.sqrt(np.sin(alpha)**2-2*delta
                                       + 2j*beta)
            r = ReflecModel._r0_Parratts(d_array, k_array)
            Ref[index]= np.absolute(r)**2
       
        return Ref
   
    @staticmethod     
    def _EDprofile(z, N, rho_array, d_array, sigma_array):
        """
        Provide an ED profile given:
        N: number of layers
        rho_array: rho[0] -> vapor phase
        d_array: 
        sigma_array: sigma[0] -> top-most interface roughness
        """
        rho=np.zeros_like(z)
        Z_array=np.zeros_like(rho_array)
        for i in range(1,N+1): # start at index 1, until N
            Z_array[i]=Z_array[i-1] - d_array[i]
        for i in range(N+1):
            term1=erf((z-Z_array[i])/np.sqrt(2)/sigma_array[i])
            term2=rho_array[i]-rho_array[i+1]
            rho=rho + 0.5*term1*term2
        rho=rho+rho_array[N+1]/2
        return rho
     
       
    def fitReflec(self, qz, Isca, qz_offset, *params):
        """
        Compute the reflectivity given the following:
        Isca-> scaling factor of the overall reflectivity
        qz_offset-> Qz_observe = Qz_true +qz_offset
        0 box: params=[sigma_0]
        1 box: params=[sigma_0, d1, rho1, sigma_1]
        2 box: params=[sigma_0, d1, rho1, sigma_1, d2, rho2, sigma_2]
        """

        sigma_array=params[0::3] 
        d_array=[0.0] # Not used, just a placeholder for vapor phas
        rho_array=[0.0] # start from vapor phase
        N=len(sigma_array)-1 # Number of layers
        if N>0:
            d_array.extend(params[1::3]) # layer parameter
            rho_array.extend(params[2::3]) # layer parameter
        rho_array.append(self._sub_rho) # ended with subphase ED

        Zs=np.arange(3*sigma_array[0],
                    -np.sum(d_array)-3*sigma_array[-1], 
                    -self._d_slab)
        ds=np.ones_like(Zs)*self._d_slab
        rhos=ReflecModel._EDprofile(Zs,N,rho_array, d_array, sigma_array)
        Qz=qz-qz_offset
        if self.normalize:
            return Isca*self._reflec_calc(Qz, ds, rhos)/self.RF_calc(Qz)
        else:
            return Isca*self._reflec_calc(Qz, ds, rhos)
           
    def renderEDprofile(self, z, *params):
        """
        Provide the ED profiles with best-fit parameters for
        fitReflec(...)
        Basically build:
        1. sigma_array
        2. d_array
        3. rho_array
        """
        sigma_array=[val for val in params[2::3]]
        N=len(sigma_array)-1
        d_array=[val for val in params[3::3]]
        d_array.insert(0,0.0) # first element is for vapor phase
        rho_array=[val for val in params[4::3]]
        rho_array.insert(0, 0.0)
        rho_array.append(self._sub_rho)
        return self._EDprofile(z,N, rho_array, d_array, sigma_array)
        
        