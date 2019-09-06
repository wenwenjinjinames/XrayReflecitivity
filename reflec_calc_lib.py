""" Reflectivity calculation for data fitting"""
import numpy as np
from scipy.special import erf

class ReflecCalc:
    """This is a Reflectivity Calculation for a given 
    stack of electron density, beta, and thickness"""
    _r0 = 2.82e-5 # Thomson electron radius; [r0] = Angstrom
    
    def __init__(self, energy, sub_rho, sub_beta = 0.0):
        """ All constants
        sub_rho -> subphase density
        sub_beta -> subphase beta
        """
        self._energy = energy
        self._lambda = 12.398/energy # wave length
        self._k0 = 2 * np.pi/self._lambda # wave-number
        self._sub_rho = sub_rho
        self._sub_beta = sub_beta
        self._alpha_c = np.sqrt(4*np.pi*self._sub_rho
            * ReflecCalc._r0) / self._k0

    def __repr__(self):
        str1 = 'x-ray energy: {} keV\n'.format(self._energy)
        str2 = 'x-ray wavelength : {} \u212b \n'.format(
            self._lambda)
        str3 =('Subphase electron density : '
              '{} e/\u212b\u00b3 \n'.format(
               self._sub_rho))
        str4 =('Core algorithm : '
              'Parratt\'s recursive method\n')

        return str1 + str2 +str3 + str4
            
    def RF_calc(self, qz):
        """
        Calculate the Fresnel Reflectivity for 
        a given subphase at a sequence of qz
        """
        Qc = 2*self._k0*np.sin(self._alpha_c)
        Qp_sq = (qz**2 - Qc**2 
            +8j * (self._k0**2) * self._sub_beta)
        Qp = np.sqrt(Qp_sq)
        r = (qz-Qp) /(qz + Qp)
        r_sq = np.absolute(r)**2
        return r_sq
        
    def fitReflec_norm(self, qz, *params):
        """ 
        Interface to optimization
        Arguments: qz, [d1, rho1, d2, rho2, ... dN, rhoN, roughness]
        """
        ds = np.array(params[0:-1:2])
        rhos = np.array(params[1:-1:2])
        sigma = params[-1]
        betas = np.zeros_like(ds);
        return ReflecCalc._reflec_calc(self, 
            qz, ds, rhos, betas, sigma, norm = True)
        
    
    def _reflec_calc(self, qz, ds, rhos, betas, sigma = 0, norm = True):    
        """ Input a sequence of parameters to 
        calculate the reflectivity
        ds-> thickness of layers
        rhos-> density of layers
        betas-> beta of layers
        sigma -> roughness
        norm-> True ->return normalized reflectivity
        """  
        N = len(ds)
        rho_e = np.zeros(N+2)
        beta = np.zeros(N+2)
        delta = np.zeros(N+2)
        d_array = np.zeros(N+2)
        for i in range(1,N+1):
            rho_e[i] = rhos[i-1]
            beta[i] = beta[i-1]
            delta[i] = (2*np.pi*rho_e[i]
                    * ReflecCalc._r0/self._k0**2)
            d_array[i] = ds[i-1]
        
        # Below is subphase
        rho_e[N+1]=self._sub_rho
        beta[N+1]=self._sub_beta
        delta[N+1]=(2*np.pi*rho_e[N+1]* ReflecCalc._r0/
                        self._k0**2)
        d_array[N+1] = np.inf
        alpha_arr = np.arcsin(qz/2/self._k0)
        
        Ref=np.zeros_like(qz)
        for (index, alpha) in enumerate(alpha_arr):
            k_array = self._k0*np.sqrt(np.sin(alpha)**2-2*delta
                                       + 2j*beta)
            r = ReflecCalc._r0_Parratts(d_array, k_array)
            Ref[index]= np.absolute(r)**2
        DW = np.exp(-(qz*sigma)**2)
        
        if norm:
            return Ref/ReflecCalc.RF_calc(self,qz)*DW
        else:
            return Ref*DW
        
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
        
        r[0] = 0
        R[0] = 0
        R[1] = (k[1]-k[0])/(k[1]+k[0])
        r[1] = R[1]
        for i in range(2, len(d)):
            R[i] = (k[i]-k[i-1])/(k[i]+k[i-1])
            num = R[i] + r[i-1]*np.exp(2j*k[i-1]*d[i-1])
            den = 1 + R[i]*r[i-1]*np.exp(2j*k[i-1]*d[i-1])
            r[i]=num/den
        return r[-1]
        
    
class Subphase:
    """
    Define subphase class:
    1st parameter : e-density
    2nd parameter : beta
    """
    def __init__(self, rho_e, beta=0.0 ):
        self.rho_e = rho_e # subphase electrond density
        self.beta = beta  # absorption parameter of the subphase
    def get_rho(self):
        return self.rho_e
    def get_beta(self):
        return self.beta
    def __repr__(self):
        s1 = ('Subphase electron density : '
        '{} e/\u212b\u00b3 \n'.format(self.rho_e))
        s2 = ('subphase beta : '
              '{}'.format(self.beta))
        return s1 + s2

class Layer:
    """
    Define a layer with 3 parameters:
    1st parameter: thickness
    2nd parameter: e-density
    3rd parameter: beta
    """
    def __init__(self, t, rho_e, beta=0.0):
        self.t = t # that is the thickness of the layer
        self.rho_e = rho_e # electron density of this layer
        self.beta = beta # beta of this layer
    def get_rho(self):
        return self.rho_e
    def get_beta(self):
        return self.beta
    def __repr__(self):
        s1 = 'Layer density : {} e/\u212b\u00b3\n'.format(self.rho_e)
        s2 = 'Layer thickness: {} \u212b'.format(self.t)         
        return s1 + s2

class Film:
    """ 
    A Film class-object has two components:
    1: subphase-obj
    2: overall roughness
    3: a list of layers
    """
    def __init__(self, subphase, roughness, layers = []):
        self.layers = layers # a sequence of layer-obj
        self.subphase = subphase
        self.sigma = roughness # interfacial roughness
    
    def __repr__(self):
        s1="A film: number of layers: {}\n".format(len(self.layers))
        s2="\tSubphase e-density is: {}".format(
                self.subphase.get_rho()) 
        return s1+s2
                                    
    
    def get_ED_profile(self, z_profile):
        """Return ED of given depth profile"""
        # By default, z =0 is the top-most interface coordinate
        # create interface coordinate first
        # Get a ED-profile give the depth profile
        
        interfaces_coords = [0]  # All interface depth
        z = 0
        rho =[0]
        rho_profile=np.zeros_like(z_profile)
        
        # What if there is no film
        if not self.layers:
            rho.append(self.subphase.get_rho())
            zp = z_profile/np.sqrt(2)/self.sigma
            rho_profile = (0.5*erf(zp)*(rho[0]-rho[-1]) 
                           + 0.5*rho[-1])
            return rho_profile
        
        # What if there is a stack of layers
        for layer in self.layers:
            z -= layer.t # each layer interface position
            interfaces_coords.append(z)
            rho.append(layer.get_rho())
        rho.append(self.subphase.get_rho())
        
        
        for i in range(len(self.layers)+1):
            zp=(z_profile-interfaces_coords[i])\
                /self.sigma/np.sqrt(2)
            rho_profile = rho_profile + \
                0.5*erf(zp)*(rho[i]-rho[i+1])
                
        rho_profile += 0.5*rho[-1]
        return rho_profile



if __name__ == '__main__':
    reflec_calc = ReflecCalc(energy = 8.0, sub_rho=0.334)
    print(reflec_calc)
