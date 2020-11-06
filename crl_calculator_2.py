"""
Calculator for CRL focus positions and beam sizes for the HED instrument.
Copyright Thomas Preston (c) 2020, European X-Ray Free-Electron Laser Facility GmbH
All rights reserved.
"""

import sys
import numpy as np
import scipy
from scipy.optimize import leastsq

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QLineEdit, \
QGridLayout, QCheckBox, QGroupBox, QHBoxLayout, QVBoxLayout

import matplotlib 
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

"""Formulae to calculate focal length for radius of curvature for Beryllium"""
def nmtoev(nm):
    """Converts eV to nm and vice versa"""
    return 1239.84193/nm

classical_elec = 2.8179e-15 # m classical electron radius
z_Be = 4. # Be charge
rho_Be = 1.85 # g/cc
atomw_Be = 9.012 # g/mol
Avogadro = 6.022e23 # mol-1
natom_Be = rho_Be*Avogadro/atomw_Be # cm-3 ion density of Be

# https://en.wikipedia.org/wiki/Refractive_index#Complex_refractive_index
def delta(energy):
    """Returns refractive index delta of Be for photon energy (eV)."""
    return classical_elec*(nmtoev(energy)*1e-9)**2 *z_Be*natom_Be*1e6/(2.*np.pi)

# OPTICS LETTERS / Vol. 27, No. 9 / May 1, 2002
def foc_length(roc, energy):
    """Returns focal length of a spherical biconcave lens for roc (mm) and photon energy (eV)."""
    return 0.5*roc*1e-3/delta(energy)

def image_dist(obj, foc):
    """Return image distance for object distance and focal length."""
    if foc == 0.:
        foc = np.inf
    return (1./foc - 1./obj)**-1

# https://en.wikipedia.org/wiki/Diffraction-limited_system
def diffr_lim(energy, beam_div):
    """Returns diffraction limited size in um for a beam of energy (eV) and divergence (urad)."""
    return 0.5*nmtoev(energy)*1e-3/np.sin(beam_div*1e-6)


"""Fixed lens parameters and component positions
# Reference - https://confluence.desy.de/pages/viewpage.action?pageId=137171886
"""
defenergy = 9000 # default energy (eV)
defbndwdth = 0.5 # default bandwidth/FWHM (%)
defsource_pos = -35 # default source pos (m)
defsource_sz = 0 # default source size (um)
defbeam_div = 2.5 # default beam divergence (urad)
defcrl3_posz_shft = 0 # default CRL3 shift (mm)
deffel_beamsz = 500 # default beam size at FEL imager (no lens) um
deffel_beamsz_wl = 500 # default beam size at FEL imager (w. lens) um
defhed_beamsz_wl = 500 # default beam size at HED imager (w. lens) um

crl1_posz = 229.0 #m
crl1roc = {} # mm
crl1roc[2] = 5.8*np.ones(1)
crl1roc[3] = 5.0*np.ones(1)
crl1roc[4] = 4.0*np.ones(1)
crl1roc[5] = 3.5*np.ones(1)
crl1roc[6] = 5.8*np.ones(2)
crl1roc[7] = 4.0*np.ones(3)
crl1roc[8] = 4.0*np.ones(7)
crl1roc[9] = 2.0*np.ones(7)

crl2_posz = 857.0 #m
crl2roc = {} # mm
crl2roc[1] = 5.8*np.ones(1)
crl2roc[2] = 5.0*np.ones(1)
crl2roc[3] = 4.0*np.ones(1)
crl2roc[4] = 3.5*np.ones(1)
crl2roc[5] = 5.8*np.ones(2)
crl2roc[6] = 5.8*np.ones(4)
crl2roc[7] = 5.8*np.ones(7)
crl2roc[8] = 4.0*np.ones(10)
crl2roc[9] = 3.5*np.ones(10)
crl2roc[10] = 2.0*np.ones(8)

crl3_posz = 962.325 #m
crl3roc = {} # mm
crl3roc[1] = 5.8*np.ones(1)
crl3roc[2] = 5.8*np.ones(3)
crl3roc[3] = 5.8*np.ones(4)
crl3roc[4] = 4.0*np.ones(10)
crl3roc[5] = 2.0*np.ones(10)
crl3roc[6] = 1.0*np.ones(10)
crl3roc[7] = 1.0*np.ones(10) # used to be 0.5
crl3roc[8] = 0.5*np.ones(10)
crl3roc[9] = 0.5*np.ones(10)
crl3roc[10] = 5.8*np.ones(2)

# FEL imager position 
fel_posz = 242.000 # m
# Mirrors 1, 2 and 3 positions
m1_posz = 290.0 # m
m2_posz = 301.36 # m
m3_posz = 390.0 # m
# Pop-in position after M1 and M2
pop1_posz = 303.682 # m
# Pop-in position after M3
pop2_posz = 400.000 # m 
# Mono positions 
mono_posz = 853.5 # m 
hrmono_posz = 855.8 # m 
# HED pop-in position
hed_posz = 939.000 # m
# TCC position (IC1)
tcc_posz = 971.300 # m
# XTD6 shutter
xtds_posz = 940.000 # m
# OPT shutter
opts_posz = 968.000 # m
# Beamstop 
ibss_posz = 980.000 # m

"""Formulae"""
def calc_crlfoc(energy, crlroc):
    """Convert radii of curvature to focal lengths at fixed photon energy (eV)."""
    crlfoc = {}
    for cc in crlroc:
        crlfoc[cc] = np.round(1./np.sum(1./foc_length(crlroc[cc], energy)), 3)
    return crlfoc

def calc_totalf(crllens, crlfoc):
    """Calculate total focal lengths for lens combinations."""
    f_crl = 0.
    if crllens.size != 0:
        for cc in crllens:
            f_crl += 1./crlfoc[cc]
        f_crl = 1./f_crl
    f_crl = np.round(f_crl, 3)
    return f_crl

def return_all_f_crl(energy, crl1lens, crl2lens, crl3lens):
    """Convert lens settings into focal lengths for energy."""
    crl1foc = calc_crlfoc(energy, crl1roc) # Calculate focal length of each lens arm
    crl2foc = calc_crlfoc(energy, crl2roc)
    crl3foc = calc_crlfoc(energy, crl3roc)
    f_crl1 = calc_totalf(crl1lens, crl1foc) # chosen config
    f_crl2 = calc_totalf(crl2lens, crl2foc)
    f_crl3 = calc_totalf(crl3lens, crl3foc)
    return f_crl1, f_crl2, f_crl3

def free_space_matrix(dist):
    """https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis for free space propogation"""
    return np.array([[1.0, dist], 
                     [0.0, 1.0]])

def lens_matrix(flen):
    """https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis for thin lens."""
    if flen != 0:
        mat = np.array([[1.0, 0.0], 
                        [-1.0/flen, 1.0]])
    else:
        mat = np.array([[1.0, 0.0], 
                        [0.0, 1.0]])
    return mat

"""Ray propogation with matrix formalism."""
def ray_trans_matrix(position, source_pos, f_crl1, f_crl2, f_crl3, crl3_posz_shft=0):
    """Only accepts single positions (not arrays) and calculates the ray transfer matrix:
    https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis"""
    if position < crl1_posz:
        mat = free_space_matrix(position - source_pos)
    elif position >= crl1_posz:
        mat = free_space_matrix(crl1_posz - source_pos)
        mat = np.matmul(lens_matrix(f_crl1), mat)
        if position < crl2_posz:
            mat = np.matmul(free_space_matrix(position - crl1_posz), mat)
        elif position >= crl2_posz:
            mat = np.matmul(free_space_matrix(crl2_posz - crl1_posz), mat)
            mat = np.matmul(lens_matrix(f_crl2), mat)
            if position < crl3_posz+crl3_posz_shft:
                mat = np.matmul(free_space_matrix(position - crl2_posz), mat)
            elif position >= crl3_posz+crl3_posz_shft:
                mat = np.matmul(free_space_matrix(crl3_posz+crl3_posz_shft - crl2_posz), mat)
                mat = np.matmul(lens_matrix(f_crl3), mat)
                mat = np.matmul(free_space_matrix(position - (crl3_posz+crl3_posz_shft)), mat)
    return mat

def check_foc_pos(source_pos, f_crl1, f_crl2, f_crl3, crl3_posz_shft):
    """Check position of foci for each CRL lens set)"""
    crl3_posz_new = crl3_posz
    if abs(crl3_posz_shft) <= 0.5:
        crl3_posz_new = crl3_posz_new + crl3_posz_shft # Shift CRL3
    else:
        textout += "WARNING: Requested CRL3 shift "+str(crl3_posz_shft*1000)+" is > 500mm\n"
        crl3_posz_shft = 0
    crl1_img_dist = crl1_posz + image_dist(obj=crl1_posz-source_pos, foc=f_crl1) # calculate image distance for focal length
    crl2_img_dist = crl2_posz + image_dist(obj=crl2_posz-crl1_img_dist, foc=f_crl2) 
    crl3_img_dist = crl3_posz_new + image_dist(obj=crl3_posz_new-crl2_img_dist, foc=f_crl3)
    return crl1_img_dist, crl2_img_dist, crl3_img_dist

def ray_propogation(energy, bndwdthev, source_pos, source_sz, beam_div, f_crl1, f_crl2, f_crl3, crl3_posz_shft, des_posz, textout):
    """Ray propogation and calculation of beam sizes along beamline. Returns beam sizes at positions along beamline."""

    crl3_posz_new = crl3_posz
    if abs(crl3_posz_shft) <= 0.5:
        crl3_posz_new = crl3_posz_new + crl3_posz_shft # Shift CRL3
    else:
        textout += "WARNING: Requested CRL3 shift "+str(crl3_posz_shft*1000)+" is > 500mm\n"
        crl3_posz_shft = 0
        
    init_vec = np.array([source_sz, beam_div*1e6]) # Initial beam vector
    key_comps = np.array([source_pos, crl1_posz, crl2_posz, crl3_posz_new, m1_posz, m2_posz, m3_posz, mono_posz, hrmono_posz, \
                          tcc_posz, xtds_posz, opts_posz, ibss_posz, des_posz])
    key_comps_names = np.array(["Source", "CRL1", "CRL2", "CRL3", "M1", "M2", "M3", "Mono", "HRMono", \
                                "IC1 TCC", "XTD shutter", "OPT shutter", "IBS", "Chosen pos."])
    # Check key components for safety
    szth = 200. # um
    for pos in np.arange(1, len(key_comps)):
        beam_sz = np.dot(ray_trans_matrix(key_comps[pos], source_pos, f_crl1, f_crl2, f_crl3, crl3_posz_shft), init_vec)[0]
        beam_sz = abs(beam_sz)
        if beam_sz < szth:
            textout += "WARNING: Beam size at "+key_comps_names[pos]+" "+str(np.round(beam_sz, 2))+"um\n"
        if key_comps[pos] != 0.0 and pos == len(key_comps)-1:
            textout += "Beam size at "+str(key_comps[pos])+"m "+str(np.round(beam_sz, 2))+"um\n"
            
    # Check diffraction limits
    crl1_img_dist, crl2_img_dist, crl3_img_dist = check_foc_pos(source_pos, f_crl1, f_crl2, f_crl3, crl3_posz_shft)
    beam_d = np.dot(ray_trans_matrix(source_pos, source_pos, f_crl1, f_crl2, f_crl3, crl3_posz_shft), init_vec)[1]
    textout += "Diffraction limit at source "+str(np.round(diffr_lim(energy, abs(beam_d)), 0))+"um\n"
    beam_d = np.dot(ray_trans_matrix(crl1_img_dist-1, source_pos, f_crl1, f_crl2, f_crl3, crl3_posz_shft), init_vec)[1]
    if f_crl1 != 0:
        textout += "Diffraction limit at CRL1 focus "+str(np.round(diffr_lim(energy, abs(beam_d)), 0))+"um\n"
    beam_d = np.dot(ray_trans_matrix(crl2_img_dist-1, source_pos, f_crl1, f_crl2, f_crl3, crl3_posz_shft), init_vec)[1]
    if f_crl2 != 0:
        textout += "Diffraction limit at CRL2 focus "+str(np.round(diffr_lim(energy, abs(beam_d)), 2))+"um\n"
    beam_d = np.dot(ray_trans_matrix(crl3_img_dist-1, source_pos, f_crl1, f_crl2, f_crl3, crl3_posz_shft), init_vec)[1]
    if f_crl3 != 0:
        textout += "Diffraction limit at CRL3 focus "+str(np.round(diffr_lim(energy, abs(beam_d)), 2))+"um\n"
    
    # Full calculation
    positions = np.arange(int(source_pos/10.0), 94)*10.0 # Every 10m up to XTD shutter
    positions = np.append(positions, np.arange(9400, 9800)/10.0) # Every 0.1m up to IBS
    positions = np.append(positions, key_comps) # Key components
    positions = np.asarray(sorted(positions)) # Sort into ascending order
    
    beam_vec = {}
    for pos in positions:
        beam_vec[pos] = np.dot(ray_trans_matrix(pos, source_pos, f_crl1, f_crl2, f_crl3, crl3_posz_shft), init_vec)
    new_pos = np.array([key for key in beam_vec.keys()]) # Dict only takes unique positions
    beam_vecs = np.array([(val[0], val[1]) for val in beam_vec.values()]).T # Values
    beam_vecs = np.append(new_pos, beam_vecs)
    beam_vecs = beam_vecs.reshape(3, len(new_pos))
    
    return beam_vecs, textout   

def calculate(energy, bndwdth, crl1lens, crl2lens, crl3lens, source_pos, source_sz, beam_div, crl3_posz_shft, des_posz):
    """First calculates focal lengths for this energy (eV). Then the total focal length of each lens set for the 
    chosen lens configuration crl1lens, crl2lens, crl3lens. Then propogates beam sizes through this for a source 
    size, position, and beam divergence."""
    bndwdthev = np.round(0.5*1e-2*bndwdth*energy, 0) # Convert to HWHM in eV
    f_crl1, f_crl2, f_crl3 = return_all_f_crl(energy, crl1lens, crl2lens, crl3lens)
    f_crl1p, f_crl2p, f_crl3p = return_all_f_crl(energy+bndwdthev, crl1lens, crl2lens, crl3lens)
    f_crl1m, f_crl2m, f_crl3m = return_all_f_crl(energy-bndwdthev, crl1lens, crl2lens, crl3lens)
    textout = "Energy "+str(energy)+" +/- "+str(bndwdthev)+" eV\n"
    textout += "Focal lengths "+str(f_crl1)+", "+str(f_crl2)+", "+str(f_crl3)+" m\n"
    textout += "Source "+str(np.round(source_pos, 3))+" m, Beam div. "+str(np.round(beam_div*1e6, 2))+" urad\n"
    crl1_img_dist, crl2_img_dist, crl3_img_dist = check_foc_pos(source_pos, f_crl1, f_crl2, f_crl3, crl3_posz_shft)
    crl1_img_distp, crl2_img_distp, crl3_img_distp = check_foc_pos(source_pos, f_crl1p, f_crl2p, f_crl3p, crl3_posz_shft)
    crl1_img_distm, crl2_img_distm, crl3_img_distm = check_foc_pos(source_pos, f_crl1m, f_crl2m, f_crl3m, crl3_posz_shft)
    # Add here for bandwidth
    if f_crl1 != 0:
        textout += "CRL 1 focus at "+str(np.round(crl1_img_dist, 3))+" m + " +str(np.round((crl1_img_distp-crl1_img_dist)*1e3, 0))+" mm - "+str(np.round((crl1_img_dist-crl1_img_distm)*1e3, 0))+" mm\n"
    if f_crl2 != 0:
        textout += "CRL 2 focus at "+str(np.round(crl2_img_dist, 3))+" m + " +str(np.round((crl2_img_distp-crl2_img_dist)*1e3, 0))+" mm - "+str(np.round((crl2_img_dist-crl2_img_distm)*1e3, 0))+" mm\n"
    if f_crl3 != 0:
        textout += "CRL 3 focus at "+str(np.round(crl3_img_dist, 3))+" m + " +str(np.round((crl3_img_distp-crl3_img_dist)*1e3, 0))+" mm - "+str(np.round((crl3_img_dist-crl3_img_distm)*1e3, 0))+" mm\n"
        textout += "Shift CRL3 by "+str(np.round(tcc_posz - crl3_img_dist, 3)*1e3)+" mm\n"
    #crl1foc = calc_crlfoc(energy, crl1roc) # Calculate focal length of each lens arm
    #crl2foc = calc_crlfoc(energy, crl2roc)
    #crl3foc = calc_crlfoc(energy, crl3roc)
    #f_crl1 = calc_totalf(crl1lens, crl1foc) # chosen config
    #f_crl2 = calc_totalf(crl2lens, crl2foc)
    #f_crl3 = calc_totalf(crl3lens, crl3foc)
    beam_vecs, textout = ray_propogation(energy, bndwdthev, source_pos, source_sz, beam_div, f_crl1, f_crl2, f_crl3, crl3_posz_shft, des_posz, textout)
    return beam_vecs[:2], textout

def fit_linear(params, xi, yi=0):
    """Minimise function ycalc - ydata = 0."""
    return params[0]*xi + params[1] - yi

def calc_div(energy, crl1lens, fel_beamsz, hed_beamsz_wl):
    """Calculates beam divergence and source position for the chosen set-up)"""
    crl1foc = calc_crlfoc(energy, crl1roc) # Calculate focal length of each lens arm
    f_crl1 = calc_totalf(crl1lens, crl1foc) # chosen config
    posz = np.array([fel_posz, hed_posz])
    beamszs = np.array([fel_beamsz, hed_beamsz_wl])
    errszs = 0.1*beamszs # set to be 10%
    toterrsz = np.sqrt(np.dot(errszs, errszs)) # Sum in quadrature
    params = 1, 0
    fit, fiterr = scipy.optimize.leastsq(fit_linear, params, args=(posz, beamszs))
    imagedist = -fit[1]/fit[0] - crl1_posz
    errimagedist = toterrsz/fit[0] # convert to error in distance
    objdist = image_dist(obj=imagedist, foc=f_crl1)
    errobjdist = abs(errimagedist*(objdist/imagedist)**2) # Convert to error in source point from lens formula
    newsourcepos = np.round(crl1_posz - objdist, 0)
    #print("Image dist.", imagedist, "Focal length", f_crl1)
    newbeamdiv = np.round(fel_beamsz/(fel_posz-newsourcepos), 2)
    errbeamdiv = abs(errszs[0]/(fel_posz-newsourcepos) + newbeamdiv*errobjdist/(fel_posz-newsourcepos)) # error in divergence
    print("Calculate source pos. and beam div.")
    print("Source pos.", newsourcepos, "m, error", np.round(errobjdist, 1), "m")
    print("Beam divergence", newbeamdiv, "urad, error", np.round(errbeamdiv, 2), "urad")
    return newsourcepos, newbeamdiv, np.round(errobjdist, 1), np.round(errbeamdiv, 2)
    
"""Here is the GUI part"""
class MainWindow(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        
        self.setWindowTitle("CRL Calculator")
        
        centralLayout = QGridLayout()
        centralLayout.setSpacing(2)
        
        topLayout = QGridLayout()
        self.msg0 = QLabel("Calculator for CRL focus positions and beam sizes for the HED instrument.\n"\
        "Copyright Thomas Preston (c) 2020, European X-Ray Free-Electron Laser Facility GmbH\n"\
        "All rights reserved.")
        topLayout.addWidget(self.msg0, 1, 0)
        
        
        self.energy = QLineEdit(str(defenergy))
        centralLayout.addWidget(QLabel("Energy (eV):"), 1, 0)
        centralLayout.addWidget(self.energy, 1, 1)
        self.energy.editingFinished.connect(self.calculate_click)
        
        self.bndwdth = QLineEdit(str(defbndwdth))
        centralLayout.addWidget(QLabel("Bandwidth (%):"), 2, 0)
        centralLayout.addWidget(self.bndwdth, 2, 1)
        self.bndwdth.editingFinished.connect(self.calculate_click)
        
        self.source_pos = QLineEdit(str(defsource_pos))
        centralLayout.addWidget(QLabel("Source position (m):"), 3, 0)
        centralLayout.addWidget(self.source_pos, 3, 1)
        self.source_pos.editingFinished.connect(self.calculate_click)
        
        #self.source_sz = QLineEdit(str(defsource_sz))
        #centralLayout.addWidget(QLabel("Source size (um):"), 3, 0)
        #centralLayout.addWidget(self.source_sz, 3, 1)
        #self.source_sz.editingFinished.connect(self.calculate_click)
        
        self.beam_div = QLineEdit(str(defbeam_div))
        centralLayout.addWidget(QLabel("Beam divergence (urad):"), 4, 0)
        centralLayout.addWidget(self.beam_div, 4, 1)
        self.beam_div.editingFinished.connect(self.calculate_click)
        
        centralLayout.addWidget(QLabel("CRL1 lens arms:"), 5, 0)
        centralLayout.addWidget(self.buildcrl1group(), 5, 1)
        
        centralLayout.addWidget(QLabel("CRL2 lens arms:"), 6, 0)
        centralLayout.addWidget(self.buildcrl2group(), 6, 1)
        
        centralLayout.addWidget(QLabel("CRL3 lens arms:"), 7, 0)
        centralLayout.addWidget(self.buildcrl3group(), 7, 1)
        
        self.crl3_posz_shft = QLineEdit(str(defcrl3_posz_shft))
        centralLayout.addWidget(QLabel("CRL3 shift (mm):"), 8, 0)
        centralLayout.addWidget(self.crl3_posz_shft, 8, 1)
        self.crl3_posz_shft.editingFinished.connect(self.calculate_click)
        
        self.des_posz = QLineEdit(str(tcc_posz))
        centralLayout.addWidget(QLabel("Chosen pos. (m):"), 9, 0)
        centralLayout.addWidget(self.des_posz, 9, 1)
        self.des_posz.editingFinished.connect(self.calculate_click)
        
        #self.pb = QPushButton('Calculate')
        #centralLayout.addWidget(self.pb, 9, 0)
        #self.pb.clicked.connect(self.calculate_click)
        centralLayout.addWidget(QLabel("Beam divergence and source size:"), 10, 0)
        
        self.fel_beamsz = QLineEdit(str(deffel_beamsz))
        centralLayout.addWidget(QLabel("FEL beam size NO lens (um):"), 11, 0)
        centralLayout.addWidget(self.fel_beamsz, 11, 1)
        self.fel_beamsz.editingFinished.connect(self.calculatebd_click)
        
        #self.fel_beamsz_wl = QLineEdit(str(deffel_beamsz_wl))
        #centralLayout.addWidget(QLabel("FEL beam size with lens (um):"), 11, 0)
        #centralLayout.addWidget(self.fel_beamsz_wl, 11, 1)
        #self.fel_beamsz_wl.editingFinished.connect(self.calculatebd_click)
        
        self.hed_beamsz_wl = QLineEdit(str(defhed_beamsz_wl))
        centralLayout.addWidget(QLabel("HED beam size with lens (um):"), 12, 0)
        centralLayout.addWidget(self.hed_beamsz_wl, 12, 1)
        self.hed_beamsz_wl.editingFinished.connect(self.calculatebd_click)
        
        #self.pb2 = QPushButton('Calculate beam div.')
        #centralLayout.addWidget(self.pb2, 13, 0)
        #self.pb2.clicked.connect(self.calculatebd_click)
        
        self.msg1 = QLabel("")
        centralLayout.addWidget(QLabel("Source position (m):"), 14, 0)
        centralLayout.addWidget(self.msg1, 14, 1)
        
        self.msg2 = QLabel("")
        centralLayout.addWidget(QLabel("Beam divergence (urad):"), 15, 0)
        centralLayout.addWidget(self.msg2, 15, 1)
        
        self.warn1 = QLabel("")
        centralLayout.addWidget(QLabel("Output and warnings:"), 16, 0)
        centralLayout.addWidget(self.warn1, 17, 1)
        
        self.xdata = np.arange(10)
        self.ydata = np.arange(10)
        plotlayout = QVBoxLayout()
        self.plot1 = Canvas(self, width=20, height=2, dpi=100)
        plotlayout.addWidget(self.plot1)
        self.plot2 = Canvas(self, width=20, height=20, dpi=100)
        plotlayout.addWidget(self.plot2)
        self.update_plot()
        
        grid_layout = QGridLayout(self)
        grid_layout.addLayout(topLayout, 20, 0, 10, 10)
        grid_layout.addLayout(centralLayout, 1, 0, 10, 10)
        grid_layout.addLayout(plotlayout, 1, 10, 30, 30)
        
        self.setLayout(grid_layout)
        self.calculate_click()
        self.show()    
        
    def buildcrl1group(self):
        groupBox = QGroupBox()
        self.c12 = QCheckBox("2",self)
        self.c13 = QCheckBox("3",self)
        self.c14 = QCheckBox("4",self)
        self.c15 = QCheckBox("5",self)
        self.c16 = QCheckBox("6",self)
        self.c17 = QCheckBox("7",self)
        self.c18 = QCheckBox("8",self)
        self.c19 = QCheckBox("9",self)
        hbox = QHBoxLayout()
        hbox.addWidget(self.c12)
        hbox.addWidget(self.c13)
        hbox.addWidget(self.c14)
        hbox.addWidget(self.c15)
        hbox.addWidget(self.c16)
        hbox.addWidget(self.c17)
        hbox.addWidget(self.c18)
        hbox.addWidget(self.c19)
        groupBox.setLayout(hbox)
        self.c12.stateChanged.connect(self.calculate_click)
        self.c13.stateChanged.connect(self.calculate_click)
        self.c14.stateChanged.connect(self.calculate_click)
        self.c15.stateChanged.connect(self.calculate_click)
        self.c16.stateChanged.connect(self.calculate_click)
        self.c17.stateChanged.connect(self.calculate_click)
        self.c18.stateChanged.connect(self.calculate_click)
        self.c19.stateChanged.connect(self.calculate_click)
        return groupBox

    def buildcrl2group(self):
        groupBox = QGroupBox()
        self.c21 = QCheckBox("1",self)
        self.c22 = QCheckBox("2",self)
        self.c23 = QCheckBox("3",self)
        self.c24 = QCheckBox("4",self)
        self.c25 = QCheckBox("5",self)
        self.c26 = QCheckBox("6",self)
        self.c27 = QCheckBox("7",self)
        self.c28 = QCheckBox("8",self)
        self.c29 = QCheckBox("9",self)
        self.c20 = QCheckBox("10",self)
        hbox = QHBoxLayout()
        hbox.addWidget(self.c21)
        hbox.addWidget(self.c22)
        hbox.addWidget(self.c23)
        hbox.addWidget(self.c24)
        hbox.addWidget(self.c25)
        hbox.addWidget(self.c26)
        hbox.addWidget(self.c27)
        hbox.addWidget(self.c28)
        hbox.addWidget(self.c29)
        hbox.addWidget(self.c20)
        self.c21.stateChanged.connect(self.calculate_click)
        self.c22.stateChanged.connect(self.calculate_click)
        self.c23.stateChanged.connect(self.calculate_click)
        self.c24.stateChanged.connect(self.calculate_click)
        self.c25.stateChanged.connect(self.calculate_click)
        self.c26.stateChanged.connect(self.calculate_click)
        self.c27.stateChanged.connect(self.calculate_click)
        self.c28.stateChanged.connect(self.calculate_click)
        self.c29.stateChanged.connect(self.calculate_click)
        self.c20.stateChanged.connect(self.calculate_click)
        groupBox.setLayout(hbox)
        return groupBox
    
    def buildcrl3group(self):
        groupBox = QGroupBox()
        
        self.c31 = QCheckBox("1",self)
        self.c32 = QCheckBox("2",self)
        self.c33 = QCheckBox("3",self)
        self.c34 = QCheckBox("4",self)
        self.c35 = QCheckBox("5",self)
        self.c36 = QCheckBox("6",self)
        self.c37 = QCheckBox("7",self)
        self.c38 = QCheckBox("8",self)
        self.c39 = QCheckBox("9",self)
        self.c30 = QCheckBox("10",self)
        hbox = QHBoxLayout()
        hbox.addWidget(self.c31)
        hbox.addWidget(self.c32)
        hbox.addWidget(self.c33)
        hbox.addWidget(self.c34)
        hbox.addWidget(self.c35)
        hbox.addWidget(self.c36)
        hbox.addWidget(self.c37)
        hbox.addWidget(self.c38)
        hbox.addWidget(self.c39)
        hbox.addWidget(self.c30)
        self.c31.stateChanged.connect(self.calculate_click)
        self.c32.stateChanged.connect(self.calculate_click)
        self.c33.stateChanged.connect(self.calculate_click)
        self.c34.stateChanged.connect(self.calculate_click)
        self.c35.stateChanged.connect(self.calculate_click)
        self.c36.stateChanged.connect(self.calculate_click)
        self.c37.stateChanged.connect(self.calculate_click)
        self.c38.stateChanged.connect(self.calculate_click)
        self.c39.stateChanged.connect(self.calculate_click)
        self.c30.stateChanged.connect(self.calculate_click)
        groupBox.setLayout(hbox)
        return groupBox
    
    def checkcrl1(self):
        crllens = np.array([\
        2*int(self.c12.checkState()/2),\
        3*int(self.c13.checkState()/2),\
        4*int(self.c14.checkState()/2),\
        5*int(self.c15.checkState()/2),\
        6*int(self.c16.checkState()/2),\
        7*int(self.c17.checkState()/2),\
        8*int(self.c18.checkState()/2),\
        9*int(self.c19.checkState()/2),\
                            ])
        crl1lens = crllens[crllens>0]
        return crl1lens
    
    def checkcrl2(self):
        crllens = np.array([\
        1*int(self.c21.checkState()/2),\
        2*int(self.c22.checkState()/2),\
        3*int(self.c23.checkState()/2),\
        4*int(self.c24.checkState()/2),\
        5*int(self.c25.checkState()/2),\
        6*int(self.c26.checkState()/2),\
        7*int(self.c27.checkState()/2),\
        8*int(self.c28.checkState()/2),\
        9*int(self.c29.checkState()/2),\
        10*int(self.c20.checkState()/2),\
                            ])
        crl2lens = crllens[crllens>0]
        return crl2lens
    
    def checkcrl3(self):
        crllens = np.array([\
        1*int(self.c31.checkState()/2),\
        2*int(self.c32.checkState()/2),\
        3*int(self.c33.checkState()/2),\
        4*int(self.c34.checkState()/2),\
        5*int(self.c35.checkState()/2),\
        6*int(self.c36.checkState()/2),\
        7*int(self.c37.checkState()/2),\
        8*int(self.c38.checkState()/2),\
        9*int(self.c39.checkState()/2),\
        10*int(self.c30.checkState()/2),\
                            ])
        crl3lens = crllens[crllens>0]
        return crl3lens
    
    def update_plot(self):
        self.plot1.axes.cla()  # Clear the canvas.
        self.plot1.axes.plot(self.xdata, self.ydata, 'black')
        self.plot1.axes.set_xlabel("Distance from nominal source (m)")
        self.plot1.axes.set_ylabel("Beam size y (um)")
        self.plot1.axes.set_title("XTD1/XTD6 Tunnel")
        self.plot1.axes.set_xlim(-50, 1000)
        ymin, ymax = 0, max(self.ydata)*1.2
        self.plot1.axes.set_ylim(ymin, ymax)
        self.plot1.axes.yaxis.set_ticks_position('both')
        self.plot1.axes.vlines(crl1_posz, ymin, ymax, color="grey", linestyle="--", label="CRL1 "+str(crl1_posz))
        self.plot1.axes.text(crl1_posz, 0.01*ymax, "CRL1", color="grey") 
        self.plot1.axes.vlines(crl2_posz, ymin, ymax, color="grey", linestyle="--", label="CRL2 "+str(crl2_posz))
        self.plot1.axes.text(crl2_posz, 0.01*ymax, "CRL2", color="grey") 
        self.plot1.axes.vlines(crl3_posz, ymin, ymax, color="grey", linestyle="--", label="CRL3 "+str(crl3_posz))
        self.plot1.axes.text(crl3_posz, 0.01*ymax, "CRL3", color="grey") 
        self.plot1.axes.vlines(tcc_posz, ymin, ymax, color="red", linestyle="--", label="TCC "+str(tcc_posz))
        self.plot1.axes.text(tcc_posz, 0.2*ymax, "TCC", color="red")
        self.plot1.axes.vlines([m1_posz, m2_posz, m3_posz], ymin, ymax, color="pink", linestyle="-.", label="Mirrors")
        #self.plot1.axes.text(m1_posz, 0.01*ymax, "M1", color="pink") 
        self.plot1.axes.text(m2_posz, 0.1*ymax, "Mirrors", color="pink") 
        #self.plot1.axes.text(m3_posz, 0.01*ymax, "M3", color="pink") 
        self.plot1.axes.vlines([mono_posz, hrmono_posz], ymin, ymax, color="lightgreen", linestyle="-.", label="Monos")
        self.plot1.axes.text(hrmono_posz, 0.1*ymax, "Monos", color="lightgreen") 
        #self.plot1.axes.vlines([xtds_posz, opts_posz, ibss_posz], 0, max(self.ydata)*1.2, color="purple", linestyle="-.", label="Shutters/IBS")
        #self.plot1.axes.legend()
        # Trigger the canvas to update and redraw.
        self.plot1.draw()
        self.plot2.axes.cla()  # Clear the canvas.
        self.plot2.axes.plot(self.xdata, self.ydata, 'black')
        self.plot2.axes.set_xlabel("Distance from nominal source (m)")
        self.plot2.axes.set_ylabel("Beam size y (um)")
        self.plot2.axes.set_title("OPT/EXP Hutch")
        self.plot2.axes.set_xlim(935, 985)
        if self.ydata[self.xdata>xtds_posz].size != 0:
            ymin, ymax = 0, max(self.ydata[self.xdata>xtds_posz])*1.2
        self.plot2.axes.set_ylim(ymin, ymax)
        self.plot2.axes.yaxis.set_ticks_position('both')
        self.plot2.axes.vlines(crl3_posz, ymin, ymax, color="grey", linestyle="--", label="CRL3 pos "+str(crl3_posz))
        self.plot2.axes.text(crl3_posz, 0.8*ymax, "CRL3", color="grey") 
        self.plot2.axes.vlines([xtds_posz, opts_posz, ibss_posz], ymin, ymax, color="purple", linestyle="-.", label="Shutters/IBS")
        self.plot2.axes.text(xtds_posz, 0.9*ymax, "XTD", color="purple") 
        self.plot2.axes.text(opts_posz, 0.9*ymax, "OPT", color="purple") 
        self.plot2.axes.text(ibss_posz, 0.9*ymax, "IBS", color="purple") 
        self.plot2.axes.vlines(tcc_posz, ymin, ymax, color="red", linestyle="--", label="TCC "+str(tcc_posz))
        self.plot2.axes.text(tcc_posz, 0.8*ymax, "TCC1", color="red")
        #self.plot2.axes.legend()
        # Trigger the canvas to update and redraw.
        self.plot2.draw()
        
    def calculate_click(self): 
        try:
            energy = float(self.energy.text())
        except:
            energy = defenergy
        try:
            bndwdth = float(self.bndwdth.text())
        except:
            bndwdth = defbndwdth
        try:
            source_pos = float(self.source_pos.text())
        except:
            source_pos = defsource_pos
        try:
            source_sz = float(self.source_sz.text())
        except:
            source_sz = defsource_sz
        try:
            beam_div = float(self.beam_div.text())*1e-6
        except:
            beam_div = defbeam_div
        try:
            crl3_posz_shft = float(self.crl3_posz_shft.text())*1e-3
        except:
            crl3_posz_shft = defcrl3_posz_shft
        try:
            des_posz = float(self.des_posz.text())
        except:
            des_posz = tcc_posz
        crl1lens = self.checkcrl1()
        crl2lens = self.checkcrl2()
        crl3lens = self.checkcrl3()
        crl3_pts, warning = calculate(energy, bndwdth, crl1lens, crl2lens, crl3lens, \
                                      source_pos, source_sz, beam_div, crl3_posz_shft, des_posz)
        self.warn1.setText(str(warning))
        self.xdata = crl3_pts[0]
        self.ydata = abs(crl3_pts[1])
        self.update_plot()

    def calculatebd_click(self):
        energy = float(self.energy.text())
        crl1lens = self.checkcrl1()
        fel_beamsz = float(self.fel_beamsz.text())
        hed_beamsz_wl = float(self.hed_beamsz_wl.text())
        newsourcepos, newbeamdiv, sperr, bderr = calc_div(energy, crl1lens, fel_beamsz, hed_beamsz_wl)
        self.msg1.setText(str(newsourcepos)+" +/- "+str(sperr))
        self.msg2.setText(str(newbeamdiv)+" +/- "+str(bderr))
        
"""Plotting"""
class Canvas(FigureCanvas):
    def __init__(self, parent = None, width = 5, height = 5, dpi = 100):
        fig = Figure()#figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.set_xlabel("Distance from nominal source (m)")
        super(Canvas, self).__init__(fig)
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MainWindow()
    sys.exit(app.exec_())
