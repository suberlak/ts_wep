# Wavefront Estimation Pipeline (WEP)

*This module is used to calculate the wavefront error in annular Zernike polynomials up to 22 terms based on the intra- and extra-focal images in the large synoptic survey telescope (LSST). The main idea is to use the transport of intensity (TIE) and assume the change of intensity only comes from the wavefront error.*

## 1. Version History

*Version 1.0*
<br/>
*Finish the WEP in totally ideal condition.*

*Author: Te-Wei Tsai*
*Date: 10-20-2017*

## 2. Platform

- *python: 3.6.2*
- *scientific pipeline: v14*

## 3. Needed Package

- *lsst_sims*
- *numpy*
- *scipy*
- *astropy*
- *matplotlib*
- *cython*
- *scikit-image*

## 4. Use of Module

*1. Setup the DM environment:*
<br/>
source $path_of_lsst_scientific_pipeline/loadLSST.bash
<br/>
setup sims_catUtils -t sims

*2. Setup the WEP environment:*
<br/>
export PYTHONPATH=$PYTHONPATH:$path_to_ts_lsst_bsc
<br/>
export PYTHONPATH=$PYTHONPATH:$path_to_ts_lsst_wep
<br/>
export PYTHONPATH=$PYTHONPATH:$path_to_ts_lsst_deblend
<br/>
export PYTHONPATH=$PYTHONPATH:$path_to_ts_lsst_isr
<br/>
export PYTHONPATH=$PYTHONPATH:$path_to_ts_tcs_wep

*3. Connect to fatboy server:*
<br/>
ssh -i $Position_of_SSH_key -L 51433:fatboy.phys.washington.edu:1433 simsuser@gateway.astro.washington.edu
<br/>
Keep this terminal open for the connection.

## 5. Content

*This module contains the following classes:*

- *WFDataCollector: Accommodate the PhoSim simulated image contains the sky and calibration products (bias, dark current, and flat dome light) into the data butler format. The registry repository will be updated if it is necessary.*
- *IsrWrapper: Do the ISR by using DM ISR library directly. The calibration products of bias, dark current, and flat dome light are used.*
- *EimgIsrWrapper: Simulate the post-ISR image by using the electronic image generated by PhoSim. This interface keeps the same as IsrWrapper.*
- *SourceSelector: Query the bright star catalog (BSC) in University of Washington (UW) to select the available target to do the WEP. If the star data is in the local database already, the query is also available.*
- *SourceProcessor: Process the post-ISR images to get the clean star images with measured optical coordinate (field x, y). The deblending algorithm is used to get the single target star image if the neighboring stars exist.*
- *WFEstimator: Calculate the wavefront error in annular Zernike polynomials up to 22 terms based on the defocal star donut images.*
- *Middleware: Communicate with subsystems by software abstraction layer (SAL).*

## 6. Target for Future Release

- *Integration of WEP and PhoSim is not done yet. There might be some inconsistency of coordinate among PhoSim, camera control system (CCS), and DM.*
- *TIE is used as the main algorithm, which is based on the single source. However, for the LSST normal case, this is not true. The initial idea here is to normalize the intensities of multiple sources.*
- *No boundary consideration of TIE studied.*
- *The use of instrument signature removal (ISR) in WEP traces to data management (DM) ISR library, which needs to customize the details/ strategies in the future release.*
- *The deblending algorithm assumes the neighboring stars have the same optical condition as the bright star. This algorithm can only handle one neighboring star that has certain magnitude and distance compared with the bright star.*
- *The algorithm to calculate the centroid of star needs a clean background.*
- *No system error is considered.*
- *No image quality determination included.*
- *No robust signal-to-noise ratio (SNR) calculation included.*
- *No master donut images by migration included.*
- *No vignette correction included.*
- *World coordinate system (WCS) is based on the focal plane with the parallax model. However, the defocal images are used in TIE. The difference and compensation between real and calculated pixel positions are not considered yet.*
- *The reliability of BSC in University of Washington (UW) is not verified.*
- *The local BSC database is not constructed. Need to use the Scheduler to give a reasonable survey route to minimize the calculation time. Another choice is to use SkyCoord() in Astropy. The ref is at: "http://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html".*
- *The mechanism to update the BSC is not included.*
- *No statistics/ strategy of selecting wavefront sensors on full-focal plane of LSST camera included.*
- *The calculation time is much longer than the spec (14 sec).*
- *The pipeline framework (e.g. luigi) and parallel calculation are not included.*
- *The data collection from data acquisition (DAQ) is not included.*
- *The commissioning camera (ComCam) mapper is mocked based on the central raft of LSST mapper.*
