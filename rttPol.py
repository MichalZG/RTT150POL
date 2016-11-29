#!/usr/bin/env python

import astropy.io.fits as fits
import glob
import numpy as np
from photutils import source_properties, properties_table
from photutils import CircularAperture, EllipticalAperture
from photutils import CircularAnnulus, EllipticalAnnulus
from photutils import detect_sources
from astropy.convolution import Gaussian2DKernel
import warnings
from astropy.table import hstack, vstack
from photutils import aperture_photometry
from astropy.table import Table, Column
from astropy.stats import sigma_clipped_stats
from astropy.time import Time as TimeObject 
import argparse
import ConfigParser
import os
import sys
import math
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
from astropy.stats import gaussian_fwhm_to_sigma
import random

warnings.filterwarnings("ignore", module="matplotlib")


class Config():

    def __init__(self):

        config = ConfigParser.RawConfigParser()
        config.read(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'config.cfg'))
        self.config = config

        # images
        self.images = config.get('images', 'images')

        # flux
        self.output_flux = config.get('flux', 'flux')
        self.output_flux_err = config.get('flux', 'flux_err')
        self.output_filter = config.get('flux', 'filter')

        # header
        self.gain_key = config.get('header', 'gain')
        self.jd_key = config.get('header', 'jd')
        self.date_key = config.get('header', 'date')
        self.time_key = config.get('header', 'time')
        self.exp_key = config.get('header', 'exp')
        self.obj_key = config.get('header', 'obj')
        self.afilter_key = config.get('header', 'afilter')
        self.bfilter_key = config.get('header', 'bfilter')

        # photometry
        self.calc_aperture = config.getboolean('photometry', 'calc_aperture')
        self.r_ap = config.getfloat('photometry', 'r_ap')
        self.r_mask = config.getfloat('photometry', 'r_mask')
        self.r_multi_ap = config.getfloat('photometry', 'r_multi_ap')
        self.r_ann_in = config.getfloat('photometry', 'r_ann_in')
        self.r_ann_out = config.getfloat('photometry', 'r_ann_out')
        self.calc_center = config.getboolean('photometry', 'calc_center')

        # regions
        self.stars_file = config.get('stars', 'stars_file')

        # default
        self.bias_name = config.get('default', 'bias_name')
        self.output_name = config.get('default', 'output_name')


def createMask(star, data_shape):
    y, x = np.ogrid[-star[1]-1:data_shape[0]-star[1]-1,
                    -star[0]-1:data_shape[1]-star[0]-1]
    mask = x * x + y * y <= cfg.r_mask * cfg.r_mask
    mask_arr = np.full(data_shape, True, dtype=bool)
    mask_arr[mask] = False

    return mask_arr


def makeProps(data, median, mask, star):

        # FIXME all variables must be in config
        sigma = 2.0 * gaussian_fwhm_to_sigma    # FWHM = 2.
        kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
        kernel.normalize()
        data[mask] = 0
        segm = detect_sources(data, median*3, npixels=5,
                              filter_kernel=kernel)
        props = properties_table(
            source_properties(data-np.uint64(median), segm),
            columns=['id', 'xcentroid', 'ycentroid', 'source_sum',
                     'semimajor_axis_sigma', 'semiminor_axis_sigma',
                     'orientation'])

        if len(props) > 1:
            props = findNearestObject(star, props, data.shape)
            return props
        else:
            return props[0]


def findNearestObject(star, props, data_shape):
    min_dist = max(data_shape)
    min_dist_id = 0
    x, y = star

    for prop in props:
        dist = math.sqrt((x - prop['xcentroid']) ** 2 +
                         (y - prop['ycentroid']) ** 2)

        if dist < min_dist:
            min_dist = dist
            min_dist_id = prop['id']

    return props[min_dist_id-1]


def makeApertures(data, stars):
    apertures = []
    masks = [createMask(star, data.shape) for star in stars]

    for i, mask in enumerate(masks):
        mean, median, std = sigma_clipped_stats(data, mask=mask,
                                                sigma=1.0, iters=5)

        if cfg.calc_center or cfg.calc_aperture:

            props = makeProps(np.copy(data), median, mask, stars[i])

        if cfg.calc_center:

            position = (props['xcentroid'],
                        props['ycentroid'])
        else:
            position = stars[i]

        if cfg.calc_aperture:
            a = props['semimajor_axis_sigma'] * cfg.r_multi_ap
            b = props['semiminor_axis_sigma'] * cfg.r_multi_ap
            theta = props['orientation']
            aperture = EllipticalAperture(position, a=a, b=b, theta=theta)
            annulus = EllipticalAnnulus(position,
                                        a_in=a+cfg.r_ann_in,
                                        a_out=a+cfg.r_ann_out,
                                        b_out=b+cfg.r_ann_out,
                                        theta=theta)
            print('i:{}, a:{}, b:{}, pos:{},{}').format(i+1, a, b, *position)
        else:
            aperture = CircularAperture(position, r=cfg.r_ap)
            annulus = CircularAnnulus(position,
                                      r_in=cfg.r_ap+cfg.r_ann_in,
                                      r_out=cfg.r_ap+cfg.r_ann_out)

        apertures.append([aperture, annulus, std])

    return apertures


def makePhot(data, hdr, stars, plot=False):

    apertures = makeApertures(data, stars)
    out_table = []

    for aperture in apertures:
        rawflux_table = aperture_photometry(data, aperture[0])
        bkgflux_table = aperture_photometry(data, aperture[1])
        phot_table = hstack([rawflux_table, bkgflux_table],
                            table_names=['raw', 'bkg'])
        bkg_mean = phot_table['aperture_sum_bkg'] / aperture[1].area()
        bkg_sum = bkg_mean * aperture[0].area()
        final_sum = phot_table['aperture_sum_raw'] - bkg_sum
        phot_table['residual_aperture_sum'] = final_sum

        phot_table.add_column(
            Column(name='residual_aperture_err_sum',
                   data=calcPhotErr(hdr, aperture,
                                    phot_table, bkgflux_table)))

        phot_table['xcenter_raw'].shape = 1
        phot_table['ycenter_raw'].shape = 1
        phot_table['xcenter_bkg'].shape = 1
        phot_table['ycenter_bkg'].shape = 1
        out_table.append(phot_table)

    out_table = vstack(out_table)

    if plot:
        makePlot(data, apertures, hdr['FILENAME'])

    return out_table


def calcPhotErr(hdr, aperture, phot_table, bkgflux_table):

    try:
        effective_gain = float(hdr[cfg.gain_key])
    except KeyError:
        effective_gain = 1.0

    err = math.sqrt(
        (phot_table['residual_aperture_sum'] / effective_gain) +
        (aperture[0].area() * aperture[2] ** 2) +
        ((aperture[0].area() ** 2 + aperture[2] ** 2) /
         (bkgflux_table['aperture_sum'] * aperture[0].area())))

    return [err]


def makePlot(data, apertures, im_name):

    plt.imshow(data, cmap='Greys', origin='lower',
               norm=LogNorm())
    for aperture in apertures:
        aperture[0].plot(linewidth=0.3, color='#d62728')
        aperture[1].plot(fill=False, linewidth=0.3, color='k')
        area = aperture[0]  # for plot mask area
        area.a, area.b = cfg.r_mask, cfg.r_mask
        area.plot(linewidth=0.2, color='k', ls=':')

    plt.savefig(im_name+'.png', dpi=300)
    plt.clf()


def saveTable(out_table, output_name):

    out_table = vstack(out_table)

    out_table.add_column(
        Column(name='COUNTS', data=out_table[cfg.output_flux]), index=0)
    out_table.add_column(
        Column(name='COUNTS_ERR', data=out_table['residual_aperture_err_sum']),
        index=1)

    out_table.rename_column(cfg.output_filter, 'FILTER')

    out_table.add_column(
        Column(name='PHASE', data=[0]*len(out_table)))
    out_table.add_column(
        Column(name='SEEING', data=[0]*len(out_table)))
    out_table.add_column(
        Column(name='AIRMASS', data=[0]*len(out_table)))
    out_table.add_column(
        Column(name='ROTANGLE', data=[0]*len(out_table)))
    out_table.add_column(
        Column(name='ROTSKYPA', data=[0]*len(out_table)))
    out_table.add_column(
        Column(name='MOON_FRAC', data=[0]*len(out_table)))
    out_table.add_column(
        Column(name='MOON_DIST', data=[0]*len(out_table)))

    out_table.write(output_name+'.csv', format='ascii', delimiter=',')

    saveToFitsTable(out_table, output_name)


def saveToFitsTable(tab, output_name):

    columns_to_remove = ['residual_aperture_sum', 'aperture_sum_raw',
                         'aperture_sum_bkg', 'xcenter_raw', 'ycenter_raw',
                         'xcenter_bkg', 'ycenter_bkg']

    tab.remove_columns(columns_to_remove)

    c1 = fits.Column(name='TIME', format='D', array=tab['TIME'])
    c2 = fits.Column(name='COUNTS', format='D', array=tab['COUNTS'])
    c3 = fits.Column(name='COUNTS_ERR', format='D', array=tab['COUNTS_ERR'])
    c4 = fits.Column(name='FILTER', format='1A', array=tab['FILTER'])
    c5 = fits.Column(name='ROTOR', format='1A', array=tab['ROTOR'])
    c6 = fits.Column(name='PHASE', format='D', array=tab['PHASE'])
    c7 = fits.Column(name='AIRMASS', format='D', array=tab['AIRMASS'])
    c8 = fits.Column(name='ROTANGLE', format='D', array=tab['ROTANGLE'])
    c9 = fits.Column(name='ROTSKYPA', format='D', array=tab['ROTSKYPA'])
    c10 = fits.Column(name='EXPTIME', format='D', array=tab['EXPTIME'])
    c11 = fits.Column(name='SEEING', format='D', array=tab['SEEING'])
    c12 = fits.Column(name='MOON_FRAC', format='D', array=tab['MOON_FRAC'])
    c13 = fits.Column(name='MOON_DIST', format='D', array=tab['MOON_DIST'])

    cols = fits.ColDefs([c1, c2, c3, c4, c5, c6, c7,
                         c8, c9, c10, c11, c12, c13])

    tbhdu = fits.BinTableHDU.from_columns(cols)
    prihdr = fits.Header()
    prihdu = fits.PrimaryHDU(header=prihdr)
    thdulist = fits.HDUList([prihdu, tbhdu])

    thdulist.writeto(output_name+'.fits', clobber=True)


def loadImages():

    images = sorted(glob.glob(
        os.path.join(os.path.curdir, cfg.images)))
    images = [name for name in images if len(name.split('.')) < 3]

    return images


def loadStars():

    try:
        stars = np.loadtxt(
            os.path.join(os.path.curdir, cfg.stars_file), dtype='f')
        stars = stars[np.argsort(stars[:, 1])]  # sort by y-axis
    except IOError:
        print('NO REGION FILE!')
        sys.exit()

    return stars


def loadBias(biasDir):

    bias_data = fits.getdata(biasDir)

    return bias_data


def createHdrTable(hdr):

    hdr_table = Table(names=('TIME', 'OBJECT', 'EXPTIME',
                             'AFILTER', 'BFILTER', 'ROTOR'),
                      dtype=('f8', 'S8', 'f8',
                             'S8', 'S8', 'i'))


    temp_time = TimeObject("T".join([hdr[cfg.date_key], hdr[cfg.time_key]]), format='isot', scale='utc')
    temp_jd = temp_time.jd

    try:
        print hdr["JD"], temp_jd, (float(hdr["JD"]) - float(temp_jd)) * 86400
    except:
        pass

    try:
        jd = float(hdr[cfg.jd_key])
    except ValueError:
        print('!!! JD problem !!!')
        print('{} - not found, try to use JD'.format(cfg.jd_key))
        jd = float(hdr['JD'])
    except KeyError:
        print('!!! JD problem !!!')
        print('!!! JD is set to random value!!!')
#        jd = 2450000. + random.random()
        jd = temp_jd

    for i in xrange(4):  # FIXME
        hdr_table.add_row([jd,
                          hdr[cfg.obj_key],
                          float(hdr[cfg.exp_key]),
                          hdr[cfg.afilter_key].strip()[-1],
                          hdr[cfg.bfilter_key].strip().split(' ')[-1],
                          i+1])
    return hdr_table


def main(args):

    out_table = []
    images = loadImages()
    stars = loadStars()

    if args.bias:
        bias_data = loadBias(args.bias)

    for im in images:
        print(im)
        hdu = fits.open(im, mode='update', ignore_missing_end=True)

        if args.bias:
            try:
                hdu[0].header['BIASCORR']
                print('%s - Bias correction already done' % im)

            except KeyError:
                hdu[0].data = hdu[0].data - bias_data
                hdu[0].header['BIASCORR'] = 'True'
                print('%s - Bias correction done' % im)

            hdu.flush()

        data = np.copy(hdu[0].data)
        hdr = hdu[0].header

        hdr_table = createHdrTable(hdr)
        phot_table = makePhot(data, hdr, stars, args.plot)
        out_table.append(hstack([hdr_table, phot_table]))

    saveTable(out_table, args.output)


if __name__ == "__main__":

    cfg = Config()
    parser = argparse.ArgumentParser(description='RTTPOL photometry')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='If set program will make plots')
    parser.add_argument('--bias', type=str, const=cfg.bias_name,
                        nargs='?', help='If set bias correction = ON, '
                                        'Default: %(const)s')
    parser.add_argument('--output', type=str, const=cfg.output_name,
                        default='output',
                        nargs='?', help='Names of the output tables'
                                        '(fits and csv) and plots, '
                                        'Default: %(const)s')
    args = parser.parse_args()

    main(args)
