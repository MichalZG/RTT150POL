import astropy.io.fits as fits
import glob
import numpy as np
from photutils.morphology import data_properties
from photutils import EllipticalAperture
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
# from photutils.morphology import centroid_2dg
from photutils import EllipticalAnnulus
from astropy.table import hstack, vstack
from photutils import aperture_photometry
from astropy.table import Table, Column
from astropy.stats import sigma_clipped_stats

med_bias_data = fits.getdata('/home/pi/Kolchoz/'
                             'RTT150/20160806RTT150/20160805/bdf/med_bias.fits')
plots = True

r_mask = 7
r_multi_ap = 3.0
r_ann_in = 1
r_ann_out = 3

gain_key = 'GAIN'
jd_key = 'JD_TCS'
exp_key = 'EXPTIME'
obj_key = 'OBJECT'
afilter_key = 'AFILTERW'
bfilter_key = 'BFILTERW'

# saxj2103
stars_pos = [[506.0, 337.4], [507.4, 247.7], [507.9, 149.3], [507.0, 61.4]]
# flux to output as COUNTS
# residual_aperture_sum or aperture_sum_raw or aperture_sum_bkg
output_flux = 'residual_aperture_sum'
outpur_flux_err = 'residual_aperture_err_sum'
# AFILTER or BFILTER
output_filter = 'AFILTER'
images = sorted(glob.glob('*'))


def createMask(pos):

    y, x = np.ogrid[-pos[1]:data_shape[0]-pos[1],
                    -pos[0]:data_shape[1]-pos[0]]
    mask = x*x + y*y <= r_mask*r_mask
    mask_arr = np.full(data_shape, True, dtype=bool)
    mask_arr[mask] = False

    return mask_arr


def makeApertures(data):

    apertures = []
    # data -= mean * 0.8
    # data[np.where(data <= 0)] = 1

    for mask in reversed(map(createMask, stars_pos)):
        mean, median, std = sigma_clipped_stats(data, mask=mask,
                                                sigma=3.0, iters=5)
        print mean, median, std, im
        props = data_properties(data-np.uint64(median), mask=mask)
        # position = centroid_2dg(data, mask=mask)
        position = (props.xcentroid.value, props.ycentroid.value)
        a = props.semimajor_axis_sigma.value * r_multi_ap
        b = props.semiminor_axis_sigma.value * r_multi_ap
        theta = props.orientation.value
        aperture = EllipticalAperture(position, a, b, theta=theta)
        annulus = EllipticalAnnulus(position, a+r_ann_in, a+r_ann_out,
                                    b+r_ann_out, theta=theta)
        apertures.append([aperture, annulus])

    return apertures


def makePhot(data, hdr, plot=True):

    apertures = makeApertures(data)
    out_table = []

    for aperture in apertures:
        rawflux_table = aperture_photometry(data, aperture[0],
                                            error=np.sqrt(data))
        bkgflux_table = aperture_photometry(data, aperture[1],
                                            error=np.sqrt(data))
        phot_table = hstack([rawflux_table, bkgflux_table],
                            table_names=['raw', 'bkg'])
        bkg_mean = phot_table['aperture_sum_bkg'] / aperture[1].area()
        bkg_sum = bkg_mean * aperture[0].area()
        final_sum = phot_table['aperture_sum_raw'] - bkg_sum
        phot_table['residual_aperture_sum'] = final_sum

        err_column = np.sqrt(np.power(phot_table['aperture_sum_err_raw'], 2) +
                             np.power(phot_table['aperture_sum_err_bkg'], 2))
        phot_table.add_column(
            Column(name='residual_aperture_err_sum',
                   data=err_column))
        phot_table['xcenter_raw'].shape = 1
        phot_table['ycenter_raw'].shape = 1
        phot_table['xcenter_bkg'].shape = 1
        phot_table['ycenter_bkg'].shape = 1
        out_table.append(phot_table)

    out_table = vstack(out_table)

    if plot:
        makePlot(data, apertures)

    return out_table


def calcPhotErr():
    try:
        effective_gain = float(hdr[gain_key])
    except KeyError:
        effective_gain = 1.0
    return effective_gain
    # sky_level, sky_sigma = background(data)
    # error = calc_total_error(data, sky_sigma, effective_gain)


def makePlot(data, apertures):
    plt.imshow(data, cmap='Greys', origin='lower',
               norm=LogNorm())
    for aperture in apertures:
        aperture[0].plot(linewidth=0.3, color='#d62728')
        aperture[1].plot(fill=False, linewidth=0.3, color='k')
    plt.savefig(im+'.png', dpi=300)
    plt.clf()


def saveTable(out_table):

    out_table = vstack(out_table)

    out_table.add_column(
        Column(name='COUNTS', data=out_table[output_flux]), index=0)
    out_table.add_column(
        Column(name='COUNTS_ERR', data=out_table[output_flux]), index=1)

    out_table.rename_column(output_filter, 'FILTER')

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

    out_table.write('sax.txt', format='ascii', delimiter=',')

    createFitsTable(out_table)


def createFitsTable(tab):

    columns_to_remove = ['residual_aperture_sum', 'residual_aperture_err_sum',
                         'aperture_sum_raw', 'aperture_sum_bkg',
                         'aperture_sum_err_raw', 'aperture_sum_err_bkg',
                         'xcenter_raw', 'ycenter_raw',
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

    tbhdu = fits.new_table(cols)
    prihdr = fits.Header()
    prihdu = fits.PrimaryHDU(header=prihdr)
    thdulist = fits.HDUList([prihdu, tbhdu])

    thdulist.writeto('sax.fits', clobber=True)


if __name__ == "__main__":

    out_table = []
    for im in images:
        hdr_table = Table(names=('TIME', 'OBJECT', 'EXPTIME',
                                 'AFILTER', 'BFILTER', 'ROTOR'),
                          dtype=('f8', 'S8', 'f8', 'S6', 'S8', 'i'))

        if im.endswith(('.py', '.txt', '.png')):
            continue
        hdu = fits.open(im, mode='update', ignore_missing_end=True)
        data_shape = hdu[0].data.shape

        try:
            hdu[0].header['BIASCORR']

        except KeyError:
            hdu[0].data = hdu[0].data - med_bias_data
            hdu[0].header['BIASCORR'] = 'True'
            print 'bias corr done'

        hdu.flush()

        data = np.copy(hdu[0].data)
        hdr = hdu[0].header

        for i in xrange(4):
            hdr_table.add_row([float(hdr[jd_key]),
                               hdr[obj_key],
                               float(hdr[exp_key]),
                               hdr[afilter_key].strip()[0],
                               hdr[bfilter_key].strip()[0],
                               i+1])

        phot_table = makePhot(data, hdr, plot=True)
        out_table.append(hstack([hdr_table, phot_table]))
    saveTable(out_table)
