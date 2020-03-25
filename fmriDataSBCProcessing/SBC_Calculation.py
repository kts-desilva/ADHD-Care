from nilearn import datasets
from nilearn import input_data
import numpy as np
from nilearn import plotting
import os

adhd_dataset = datasets.fetch_adhd(n_subjects=40)
fsaverage = datasets.fetch_surf_fsaverage()

for i in range(0,40):
    func_filename = adhd_dataset.func[i]
    confound_filename = adhd_dataset.confounds[i]

    sub_id=func_filename.split('/')[-1]
    sub_id=sub_id.split('_')[0]

    print('***',sub_id)
    print(func_filename)
    print(confound_filename)

    pcc_coords = [ (47, -57, 20)]

    seed_masker = input_data.NiftiSpheresMasker(
        pcc_coords, radius=8,
        detrend=True, standardize=True,
        low_pass=0.1, high_pass=0.01, t_r=2.,
        memory='nilearn_cache', memory_level=1, verbose=0)


    seed_time_series = seed_masker.fit_transform(func_filename,
                                                 confounds=[confound_filename])

    brain_masker = input_data.NiftiMasker(
        smoothing_fwhm=6,
        detrend=True, standardize=True,
        low_pass=0.1, high_pass=0.01, t_r=2.,
        memory='nilearn_cache', memory_level=1, verbose=0)

    brain_time_series = brain_masker.fit_transform(func_filename,
                                                   confounds=[confound_filename])

    print("seed time series shape: (%s, %s)" % seed_time_series.shape)
    print("brain time series shape: (%s, %s)" % brain_time_series.shape)

    seed_based_correlations = np.dot(brain_time_series.T, seed_time_series) / \
                              seed_time_series.shape[0]

    print("seed-based correlation shape: (%s, %s)" % seed_based_correlations.shape)
    print("seed-based correlation: min = %.3f; max = %.3f" % (
        seed_based_correlations.min(), seed_based_correlations.max()))

    seed_based_correlations_fisher_z = np.arctanh(seed_based_correlations)
    print("seed-based correlation Fisher-z transformed: min = %.3f; max = %.3f" % (
        seed_based_correlations_fisher_z.min(),
        seed_based_correlations_fisher_z.max()))

    # Finally, we can tranform the correlation array back to a Nifti image
    # object, that we can save.
    seed_based_correlation_img = brain_masker.inverse_transform(
        seed_based_correlations.T)
    seed_based_correlation_img.to_filename('seed_correlation_IPC_R/sbc_{nid}_z.nii.gz'.format(nid=sub_id))

    display = plotting.plot_stat_map(seed_based_correlation_img, threshold=0.3,
                                         cut_coords=pcc_coords[0])

    display.add_markers(marker_coords=pcc_coords, marker_color='g',marker_size=300)
    display.annotate(size=20)
    # display.colorbar(size=15)

    #display.savefig('correlation_images/3d/pcc_seed_correlation_%d.jpeg'%(i))
    if i<2:
        display.savefig('sbc_z_1_20_%d.png'%(i))


