from nipype.interfaces import fsl
from nipype.interfaces.fsl.model import Cluster
from nipype.interfaces.fsl import maths
from os.path import join as opj
import os
import pandas as pd
import shutil

def main(t_val, con, sign, base_dir, source):

    if source == "neurosynth":
        if not os.path.exists(opj(base_dir,'neurosynth')):
            os.makedirs(opj(base_dir,'neurosynth'))

        fsl.FSLCommand.set_default_output_type('NIFTI')
        cl = Cluster()
        cl.terminal_output = "file_stdout"
        cl.inputs.threshold = t_val
        cl.inputs.in_file = opj(base_dir,'multisensory_association-test_z_FDR_0.01.nii')
        cl.inputs.out_localmax_txt_file = opj(base_dir,'neurosynth/localmax_stats_neurosynth.txt')
        cl.inputs.out_index_file = opj(base_dir,'neurosynth/cluster_index')
        cl.inputs.use_mm = True

        #if not os.path.isfile(cl.inputs.out_localmax_txt_file):
        res = cl.run()
        res.runtime.stdout
        text_file = pd.read_csv('stdout.nipype', sep="\t")
        text_file.to_csv("stdout.csv", sep="\t")
        shutil.copy("stdout.csv", opj(base_dir,'neurosynth/stats_neurosynth.csv'))

        ext_b = maths.Threshold()
        ext_a = maths.Threshold()

        ext_b.inputs.in_file = cl.inputs.out_index_file+'.nii'
        ext_b.inputs.internal_datatype = "int"
        ext_a.inputs.internal_datatype = "int"
        ext_a.args = 'bin'
        for idx,cluster in enumerate(range(max(text_file["Cluster Index"]))):
            if int(text_file["Voxels"][idx]) >= 50:
                print(text_file["Cluster Index"][cluster])
                ext_b.inputs.thresh = text_file["Cluster Index"][cluster]
                ext_b.inputs.out_file = opj(base_dir,'neurosynth/cluster_'+str(cluster)+'.nii')
                ext_b.run()

                ext_a.inputs.in_file = ext_b.inputs.out_file
                ext_a.inputs.thresh = text_file["Cluster Index"][cluster]
                ext_a.inputs.direction = "above"
                ext_a.inputs.out_file = ext_b.inputs.out_file
                ext_a.run()

    else:

        if not os.path.exists(opj(base_dir,'ROI')):
            os.makedirs(opj(base_dir,'ROI'))

        fsl.FSLCommand.set_default_output_type('NIFTI')
        cl = Cluster()
        cl.terminal_output = "file_stdout"
        cl.inputs.threshold = t_val
        if sign == "pos":
            cl.inputs.in_file = opj(base_dir,'2ndLevel/SnPM_SecondLevel_con')+ str(con) + '/snpmT+.img' #'/SnPM_filtered_t'+str(t_val).replace('.', '_')+'_'+sign+'.nii'
        else:
            cl.inputs.in_file = opj(base_dir,'2ndLevel/SnPM_SecondLevel_con')+ str(con) + '/snpmT-.img' 
        cl.inputs.out_localmax_txt_file = opj(base_dir,'ROI/localmax_stats_con'+str(con)+'_'+str(t_val).replace('.', '_')+'_'+sign+'.txt')
        cl.inputs.out_index_file = opj(base_dir,'ROI/cluster_index_con'+str(con)+'_'+str(t_val).replace('.', '_')+'_'+sign)
        cl.inputs.use_mm = True

        #if not os.path.isfile(cl.inputs.out_localmax_txt_file):
        res = cl.run()
        res.runtime.stdout
        text_file = pd.read_csv('stdout.nipype', sep="\t")
        text_file.to_csv("stdout.csv", sep="\t")
        shutil.copy("stdout.csv", opj(base_dir,'ROI/stats_con'+str(con)+'_'+str(t_val).replace('.', '_')+'_'+sign+'.csv'))
    
            

        ext_b = maths.Threshold()
        ext_a = maths.Threshold()

        ext_b.inputs.in_file = cl.inputs.out_index_file+'.nii'
        ext_b.inputs.internal_datatype = "int"
        ext_a.inputs.internal_datatype = "int"
        ext_a.args = 'bin'
        for idx,cluster in enumerate(range(max(text_file["Cluster Index"]))):
            if int(text_file["Voxels"][idx]) >= 50:
                print(text_file["Cluster Index"][cluster])
                ext_b.inputs.thresh = text_file["Cluster Index"][cluster]
                ext_b.inputs.out_file = opj(base_dir,'ROI/cluster_'+str(cluster)+"_con"+str(con)+'_'+str(t_val).replace('.', '_')+'_'+sign+'.nii')
                ext_b.run()

                ext_a.inputs.in_file = ext_b.inputs.out_file
                ext_a.inputs.thresh = text_file["Cluster Index"][cluster]
                ext_a.inputs.direction = "above"
                ext_a.inputs.out_file = ext_b.inputs.out_file
                ext_a.run()




if __name__ == "__main__":
    main(3.1,1, 'pos', '/mnt/d/multlearn-sns/SPM/nipype/nipype/model2', source="fmri")

