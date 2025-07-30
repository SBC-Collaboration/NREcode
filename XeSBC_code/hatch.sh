nameJob="emcee_test"

SUBDIR="/home/runze/Documents/results/Epoch_storage"

if [ ! -d $SUBDIR ]; then
    mkdir $SUBDIR
fi

#cp job_fitnest.sh ${SUBDIR}/job_fitnest.sh
#cp MCMC.py ${SUBDIR}/MCMC.py
#cp MC_argon_full_20250701_LSS07_2E5_Noahformat.txt ${SUBDIR}/MC_argon_full_20250701_LSS07_2E5_Noahformat.txt
#cp MC_argon_full_20250701_LSS07_2E5 ${SUBDIR}/MC_argon_full_20250701_LSS07_2E5
#cp *.txt ${SUBDIR}/

#cd $SUBDIR

#sbatch job_fitnest.sh $1 $2
sbatch job_fitnest.sh

sleep 1

echo submitted
