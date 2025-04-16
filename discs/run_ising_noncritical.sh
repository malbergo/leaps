for sampler in "randomwalk" "dlmc" "dmala" "hammingball" "blockgibbs" "gibbs" "path_auxiliary" "locallybalanced" "gwg"
do
    model="isingnoncritical" sampler="${sampler}" ./discs/experiment/run_sampling_local.sh
done