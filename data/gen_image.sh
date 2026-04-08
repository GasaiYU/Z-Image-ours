export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HOME=/mmu-vcg/gaomingju/data/T2I/hf_cache/


python data/generate_triplet_images.py --n_gpus 8 --batch_size 1
