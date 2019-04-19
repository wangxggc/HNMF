source config
source activate tensorflow

ROOT=`pwd`
time=`date +"%m%d%H%M"`
OUT_DIR=${ROOT}/output_${time}/
mkdir -p ${OUT_DIR}

CUDA_VISIBLE_DEVICES="0" python main.py \
    --data_file=input.txt \
    --save_dir=${OUT_DIR} \
    --num_layers=3 \
    --num_split=2 \
    --init_k=100 \
    --epoch=100 \
    --min_hold=5 \
    --in_epoch=50 \
    --lr=0.001 \
    --a=0.9 \
    --a0=1. \
    --a1=1. \
    --b=0.0001 > ${OUT_DIR}/log.txt
cp run.sh ${OUT_DIR}
cd ${OUT_DIR}
SCRIPT=${ROOT}/topic.py
python ${SCRIPT} --dict_file=wordmap.txt --layer=0 --u_file=U-0.txt > T-0.txt
python ${SCRIPT} --dict_file=wordmap.txt --layer=1 --u_file=U-1.txt > T-1.txt
python ${SCRIPT} --dict_file=wordmap.txt --layer=2 --u_file=U-2.txt > T-2.txt
