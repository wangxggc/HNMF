source config
source activate tensorflow

ROOT=`pwd`
time=`date +"%m%d%H%M"`
OUT_DIR=${ROOT}/output_${time}/
mkdir -p ${OUT_DIR}

CUDA_VISIBLE_DEVICES="0" python main_rerun.py \
    --data_file=input.txt \
    --break_point_dir=${ORI_DIR} \
    --save_dir=${OUT_DIR} \
    --num_layers=3 \
    --start_layer=1 \
    --num_split=2 \
    --init_k=100 \
    --epoch=100 \
    --min_hold=10 \
    --in_epoch=50 \
    --lr=0.002 \
    --a=0.9 \
    --a0=1. \
    --a1=1. \
    --b=0.0001 > ${OUT_DIR}/log.re.txt
cp run.sh ${OUT_DIR}
cd ${OUT_DIR}
SCRIPT=topic.py
python ${SCRIPT} --dict_file=wordmap.txt --layer=0 --u_file=U-0.txt > T-0.txt
python ${SCRIPT} --dict_file=wordmap.txt --layer=1 --u_file=U-1.txt > T-1.txt
python ${SCRIPT} --dict_file=wordmap.txt --layer=2 --u_file=U-2.txt > T-2.txt
