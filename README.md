# HNMF：HSOC
Hierarchical Topic Model Implement by NMF

### 环境

tensorflow11.0及以上

### 适用数据规模

30W文档 X 10W 词大概消耗 6GB左右内存，理论上1000W万文档*10W的词大概会在200G内存左右

### 更大规模数据

由于TF对稀疏矩阵的乘法有最大非零元素个数的限制，如果运行时出现类似警告，可以修改`model.py@D_mul_U` 将参数 `num_parts` 修改为更大的数字，比如从`10`修改为`100`

风险说明：修改`num_part`会使速度变慢一丢丢 



### 基本说明

基于非负矩阵分解，NMF实现的层次主题模型

[Liu R, Wang X, Wang D, et al. Topic splitting: a hierarchical topic model based on non-negative matrix factorization[J]. Journal of Systems Science and Systems Engineering, 2018, 27(4): 479-496.](<https://rd.springer.com/article/10.1007/s11518-018-5375-7>)

### 模型结构

$min {||D-U_{l+1}V_{l+1}||^2+\alpha||U_{l+1}^TU_{l+1}-I_{l+1}||^2 + \alpha_{0}||U_l^iV_l^i-U_{l+1}^{ii}V_{l+1}^{ii}||^2 + \alpha_{1} ||{U_{l+1}^{ii}}^TU_{l+1}^{ii}-I_{l+1}^{ii}||^2}+\beta||V_{l+1}||^{1}$

> 公式符号跟原论文可能存在不一致

### 运行  run.sh

```bash
python main.py \
    --data_file=$input_file \
    --save_dir=$output_dir \
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
    --b=0.0001> ${OUT_DIR}/log.txt
```

输出文件说明：

`docids.txt` doc_id，脚本会SHUFFLE数据，文件里记录了原始ID位置

`wordmap.txt` 词典，词ID即行号，从0开始

### 断点运行 run_from_break.sh

```bash
python main_rerun.py \
    --data_file=$input_file \
    --break_point_dir=$break_dir \
    --save_dir=$output_dir \
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
    --a1=1. > ${OUT_DIR}/log.re.txt
```

相对直接运行多了两个参数，分别是原模型目录`--break_point_dir`和开始层次`--start_layer`

### 断点运行说明

当数据量非常大时容易发生梯度爆炸或某层模型不收敛的情况，这时候可以从某个固定的层重跑，可以尝试重跑多次，直到模型收敛

### 查看Topic

`python topic.py --dict_file=wordmap.txt --layer=0 --u_file=U-0.txt > T-0.txt`

### 模型输入

输入格式与[<http://jgibblda.sourceforge.net/>](<http://jgibblda.sourceforge.net/>)一致

```text
#docNUM
Word1 Word2 ...
```
### Demo
![](https://raw.githubusercontent.com/wangxggc/HNMF/master/Demo1.tiff)
![](https://raw.githubusercontent.com/wangxggc/HNMF/master/Demo2.jpg)
